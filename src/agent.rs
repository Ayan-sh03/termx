use crate::llm_client::LlmClient;
use crate::session::Session;
use crate::tool_registry::ToolRegistry;
use crate::types::Message;
use crate::utils::{clip, display_diff_side_by_side};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashSet;
use std::io::{self, Write};
use tokio::time::{Duration, timeout};

#[async_trait]
pub trait LlmClientTrait {
    async fn chat_once(&self, messages: &[Message], tools: &Value) -> anyhow::Result<Message>;
    #[allow(dead_code)]
    async fn chat_once_no_stream(
        &self,
        messages: &[Message],
        // tools: &Value,
    ) -> anyhow::Result<Message>;
}

// Implement trait for real LlmClient
#[async_trait]
impl LlmClientTrait for LlmClient {
    async fn chat_once(&self, messages: &[Message], tools: &Value) -> anyhow::Result<Message> {
        self.chat_once(messages, tools).await
    }

    async fn chat_once_no_stream(&self, messages: &[Message]) -> anyhow::Result<Message> {
        self.chat_once_no_stream(messages).await
    }
}

#[derive(Clone)]
pub struct AgentOptions {
    pub max_steps: usize,
    pub yolo: bool, // auto-approve tools
    pub step_timeout: Duration,
    pub observation_clip: usize, // chars per tool output
}

pub struct Agent {
    llm: Box<dyn LlmClientTrait + Send + Sync>,
    tools: ToolRegistry,
    opts: AgentOptions,
}

impl Agent {
    #[allow(dead_code)]
    pub fn new(
        llm: Box<dyn LlmClientTrait + Send + Sync>,
        tools: ToolRegistry,
        opts: AgentOptions,
    ) -> Self {
        Self { llm, tools, opts }
    }

    // Convenience constructor for real LlmClient
    pub fn with_real_client(llm: LlmClient, tools: ToolRegistry, opts: AgentOptions) -> Self {
        Self {
            llm: Box::new(llm),
            tools,
            opts,
        }
    }

    // Compact older messages to keep context light. We do a simple heuristic:
    // - Keep the last N messages untouched.
    // - For older "tool" messages, clip content to a budget.
    pub fn compact_history(&self, session: &mut Session) {
        // Example heuristic: clip any tool message content longer than budget.
        for m in session.messages.iter_mut() {
            if m.role == "tool" {
                if let Some(c) = &m.content {
                    if c.len() > self.opts.observation_clip {
                        let head = &c[..self.opts.observation_clip];
                        m.content = Some(format!("{}… [truncated]", head));
                    }
                }
            }
        }
        // You can also drop very old messages if they exceed some count/size.
        let _ = session.save_session();
    }

    pub async fn run_turn(&self, session: &mut Session) -> anyhow::Result<Option<String>> {
        self.compact_history(session);

        // Single LLM step
        let llm_step = timeout(
            self.opts.step_timeout,
            self.llm.chat_once(&session.messages, self.tools.schemas()),
        )
        .await??;

        // Record assistant step
        session.add_message(llm_step.clone());

        if let Some(tcs) = &llm_step.tool_calls {
            for tc in tcs {
                let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                    Ok(v) => v,
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to parse tool arguments for '{}': {}. Raw arguments: {}",
                            tc.function.name, e, tc.function.arguments
                        );
                        // eprintln!("\u{001b}[91mWarning:\u{001b}[0m {}", error_msg);
                        return Err(anyhow::anyhow!(
                            "Tool argument parsing failed: {}",
                            error_msg
                        ));
                    }
                };

                println!("\n\u{001b}[35m▌🔧 {}\u{001b}[0m", tc.function.name);

                // Special handling for edit_file
                if tc.function.name == "edit_file" {
                    if let (Some(old_str), Some(new_str)) = (
                        args.get("old_str").and_then(|v| v.as_str()),
                        args.get("new_str").and_then(|v| v.as_str()),
                    ) {
                        display_diff_side_by_side(old_str, new_str);

                        if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                            println!("\u{001b}[90m   Path: {}\u{001b}[0m", path);
                        }
                    }
                } else if tc.function.name == "read_file" {
                    let path = args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(missing path)");
                    let start = args
                        .get("start_line")
                        .and_then(|v| v.as_u64())
                        .map(|n| n.to_string());
                    let end = args
                        .get("end_line")
                        .and_then(|v| v.as_u64())
                        .map(|n| n.to_string());

                    let range = match (start.as_deref(), end.as_deref()) {
                        (Some(s), Some(e)) => format!(" (lines {}-{})", s, e),
                        (Some(s), None) => format!(" (from line {})", s),
                        (None, Some(e)) => format!(" (through line {})", e),
                        (None, None) => String::new(),
                    };

                    println!("\u{001b}[90mRead File: {}{}\u{001b}[0m", path, range);
                } else {
                    // For other tools, show pretty JSON
                    let pretty_args = serde_json::to_string_pretty(&args)
                        .unwrap_or_else(|_| tc.function.arguments.clone());
                    println!("\u{001b}[90m{}\u{001b}[0m", pretty_args);
                }

                let _ = session.save_session();
            }
        }

        // If no tool calls: either final content or a no-op; return final if any
        if llm_step.tool_calls.is_none() {
            if let Some(text) = llm_step.content.clone() {
                let trimmed = text.trim().to_string();
                if !trimmed.is_empty() {
                    return Ok(Some(trimmed));
                }
            }
            return Ok(None);
        }

        // Tool calls present: execute them (in parallel if independent)
        let tool_calls = llm_step.tool_calls.clone().unwrap();

        // Collect files that have been requested to be read in the session so far.
        // This lets us block writes to files that haven't been read yet.
        let mut read_files_history: HashSet<String> = HashSet::new();
        for m in &session.messages {
            if let Some(calls) = &m.tool_calls {
                for call in calls {
                    if call.function.name == "read_file" {
                        if let Ok(v) = serde_json::from_str::<Value>(&call.function.arguments) {
                            if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
                                read_files_history.insert(p.to_string());
                            }
                        }
                    }
                }
            }
        }
        let mut tasks = vec![];

        for tool_call in tool_calls {
            let name = tool_call.function.name.clone();
            let id = tool_call.id.clone();
            let args_raw = tool_call.function.arguments.clone();
            let yolo = self.opts.yolo;
            let read_files_history = read_files_history.clone();

            tasks.push(tokio::spawn(async move {
                // Approval (synchronous user prompt) unless YOLO
                if !yolo && crate::tools::requires_approval(&name) {
                    let approval_prompt = crate::tools::format_tool_approval();
                    print!("{}", approval_prompt);
                    let _ = io::stdout().flush();

                    match crate::tools::get_user_approval("Proceed") {
                        Ok(true) => {
                            println!("\u{001b}[92m✓ Approved\u{001b}[0m");
                        }
                        Ok(false) => {
                            println!("\u{001b}[91m✗ Denied by user\u{001b}[0m");
                            return Ok::<(String, String), anyhow::Error>((
                                id,
                                "User denied execution".to_string(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("Approval error: {}", e);
                            return Ok::<(String, String), anyhow::Error>((
                                id,
                                format!("Error: {}", e),
                            ));
                        }
                    }
                }

                // Parse args safely
                let args: Value = match serde_json::from_str(&args_raw) {
                    Ok(v) => v,
                    Err(e) => {
                        let error_msg = format!(
                            "JSON parsing error for tool '{}': {}. Arguments received: {}",
                            name, e, args_raw
                        );
                        eprintln!("\u{001b}[91mError:\u{001b}[0m {}", error_msg);
                        return Ok::<(String, String), anyhow::Error>((
                            id,
                            format!("Failed to parse tool arguments: {}", e),
                        ));
                    }
                };

                // Dispatch
                let obs = match name.as_str() {
                    "list_dir" => {
                        let path = args["path"].as_str().unwrap_or(".");
                        let list = crate::tools::list_dir(path);
                        if list.is_empty() {
                            "Directory is empty".to_string()
                        } else {
                            list.join("\n")
                        }
                    }
                    "read_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        let start = args
                            .get("start_line")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as usize);
                        let end = args
                            .get("end_line")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as usize);
                        crate::tools::read_file(path, start, end)
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "write_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        if !path.is_empty() && !read_files_history.contains(path) {
                            // Enforce read-before-write policy and return feedback to the agent
                            format!(
                                "Policy: read the file before writing. Please call read_file on '{}' first.",
                                path
                            )
                        } else {
                            let content = args["content"].as_str().unwrap_or("");
                            crate::tools::write_file(path, content)
                                .unwrap_or_else(|e| format!("Error: {}", e))
                        }
                    }
                    "run_shell" => {
                        let cmd = args["command"].as_str().unwrap_or("");
                        crate::tools::run_shell(cmd).unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "search_in_files" => {
                        let path = args["path"].as_str().unwrap_or(".");
                        let case_sensitive = args.get("case_sensitive").and_then(|v| v.as_bool());
                        let pattern = args["pattern"].as_str().unwrap_or("");
                        crate::tools::search_in_files(pattern, path, case_sensitive)
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "edit_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        let old_str = args["old_str"].as_str().unwrap_or("");
                        let new_str = args["new_str"].as_str().unwrap_or("");
                        crate::tools::edit_file(path, old_str, new_str)
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "insert_in_file" => {
                        let path = args["path"].as_str().unwrap_or(".");
                        let content = args["content"].as_str().unwrap_or("");
                        let anchor = args["anchor"].as_str().unwrap_or("");
                        let position = args["position"].as_str().unwrap_or("");

                        crate::tools::insert_in_file(path, anchor, content, position)
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "ask_orackle" => {
                        let query = args["query"].as_str().unwrap_or("");
                        crate::tools::ask_orackle(query)
                            .await
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    _ => "Error: unknown tool".to_string(),
                };

                Ok::<(String, String), anyhow::Error>((id, obs))
            }));
        }

        // Gather results and append as tool messages (Observations)
        for t in tasks {
            match t.await {
                Ok(Ok((tool_call_id, observation))) => {
                    // Clip observation to keep context small
                    let clipped = clip(&observation, self.opts.observation_clip);
                    session.add_message(Message {
                        role: "tool".to_string(),
                        content: Some(clipped),
                        tool_calls: None,
                        tool_call_id: Some(tool_call_id),
                    });
                }
                Ok(Err(e)) => {
                    session.add_message(Message {
                        role: "tool".to_string(),
                        content: Some(format!("Error: {}", e)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Err(join_err) => {
                    session.add_message(Message {
                        role: "tool".to_string(),
                        content: Some(format!("Join error: {}", join_err)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }

        // After appending Observations, we do not return a final answer yet.
        // The caller will call run_turn again, which lets the LLM continue.
        Ok(None)
    }

    pub async fn run_agent_loop(
        &self,
        initial_user_input: String,
        session: &mut Session,
    ) -> anyhow::Result<()> {
        // Seed with user input
        session.add_message(Message {
            role: "user".into(),
            content: Some(initial_user_input),
            tool_calls: None,
            tool_call_id: None,
        });

        for step in 0..self.opts.max_steps {
            let final_text = self.run_turn(session).await?;
            if let Some(_output) = final_text {
                return Ok(());
            }
            // If run_turn returned None, it means tools were called and
            // Observations appended. Continue the loop to let LLM react.
            if step + 1 == self.opts.max_steps {
                // If we reach max steps without final text, summarize last turn
                println!("(Reached step limit without final answer.)");
            }
        }
        Ok(())
    }
}
