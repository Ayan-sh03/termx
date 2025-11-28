mod agent;
mod llm_client;
mod session;
mod tool_registry;
mod tools;
mod types;
mod utils;

#[cfg(test)]
mod mocks;
#[cfg(test)]
mod tests;
use agent::{Agent, AgentOptions};
use chrono::Utc;
use llm_client::LlmClient;
use session::Session;
use std::env;
use std::io::{self, Write};
use std::process::Command;
use tool_registry::ToolRegistry;
use types::Message;

// ----------------------------------- Main -----------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    create_agent_dir();

    // ASCII Art Banner
    println!(
        r#"
        ███████████ ██████████ ███████████   ██████   ██████ █████ █████
       ░█░░░███░░░█░░███░░░░░█░░███░░░░░███ ░░██████ ██████ ░░███ ░░███
       ░   ░███  ░  ░███  █ ░  ░███    ░███  ░███░█████░███  ░░███ ███
           ░███     ░██████    ░██████████   ░███░░███ ░███   ░░█████
           ░███     ░███░░█    ░███░░░░░███  ░███ ░░░  ░███    ███░███
           ░███     ░███ ░   █ ░███    ░███  ░███      ░███   ███ ░░███
           █████    ██████████ █████   █████ █████     █████ █████ █████
          ░░░░░    ░░░░░░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░ ░░░░░
        "#
    );

    println!("\u{001b}[94mtermx - Advanced Coding Assistant\u{001b}[0m");
    println!(
        "\u{001b}[90mStarted at: {}\u{001b}[0m",
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );

    // Environment
    let base_url = env::var("OPENAI_BASE_URL").expect("OPENAI_BASE_URL not set");
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| {
        // Your original used "glm-4.5-air"; keep configurable
        "glm-4.6".to_string()
    });

    let llm = LlmClient::new(base_url, api_key, model.clone())?;
    let tools = ToolRegistry::new();
    let opts = AgentOptions {
        max_steps: 50,
        yolo: false, // set true to auto-approve tool calls
        step_timeout: tokio::time::Duration::from_secs(45),
        observation_clip: 4000, // keep large enough for code blocks
    };
    let agent = Agent::with_real_client(llm, tools, opts);

    // Create session with system message
    let mut session = Session::new(Some("Coding Session"), Some(&model));
    session.add_message(Message {
        role: "system".to_string(),
        content: Some(
            "You are an advanced coding assistant with expert-level reasoning capabilities.

        ## CORE PRINCIPLES
        1. **Think Before Acting**: Always analyze the task thoroughly before using tools
        2. **Plan & Decompose**: Break complex tasks into clear, sequential steps
        3. **Verify Results**: Double-check your work before presenting final answers
        4. **Learn & Adapt**: Use feedback to improve your approach
        5. **Be Efficient**: Use tools in parallel when possible, avoid redundant operations

        ## TASK EXECUTION STRATEGY
        1. **Understand**: Clarify the user's goal and constraints
        2. **Plan**: Outline the steps needed to complete the task
        3. **Execute**: Use tools systematically and efficiently
        4. **Verify**: Test and validate your implementation
        5. **Summarize**: Provide clear explanation of what was accomplished

        ## TOOL USAGE GUIDELINES
        - **read_file**: Gather context before making changes
        - **list_dir**: Understand project structure
        - **search_in_files**: Find relevant code patterns
        - **edit_file/insert_in_file**: Make precise, targeted changes
        - **write_file**: Create new files with proper structure
        - **run_shell**: Execute commands when necessary

        ## QUALITY STANDARDS
        - Never fabricate file contents or code
        - Ensure code is syntactically correct and follows conventions
        - Test your changes when possible
        - Provide clear explanations of your approach
        - Ask for clarification if the task is ambiguous

        ## COMMUNICATION STYLE
        - Be concise but thorough in your explanations
        - Show your reasoning process for complex tasks
        - Highlight important changes or decisions
        - Provide context for why certain approaches were chosen

        Remember: Your goal is to deliver high-quality, working solutions while being transparent about your process."
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
    });

    loop {
        print!("\u{001b}[93mYou:\u{001b}[0m ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            eprintln!("Failed to read input.");
            continue;
        }

        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("quit") {
            println!(
                r#"
{cyan}Session Summary:{reset}
{green}  Session ID:{reset}    {}
{green}  Total Messages:{reset} {}
{green}  Ended at:{reset}      {}
{cyan}Thank you for using termx! 🚀{reset}"#,
                session.id,
                session.messages.len(),
                Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                cyan = "\x1b[36m",
                green = "\x1b[32m",
                reset = "\x1b[0m"
            );
            break;
        } else if trimmed.eq_ignore_ascii_case("help") {
            println!(
                r#"
{cyan}Available Commands:{reset}
{green}  help{reset}     - Show this help message
{green}  clear{reset}    - Clear the terminal screen
{green}  quit{reset}     - Exit the program and show session summary
{green}  status{reset}   - Show current session information

{cyan}Usage:{reset}
Simply type your coding task or question as a natural language prompt.
The agent will use various tools to help you with your request."#,
                cyan = "\x1b[36m",
                green = "\x1b[32m",
                reset = "\x1b[0m"
            );
            continue;
        } else if trimmed.eq_ignore_ascii_case("clear") {
            // Clear terminal screen
            if cfg!(target_os = "windows") {
                Command::new("cmd").args(&["/C", "cls"]).status().ok();
            } else {
                Command::new("clear").status().ok();
            }
            continue;
        } else if trimmed.eq_ignore_ascii_case("status") {
            println!(
                r#"
{cyan}Session Status:{reset}
{green}  Session ID:{reset} {}
{green}  Model:{reset}     {}
{green}  Messages:{reset}  {}
{green}  Duration:{reset}  Since {}"#,
                session.id,
                session.model.as_deref().unwrap_or("default"),
                session.messages.len(),
                Utc::now().format("%H:%M:%S"),
                cyan = "\x1b[36m",
                green = "\x1b[32m",
                reset = "\x1b[0m"
            );
            continue;
        }

        print!("\u{001b}[96mAgent:\u{001b}[0m ");
        io::stdout().flush().unwrap();

        if let Err(e) = agent
            .run_agent_loop(trimmed.to_string(), &mut session)
            .await
        {
            eprintln!("\n\u{001b}[91mError:\u{001b}[0m {}", e);
            println!(
                "\n\u{001b}[96mAgent:\u{001b}[0m Something went wrong. Please try again or type 'help' for available commands."
            );
        } else {
            // Print newline to separate from next prompt
            println!();
        }
    }

    Ok(())
}

fn create_agent_dir() {
    if let Err(err) = std::fs::create_dir(".termx") {
        if err.kind() != std::io::ErrorKind::AlreadyExists {
            eprintln!("Error creating .termx: {}", err);
        }
    }
}
