use crate::agent::{Agent, AgentOptions};
use crate::mocks::mock_llm_client::MockLlmClient;
use crate::session::Session;
use crate::tool_registry::ToolRegistry;
use crate::types::Message;
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent() -> Agent {
        let mock_client = MockLlmClient::new();
        let tools = ToolRegistry::new();
        let opts = AgentOptions {
            max_steps: 20,
            yolo: true, // auto-approve for tests
            step_timeout: Duration::from_secs(10),
            observation_clip: 1000,
        };

        Agent::new(Box::new(mock_client), tools, opts)
    }

    fn create_test_agent_with_clip(clip: usize) -> Agent {
        let mock_client = MockLlmClient::new();
        let tools = ToolRegistry::new();
        let opts = AgentOptions {
            max_steps: 5,
            yolo: true, // auto-approve for tests
            step_timeout: Duration::from_secs(10),
            observation_clip: clip,
        };

        Agent::new(Box::new(mock_client), tools, opts)
    }

    #[tokio::test]
    async fn test_agent_compact_history() {
        let mut session = Session::new(None, None);
        let opts = AgentOptions {
            max_steps: 5,
            yolo: true,
            step_timeout: Duration::from_secs(10),
            observation_clip: 50, // Small clip for testing
        };

        // Add a long tool response (longer than 50 chars)
        session.add_message(Message {
            role: "tool".to_string(),
            content: Some("This is a very long tool response that definitely exceeds the observation clip limit of fifty characters and should be truncated".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Add a normal message
        session.add_message(Message {
            role: "user".to_string(),
            content: Some("Normal message".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Create agent and test compaction
        let agent = create_test_agent_with_clip(opts.observation_clip);

        // Test that tool messages get clipped
        agent.compact_history(&mut session);

        // Check that the tool message was clipped
        let tool_content = session.messages[0].content.as_ref().unwrap();
        if tool_content.len() > opts.observation_clip {
            assert!(tool_content.contains("… [truncated]"));
            assert!(tool_content.len() <= opts.observation_clip + "… [truncated]".len());
        }

        // Check that normal messages are not affected
        assert_eq!(
            session.messages[1].content,
            Some("Normal message".to_string())
        );
    }

    #[test]
    fn test_agent_options_creation() {
        let opts = AgentOptions {
            max_steps: 10,
            yolo: false,
            step_timeout: Duration::from_secs(30),
            observation_clip: 2000,
        };

        assert_eq!(opts.max_steps, 10);
        assert!(!opts.yolo);
        assert_eq!(opts.step_timeout, Duration::from_secs(30));
        assert_eq!(opts.observation_clip, 2000);
    }

    #[test]
    fn test_tool_registry_creation() {
        let tools = ToolRegistry::new();
        let schemas = tools.schemas();

        // Verify that we have the expected tools
        assert!(schemas.is_array());
        let tool_names: Vec<String> = schemas
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str())
            .map(|name| name.to_string())
            .collect();

        assert!(tool_names.contains(&"list_dir".to_string()));
        assert!(tool_names.contains(&"read_file".to_string()));
        assert!(tool_names.contains(&"write_file".to_string()));
        assert!(tool_names.contains(&"run_shell".to_string()));
        assert!(tool_names.contains(&"search_in_files".to_string()));
        assert!(tool_names.contains(&"edit_file".to_string()));
        assert!(tool_names.contains(&"insert_in_file".to_string()));
    }

    #[test]
    fn test_message_parsing() {
        // Test that we can parse tool call arguments correctly
        let tool_args = r#"{"path": "/test/path", "content": "test content"}"#;
        let parsed: serde_json::Value = serde_json::from_str(tool_args).unwrap();

        assert_eq!(parsed["path"], "/test/path");
        assert_eq!(parsed["content"], "test content");
    }

    #[tokio::test]
    async fn test_agent_with_text_response() {
        let mut session = Session::new(None, None);

        // Add system message
        session.add_message(Message {
            role: "system".to_string(),
            content: Some("You are a helpful assistant".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Create agent with mock that returns text
        let mut mock_client = MockLlmClient::new();
        mock_client.add_text_response("Hello! How can I help you today?");

        let agent = Agent::new(
            Box::new(mock_client),
            ToolRegistry::new(),
            AgentOptions {
                max_steps: 5,
                yolo: true,
                step_timeout: Duration::from_secs(10),
                observation_clip: 1000,
            },
        );

        // Run a turn
        let result = agent.run_turn(&mut session).await.unwrap();

        // Should get Some(String) for text response
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "Hello! How can I help you today?");
    }

    #[tokio::test]
    async fn test_agent_with_tool_call() {
        let mut session = Session::new(None, None);

        // Add system message
        session.add_message(Message {
            role: "system".to_string(),
            content: Some("You are a helpful assistant".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Add user message requesting a tool call
        session.add_message(Message {
            role: "user".to_string(),
            content: Some("List the current directory".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Create agent with mock that returns a tool call
        let mut mock_client = MockLlmClient::new();
        mock_client.add_tool_call_response("list_dir", r#"{"path": "."}"#);

        let agent = Agent::new(
            Box::new(mock_client),
            ToolRegistry::new(),
            AgentOptions {
                max_steps: 5,
                yolo: true,
                step_timeout: Duration::from_secs(10),
                observation_clip: 1000,
            },
        );

        // Run a turn - should return None for tool call (needs another turn)
        let result = agent.run_turn(&mut session).await.unwrap();

        // Should get None for tool call (needs another turn to process tool result)
        assert!(result.is_none());

        // Verify that tool call was added to session
        // The agent should have added the assistant message with tool calls
        let assistant_message = &session
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .unwrap();
        assert!(assistant_message.tool_calls.is_some());

        let tool_calls = assistant_message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "list_dir");
    }

    #[test]
    fn test_session_with_agent_workflow() {
        let mut session = Session::new(Some("Test Session"), Some("test-model"));

        // Simulate a simple workflow
        session.add_message(Message {
            role: "system".to_string(),
            content: Some("You are a helpful assistant".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        session.add_message(Message {
            role: "user".to_string(),
            content: Some("List the current directory".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Verify session state
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "system");
        assert_eq!(session.messages[1].role, "user");
        assert_eq!(session.title, Some("Test Session".to_string()));
        assert_eq!(session.model, Some("test-model".to_string()));
    }
}
