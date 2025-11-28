use std::process::{Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;
const TIMEOUT_SECONDS: u64 = 30;
const DENIED_COMMANDS: &[&str] = &["rm", "dd", "mkfs", ":(", "sudo", "su"];

pub fn run_shell(command: &str) -> Result<String, String> {
    // 1. Check denylist
    let mut parts = command.split_whitespace();
    let command_name = parts.next().ok_or("Empty command".to_string())?;

    if DENIED_COMMANDS.contains(&command_name) {
        return Err("Denied command".to_string());
    }

    // Show command being executed
    println!("\x1b[36m▌ Running: {}\x1b[0m", command);

    // 2. Spawn process (don't wait yet)
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn: {}", e))?;

    // 3. Wait with timeout
    let timeout = Duration::from_secs(TIMEOUT_SECONDS);
    match child
        .wait_timeout(timeout)
        .map_err(|e| format!("Wait error: {}", e))?
    {
        Some(status) => {
            // Process finished within timeout
            let output = child
                .wait_with_output()
                .map_err(|e| format!("Failed to get output: {}", e))?;

            let stdout_str = String::from_utf8_lossy(&output.stdout);
            let stderr_str = String::from_utf8_lossy(&output.stderr);

            // Show output in TUI with color coding
            if !stdout_str.is_empty() {
                let lines: Vec<&str> = stdout_str.lines().take(6).collect();
                for line in lines {
                    println!("\x1b[32m{}\x1b[0m", line);
                }
                if stdout_str.lines().count() > 6 {
                    println!("\x1b[90m... ({} more lines)\x1b[0m", stdout_str.lines().count() - 6);
                }
            }

            if !stderr_str.is_empty() {
                let lines: Vec<&str> = stderr_str.lines().take(3).collect();
                for line in lines {
                    println!("\x1b[31m{}\x1b[0m", line);
                }
                if stderr_str.lines().count() > 3 {
                    println!("\x1b[90m... ({} more error lines)\x1b[0m", stderr_str.lines().count() - 3);
                }
            }

            if status.success() {
                Ok(stdout_str.to_string())
            } else {
                Err(stderr_str.to_string())
            }
        }
        None => {
            // Timeout reached, kill the process
            println!("\x1b[33m⚠ Command timed out after {} seconds\x1b[0m", TIMEOUT_SECONDS);
            child.kill().map_err(|e| format!("Failed to kill: {}", e))?;
            Err(format!(
                "Command timed out after {} seconds",
                TIMEOUT_SECONDS
            ))
        }
    }
}
