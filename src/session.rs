use std::{fs::File, path::PathBuf};

use chrono::Utc;
use tokio::fs;
use uuid::Uuid;

use crate::types::Message;
pub use crate::types::Session;
impl Session {
    pub fn new(title: Option<&str>, model: Option<&str>) -> Session {
        Session {
            id: Uuid::new_v4().to_string(),
            messages: Vec::<Message>::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            title: title.map(|s| s.to_string()),
            model: model.map(|s| s.to_string()),
        }
    }

    // Append one message
    pub fn add_message(&mut self, msg: Message) {
        self.messages.push(msg);
        self.updated_at = Utc::now();
    }

    pub fn save_session(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.updated_at = Utc::now();
        let sessions_dir = PathBuf::from("sessions");
        let _ = fs::create_dir(&sessions_dir);

        let file_path = sessions_dir.join(format!("{}.json", self.id));
        let file = File::create(file_path)?;

        serde_json::to_writer_pretty(file, self)?;

        Ok(())
    }
}
