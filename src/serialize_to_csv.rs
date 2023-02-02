use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Csv {
    speaker: String,
    date: String,
    talk: String,
}

impl Csv {
    pub fn new(speaker: String, date: String, talk: String) -> Self {
        Self {
            speaker,
            date,
            talk,
        }
    }
}
