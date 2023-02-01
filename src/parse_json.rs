use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[derive(Serialize, Deserialize, Debug)]
pub struct Talk {
    #[serde(with = "time::serde::iso8601")]
    pub date: OffsetDateTime,
    pub discord_name: String,
    pub uuid: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TalkList(pub Vec<Talk>);

impl TalkList {
    pub fn read(json: &PathBuf) -> Self {
        let file = std::fs::File::open(json).expect("failed to open file");
        serde_json::from_reader(file).expect("failed to parse json")
    }
}
