#[cfg(windows)]
pub mod detour;
pub mod hotpatch;
pub mod layout;
pub mod link;
pub mod rustc;
pub mod statics;

pub type Result<T> = anyhow::Result<T>;
