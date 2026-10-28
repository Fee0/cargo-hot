//! Redirecting the old code of the running app to patched functions, on Windows x64.
//!
//! A patch only takes effect where the app calls through subsecond's jump table, in practice
//! the few entry points that iced wraps (`update`, `view`, ...). Everything the old code holds on
//! to keeps running the old code: function pointers, trait objects, callbacks, threads.
//!
//! So before the app switches to a patch, cargo-hot loads the patch into the app and overwrites
//! the entry of every old function that the patch replaces, in the executable and in earlier
//! patches, with a jump to the new one:
//!
//! ```text
//! E9 <rel32>    jmp new_function
//! ```
//!
//! The 32-bit offset reaches the patch, since patches are loaded next to the executable. The
//! app's threads are suspended while writing, and the write is retried while one of them is
//! executing the first bytes of a function being overwritten.

/// A jump from the entry of an old function to its new version.
#[derive(Debug, Clone, Copy)]
pub struct Detour {
    pub from: u64,
    pub to: u64,
    /// The base of the image `from` is in. Jumps in different images are written separately,
    /// since the memory between images may be unmapped, or hold data whose protection must not
    /// change.
    pub image: u64,
}

/// The size of a `jmp rel32`.
pub const JUMP_SIZE: u64 = 5;

#[cfg(windows)]
mod windows;

#[cfg(windows)]
pub use windows::{apply, load};
