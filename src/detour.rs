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
#![allow(unsafe_code)]

use anyhow::{Context, bail};

use std::ffi::c_void;
use std::path::Path;
use std::time::Duration;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, FlushInstructionCache, GetThreadContext, ReadProcessMemory,
    WriteProcessMemory,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAllocEx,
    VirtualFreeEx, VirtualProtectEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    EnumProcessModulesEx, GetModuleFileNameExW, LIST_MODULES_ALL,
};
use windows_sys::Win32::System::Threading::{
    CreateRemoteThread, OpenProcess, OpenThread, PROCESS_CREATE_THREAD, PROCESS_QUERY_INFORMATION,
    PROCESS_VM_OPERATION, PROCESS_VM_READ, PROCESS_VM_WRITE, ResumeThread, SuspendThread,
    THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME, WaitForSingleObject,
};

/// A jump from the entry of an old function to its new version.
#[derive(Debug, Clone, Copy)]
pub struct Detour {
    pub from: u64,
    pub to: u64,
}

/// The size of a `jmp rel32`.
pub const JUMP_SIZE: u64 = 5;

/// How long loading a patch into the app may take.
const LOAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Load `module` into the app, which must place it at `base`.
///
/// The app loads the patch itself once it gets it, and runs it right away, so loading it first
/// lets the detours be in place by then. `LoadLibraryW` runs in a thread of the app, and
/// `kernel32.dll`, which exports it, is at the same address in every process.
///
/// The app is the process with id `pid`.
pub fn load(pid: u32, module: &Path, base: u64) -> anyhow::Result<()> {
    let process = Process::open(pid)?;
    let process = process.0;

    let path: Vec<u16> = module
        .as_os_str()
        .to_string_lossy()
        .encode_utf16()
        .chain(Some(0))
        .collect();

    let size = path.len() * 2;
    let kernel32: Vec<u16> = "kernel32.dll".encode_utf16().chain(Some(0)).collect();

    // SAFETY: the remote buffer holds the path until the thread loading it has finished, and
    // `LoadLibraryW` has the signature of a thread routine on x64.
    unsafe {
        let load_library = GetProcAddress(
            GetModuleHandleW(kernel32.as_ptr()),
            c"LoadLibraryW".as_ptr().cast(),
        )
        .context("No LoadLibraryW")?;

        let remote = VirtualAllocEx(
            process,
            std::ptr::null(),
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );

        if remote.is_null() {
            bail!("Failed to allocate in the app: {}", last_error());
        }

        let result = (|| {
            let written = WriteProcessMemory(
                process,
                remote,
                path.as_ptr().cast(),
                size,
                std::ptr::null_mut(),
            );

            if written == 0 {
                bail!("Failed to write to the app: {}", last_error());
            }

            let thread = CreateRemoteThread(
                process,
                std::ptr::null(),
                0,
                Some(std::mem::transmute::<
                    unsafe extern "system" fn() -> isize,
                    unsafe extern "system" fn(*mut c_void) -> u32,
                >(load_library)),
                remote,
                0,
                std::ptr::null_mut(),
            );

            if thread.is_null() {
                bail!("Failed to start a thread in the app: {}", last_error());
            }

            let waited = WaitForSingleObject(thread, LOAD_TIMEOUT.as_millis() as u32);
            let _ = CloseHandle(thread);

            if waited != WAIT_OBJECT_0 {
                bail!("The app did not load the patch in time");
            }

            Ok(())
        })();

        let _ = VirtualFreeEx(process, remote, 0, MEM_RELEASE);
        result?;
    }

    let loaded = find_module(process, module)?.context("The app failed to load the patch")?;

    if loaded != base {
        bail!("The app loaded the patch at {loaded:#x} instead of {base:#x}");
    }

    Ok(())
}

/// Redirect old functions to the patch, which must be loaded.
pub fn apply(pid: u32, detours: &mut [Detour]) -> anyhow::Result<()> {
    let process = Process::open(pid)?;
    let process = process.0;

    detours.sort_unstable_by_key(|detour| detour.from);

    // The app's threads might be executing the first bytes of a function at any moment, but
    // rarely for long.
    for _ in 0..100 {
        let threads = Threads::suspend(pid)?;

        if threads.any_executing(detours)? {
            drop(threads);
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        let result = write(process, detours);
        drop(threads);

        // SAFETY: flushing the whole cache of a process we have a handle to.
        let _ = unsafe { FlushInstructionCache(process, std::ptr::null(), 0) };

        return result;
    }

    bail!("The app kept executing the functions to redirect")
}

/// A handle to the app's process, closed on drop.
struct Process(HANDLE);

impl Process {
    /// Open the process with id `pid`, which cargo-hot started, so it may write to it.
    fn open(pid: u32) -> anyhow::Result<Self> {
        // SAFETY: the handle is closed on drop.
        let handle = unsafe {
            OpenProcess(
                PROCESS_CREATE_THREAD
                    | PROCESS_QUERY_INFORMATION
                    | PROCESS_VM_OPERATION
                    | PROCESS_VM_READ
                    | PROCESS_VM_WRITE,
                0,
                pid,
            )
        };

        if handle.is_null() {
            bail!("Failed to open the app's process: {}", last_error());
        }

        Ok(Self(handle))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: we opened the handle.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Write the jumps, page range by page range.
fn write(process: HANDLE, detours: &[Detour]) -> anyhow::Result<()> {
    const PAGE: u64 = 4096;
    // Neighboring detours are written together, with the pages between them.
    const MAX_GAP: u64 = 16 * PAGE;

    let mut start = 0;

    while start < detours.len() {
        let first = detours[start].from & !(PAGE - 1);
        let mut end = start + 1;
        let mut last = detours[start].from + JUMP_SIZE;

        while end < detours.len() && detours[end].from <= last + MAX_GAP {
            last = last.max(detours[end].from + JUMP_SIZE);
            end += 1;
        }

        let last = last.next_multiple_of(PAGE);
        let mut buffer = vec![0u8; (last - first) as usize];

        let jumps = detours[start..end]
            .iter()
            .map(|detour| {
                let relative = detour.to as i64 - (detour.from + JUMP_SIZE) as i64;
                let relative = i32::try_from(relative)
                    .with_context(|| format!("{:#x} is out of reach", detour.to))?;

                Ok(((detour.from - first) as usize, relative))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut protection = 0;

        // SAFETY: the range covers code of the app's images, whose threads are suspended.
        unsafe {
            if VirtualProtectEx(
                process,
                first as *const c_void,
                buffer.len(),
                PAGE_EXECUTE_READWRITE,
                &mut protection,
            ) == 0
            {
                bail!("Failed to make {first:#x} writable: {}", last_error());
            }

            let read = ReadProcessMemory(
                process,
                first as *const c_void,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                std::ptr::null_mut(),
            );

            if read != 0 {
                for (offset, relative) in jumps {
                    buffer[offset] = 0xE9;
                    buffer[offset + 1..offset + 5].copy_from_slice(&relative.to_le_bytes());
                }

                let written = WriteProcessMemory(
                    process,
                    first as *const c_void,
                    buffer.as_ptr().cast(),
                    buffer.len(),
                    std::ptr::null_mut(),
                );

                if written == 0 {
                    let error = last_error();
                    restore(process, first, buffer.len(), protection);
                    bail!("Failed to write at {first:#x}: {error}");
                }
            }

            restore(process, first, buffer.len(), protection);

            if read == 0 {
                bail!("Failed to read at {first:#x}: {}", last_error());
            }
        }

        start = end;
    }

    Ok(())
}

/// Restore the protection of a range made writable.
fn restore(process: HANDLE, address: u64, size: usize, protection: u32) {
    let mut previous = 0;

    // SAFETY: the range was protected with this value before.
    let _ = unsafe {
        VirtualProtectEx(
            process,
            address as *const c_void,
            size,
            protection,
            &mut previous,
        )
    };
}

/// The base address of `module` in the app, if loaded.
fn find_module(process: HANDLE, module: &Path) -> anyhow::Result<Option<u64>> {
    let name = module
        .file_name()
        .context("No module name")?
        .to_string_lossy()
        .to_lowercase();

    let mut modules = vec![std::ptr::null_mut(); 4096];
    let mut needed = 0;

    // SAFETY: the buffers are as large as we say.
    let listed = unsafe {
        EnumProcessModulesEx(
            process,
            modules.as_mut_ptr(),
            (modules.len() * size_of::<*mut c_void>()) as u32,
            &mut needed,
            LIST_MODULES_ALL,
        )
    };

    if listed == 0 {
        bail!("Failed to list the modules of the app: {}", last_error());
    }

    modules.truncate(needed as usize / size_of::<*mut c_void>());

    for module in modules {
        let mut path = [0u16; 1024];

        // SAFETY: as above.
        let length =
            unsafe { GetModuleFileNameExW(process, module, path.as_mut_ptr(), path.len() as u32) };

        let path = String::from_utf16_lossy(&path[..length as usize]).to_lowercase();

        if path.ends_with(&name) {
            return Ok(Some(module as u64));
        }
    }

    Ok(None)
}

/// The suspended threads of a process, resumed on drop.
struct Threads(Vec<HANDLE>);

impl Threads {
    fn suspend(pid: u32) -> anyhow::Result<Self> {
        let mut threads = Self(Vec::new());

        // SAFETY: the snapshot is closed below, and every opened thread on drop.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);

            if snapshot == INVALID_HANDLE_VALUE {
                bail!("Failed to list the app's threads: {}", last_error());
            }

            let mut entry: THREADENTRY32 = std::mem::zeroed();
            entry.dwSize = size_of::<THREADENTRY32>() as u32;

            let mut more = Thread32First(snapshot, &mut entry) != 0;

            while more {
                if entry.th32OwnerProcessID == pid {
                    let thread = OpenThread(
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT,
                        0,
                        entry.th32ThreadID,
                    );

                    if !thread.is_null() {
                        if SuspendThread(thread) == u32::MAX {
                            let _ = CloseHandle(thread);
                        } else {
                            threads.0.push(thread);
                        }
                    }
                }

                more = Thread32Next(snapshot, &mut entry) != 0;
            }

            let _ = CloseHandle(snapshot);
        }

        Ok(threads)
    }

    /// Whether a thread is about to execute bytes that a detour overwrites.
    fn any_executing(&self, detours: &[Detour]) -> anyhow::Result<bool> {
        for thread in &self.0 {
            let mut context = Box::<CONTEXT>::new(
                // SAFETY: a CONTEXT is plain data.
                unsafe { std::mem::zeroed() },
            );
            context.ContextFlags = CONTEXT_CONTROL_AMD64;

            // SAFETY: the thread is suspended and the context aligned by its type.
            if unsafe { GetThreadContext(*thread, &mut *context) } == 0 {
                bail!("Failed to read a thread's state: {}", last_error());
            }

            let rip = context.Rip;
            let index = detours.partition_point(|detour| detour.from <= rip);

            if index > 0 && rip < detours[index - 1].from + JUMP_SIZE {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl Drop for Threads {
    fn drop(&mut self) {
        for thread in self.0.drain(..) {
            // SAFETY: we suspended and opened these threads.
            unsafe {
                let _ = ResumeThread(thread);
                let _ = CloseHandle(thread);
            }
        }
    }
}

fn last_error() -> std::io::Error {
    std::io::Error::last_os_error()
}
