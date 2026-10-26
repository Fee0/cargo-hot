use serde::{Deserialize, Serialize};
use std::{
    env::{args, vars},
    path::PathBuf,
};

/// The environment variable indicating where the args directory is located.
///
/// When the wrapper runs, it writes the arguments of each workspace crate to its own file in
/// this directory (see [`args_file_name`]), so hot patches can replay them.
pub const DX_RUSTC_WRAPPER_ENV_VAR: &str = "DX_RUSTC";

/// The environment variable holding the workspace root. Only crates inside of it are recorded.
pub const WORKSPACE_DIR_ENV_VAR: &str = "CARGO_HOT_WORKSPACE_DIR";

/// Is `dx` being used as a rustc wrapper?
///
/// This is primarily used to intercept cargo, enabling fast hot-patching by caching the environment
/// cargo setups up for the user's current project.
///
/// In a differenet world we could simply rely on cargo printing link args and the rustc command, but
/// it doesn't seem to output that in a reliable, parseable, cross-platform format (ie using command
/// files on windows...), so we're forced to do this interception nonsense.
pub fn is_wrapping_rustc() -> bool {
    std::env::var(DX_RUSTC_WRAPPER_ENV_VAR).is_ok()
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Args {
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
    pub link_args: Vec<String>,
}

impl Args {
    /// The value of a `--flag value` argument.
    pub fn value(&self, flag: &str) -> Option<&str> {
        self.args
            .iter()
            .position(|arg| arg == flag)
            .and_then(|index| self.args.get(index + 1))
            .map(String::as_str)
    }

    /// The value of a `-C name=value` codegen option, in any of its spellings.
    pub fn codegen_option(&self, name: &str) -> Option<&str> {
        let prefix = format!("{name}=");

        self.args.iter().enumerate().find_map(|(index, arg)| {
            let option = match arg.strip_prefix("-C") {
                Some("") => self.args.get(index + 1)?.as_str(),
                Some(option) => option,
                None => return None,
            };

            option.strip_prefix(&prefix)
        })
    }
}

/// The file a crate's arguments are recorded in: `{crate_name}.lib.json` or
/// `{crate_name}.bin.json`, so the library and the binary of a package do not overwrite
/// each other.
pub fn args_file_name(crate_name: &str, is_lib: bool) -> String {
    format!("{crate_name}.{}.json", if is_lib { "lib" } else { "bin" })
}

/// Check if the arguments indicate a linking step, including those in command files.
fn has_linking_args() -> bool {
    for arg in std::env::args() {
        // Direct check for linker-like arguments
        if arg.ends_with(".o") || arg == "-flavor" {
            return true;
        }

        // Check inside command files
        if let Some(path_str) = arg.strip_prefix('@')
            && let Ok(file_binary) = std::fs::read(path_str)
        {
            // Handle both UTF-8 and UTF-16LE encodings for response files.
            let content = String::from_utf8(file_binary.clone()).unwrap_or_else(|_| {
                let binary_u16le: Vec<u16> = file_binary
                    .chunks_exact(2)
                    .map(|a| u16::from_le_bytes([a[0], a[1]]))
                    .collect();
                String::from_utf16_lossy(&binary_u16le)
            });

            // Check if any line in the command file contains linking indicators.
            if content.lines().any(|line| {
                let trimmed_line = line.trim().trim_matches('"');
                trimmed_line.ends_with(".o") || trimmed_line == "-flavor"
            }) {
                return true;
            }
        }
    }

    false
}

/// Run rustc directly, but output the result to a file.
///
/// <https://doc.rust-lang.org/cargo/reference/config.html#buildrustc>
pub async fn run_rustc() {
    // If we are being asked to link, delegate to the linker action.
    if has_linking_args() {
        return crate::link::LinkAction::from_env()
            .expect("Linker action not found")
            .run_link()
            .await;
    }

    let args_dir: PathBuf = std::env::var(DX_RUSTC_WRAPPER_ENV_VAR)
        .expect("DX_RUSTC not set")
        .into();

    let mut rustc_args = Args {
        args: args().skip(1).collect::<Vec<_>>(),
        envs: vars().collect::<_>(),
        link_args: Default::default(),
    };

    // Cargo runs `rustc -vV` and the like with the wrapper too; those have no crate name. A
    // crate name of `___` is cargo probing the target, not building anything.
    let crate_name = rustc_args
        .value("--crate-name")
        .filter(|name| *name != "___")
        .map(str::to_owned);

    let is_workspace_member = std::env::var_os("CARGO_MANIFEST_DIR")
        .zip(std::env::var_os(WORKSPACE_DIR_ENV_VAR))
        .is_some_and(|(manifest_dir, workspace_dir)| {
            PathBuf::from(manifest_dir).starts_with(workspace_dir)
        });

    if let Some(crate_name) = crate_name
        && is_workspace_member
    {
        rustc_args
            .envs
            .retain_mut(|(key, _)| key != "CARGO_MAKEFLAGS");

        let is_lib = matches!(rustc_args.value("--crate-type"), Some("lib" | "rlib"));

        std::fs::create_dir_all(&args_dir).expect("Failed to create the rustc args directory");
        std::fs::write(
            args_dir.join(args_file_name(&crate_name, is_lib)),
            serde_json::to_string(&rustc_args).expect("Failed to serialize rustc args"),
        )
        .expect("Failed to write rustc args to file");
    }

    // Run the actual rustc command
    // We want all stdout/stderr to be inherited, so the running process can see the output
    //
    // Note that the args format we get from the wrapper includes the `rustc` command itself, so we
    // need to skip that - we already skipped the first arg when we created the args struct.
    let rustc = std::process::Command::new("rustc")
        .args(rustc_args.args.iter().skip(1))
        .envs(rustc_args.envs)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .current_dir(std::env::current_dir().expect("Failed to get current dir"))
        .status();

    // Propagate the exit code
    std::process::exit(rustc.unwrap().code().unwrap())
}
