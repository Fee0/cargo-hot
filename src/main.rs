use cargo_hot::Result;
use cargo_hot::hotpatch;
use cargo_hot::layout;
use cargo_hot::link::{self, LinkerFlavor};
use cargo_hot::rustc;
use cargo_hot::statics;
use cargo_hot_protocol::server;

use cargo::GlobalContext;
use cargo::core::dependency::DepKind;
use cargo::core::{Package, Target, TargetKind, Workspace};
use cargo::util::{Filesystem, command_prelude::*};

use anyhow::{Context, anyhow};
use itertools::Itertools;
use serde::Deserialize;
use subsecond_types::JumpTable;
use target_lexicon::{Architecture, OperatingSystem, Triple};
use tokio::process;
use tokio::sync::mpsc;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

#[tokio::main]
async fn main() -> Result<ExitCode> {
    if rustc::is_wrapping_rustc() {
        rustc::run_rustc().await;
        return Ok(ExitCode::SUCCESS);
    }

    let gctx = GlobalContext::default()?;
    let _token = cargo::util::job::setup();

    let matches = Command::new("cargo")
        .bin_name("cargo")
        .subcommand(command())
        .try_get_matches()?;

    let (_, args) = matches
        .subcommand()
        .ok_or(anyhow!("`cargo-hot` must be called with `hot` subcommand"))?;

    init_logger(args.get_count("verbose"));

    let server = Server::new(gctx, args).await?;

    server.run().await
}

/// Errors are always shown. `-v` adds debug logs and `-vv` adds trace logs.
///
/// Logs go to stderr, and the logs of the `cargo` library itself are kept at warnings
/// since they are very noisy.
fn init_logger(verbose: u8) {
    use tracing_subscriber::filter::{LevelFilter, Targets};
    use tracing_subscriber::prelude::*;

    let level = match verbose {
        0 => LevelFilter::INFO,
        1 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    };

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(
            Targets::new()
                .with_target("cargo_hot", level)
                .with_default(LevelFilter::WARN),
        )
        .init();
}

fn command() -> Command {
    subcommand("hot")
        .about("Run a binary or example of the local package in hot reloading mode")
        .arg(
            Arg::new("args")
                .value_name("ARGS")
                .help("Arguments for the binary or example to run")
                .value_parser(value_parser!(OsString))
                .num_args(0..)
                .trailing_var_arg(true),
        )
        .arg_message_format()
        .arg_silent_suggestion()
        .arg_package("Package with the target to run")
        .arg_targets_bin_example(
            "Name of the bin target to run",
            "Name of the example target to run",
        )
        .arg_features()
        .arg_parallel()
        .arg_release("Build artifacts in release mode, with optimizations")
        .arg_profile("Build artifacts with the specified profile")
        .arg_target_triple("Build for the target triple")
        .arg_target_dir()
        .arg_manifest_path()
        .arg_lockfile_path()
        .arg_ignore_rust_version()
        .arg_unit_graph()
        .arg(flag(
            "auto-restart",
            "Rebuild and restart the app when it crashes, instead of exiting",
        ))
        .arg(flag(
            "no-redirect",
            "Only run patched code where the app calls through its hot reloading entry points \
            (like iced's `update` and `view`), instead of redirecting all old code to it",
        ))
        .arg(flag(
            "full-debuginfo",
            "Keep the profile's debug info for the hot reloaded crate \
            (by default it is reduced to line tables, for faster patches)",
        ))
        .arg_timings()
        .arg(
            opt(
                "verbose",
                "Use verbose output (-vv very verbose/build.rs output)",
            )
            .short('v')
            .action(ArgAction::Count)
            .global(true),
        )
        .after_help(color_print::cstr!(
            "Run `<cyan,bold>cargo help run</>` for more detailed information.\n"
        ))
}

/// The size of the jump that redirects a function.
const JUMP_SIZE: u64 = 5;

/// Windows maps images at multiples of 64 KiB.
const PATCH_ALIGNMENT: u64 = 64 << 10;

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// With `--auto-restart`, a crash this soon after starting is not restarted right away,
/// since the new app would likely crash the same way. The next change restarts it.
const CRASH_LOOP_WINDOW: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub struct Server {
    gctx: GlobalContext,
    sysroot: PathBuf,
    crate_target: Target,
    crate_dir: PathBuf,
    /// Source directory of the target being run. Changes here are hotpatched.
    crate_src_dir: PathBuf,
    /// The workspace libraries the target depends on, dependencies first. Their changes are
    /// hotpatched too, by recompiling them and every crate between them and the target.
    workspace_crates: Vec<WorkspaceCrate>,
    workspace_dir: PathBuf,
    profile: String,
    triple: Triple,
    package: String,
    features: Vec<String>,
    exe_args: Vec<OsString>,
    extra_cargo_args: Vec<String>,
    extra_rustc_args: Vec<String>,
    verbose: u8,
    auto_restart: bool,
    redirect: bool,
    full_debuginfo: bool,
    no_default_features: bool,
    target_dir: Filesystem,
    custom_linker: Option<PathBuf>,
}

/// A library of the workspace that the target depends on.
#[derive(Debug)]
struct WorkspaceCrate {
    package: String,
    crate_name: String,
    src_dir: PathBuf,
    is_proc_macro: bool,
    /// Crate names of the workspace libraries this one depends on directly.
    dependencies: Vec<String>,
}

/// Workspace libraries changed since the app was started, which every patch has to carry.
#[derive(Debug, Default)]
struct PatchState {
    /// Recompiled since the app was started. A patch only replaces the functions it contains,
    /// so leaving one of these out would bring its old code back.
    modified: BTreeSet<String>,
    /// Must be recompiled before the next patch, because their last compile failed or one of
    /// their dependencies changed since.
    pending: BTreeSet<String>,
    /// Where the next patch is loaded, on Windows (see [`Server::patch_base`]).
    next_patch_base: Option<u64>,
    /// The patches loaded by the app, whose functions are redirected to newer patches too.
    patches: Vec<LoadedPatch>,
}

/// A patch loaded by the app.
#[derive(Debug)]
struct LoadedPatch {
    base: u64,
    functions: HashMap<String, hotpatch::Function>,
}

/// A patch, ready to be sent to the app.
struct Patch {
    jump_table: JumpTable,
    /// The jumps from old functions to the patched ones, once the app has loaded the patch.
    detours: Vec<Detour>,
}

/// A jump from the entry of an old function to its new version (see `cargo_hot::detour`).
#[derive(Debug, Clone, Copy)]
struct Detour {
    from: u64,
    to: u64,
}

#[derive(Clone, Debug)]
pub struct Build {
    exe: PathBuf,
    is_fresh: bool,
    direct_rustc: rustc::Args,
    time_start: SystemTime,
    patch_cache: Option<Arc<hotpatch::Cache>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuildMode {
    Fat,
    Thin {
        rustc_args: rustc::Args,
        /// Recompiled workspace libraries to link into the patch.
        workspace_rlibs: Vec<PathBuf>,
        /// The address the patch must be loaded at, if any.
        patch_base: Option<u64>,
        changed_files: Vec<PathBuf>,
        aslr_reference: u64,
        cache: Arc<hotpatch::Cache>,
    },
}

impl Server {
    async fn new(gctx: GlobalContext, args: &ArgMatches) -> Result<Self> {
        let sysroot = process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .await
            .map(|out| String::from_utf8(out.stdout).map(|s| s.trim().to_string()))?
            .context("Failed to extract rustc sysroot output")?;

        let target_kind = if args.contains_id("example") {
            TargetKind::ExampleBin
        } else {
            TargetKind::Bin
        };

        let workspace = args.workspace(&gctx)?;

        let compile_opts = args.compile_options(
            &gctx,
            CompileMode::Build,
            Some(&workspace),
            ProfileChecking::Custom,
        )?;

        let packages = compile_opts.spec.get_packages(&workspace)?;
        let main_package = packages.first().unwrap();

        let target_name = args
            .get_one("example")
            .cloned()
            .or(args.get_one("bin").cloned())
            .or_else(|| {
                if let Some(default_run) = &main_package.manifest().default_run() {
                    return Some(default_run.to_string());
                }

                let bin_count = packages
                    .iter()
                    .flat_map(|packages| packages.targets())
                    .filter(|target| target.kind() == &target_kind)
                    .count();

                if bin_count != 1 {
                    return None;
                }

                main_package.targets().iter().find_map(|x| {
                    if x.kind() == &target_kind {
                        Some(x.name().to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(main_package.name().to_string());

        let crate_target = main_package
            .targets()
            .iter()
            .find(|target| {
                target_name == target.name() && target.kind() == &target_kind
            })
            .with_context(|| {
                let target_of_kind = |kind|-> String {
                    let filtered_packages = main_package
                .targets()
                .iter()
                .filter_map(|target| {
                    (target.kind() == kind).then_some(target.name().to_string())
                }).collect::<Vec<_>>();

                filtered_packages.join(", ")};

                if let Some(example) = &args.get_one::<String>("example"){
                    let examples = target_of_kind(&TargetKind::ExampleBin);
                    format!("Failed to find example {example}. \nAvailable examples are:\n{examples}")
                } else if let Some(bin) = &args.get_one::<String>("bin") {
                    let binaries = target_of_kind(&TargetKind::Bin);
                    format!("Failed to find binary {bin}. \nAvailable binaries are:\n{binaries}")
                } else {
                    format!("Failed to find target {target_name}. \nIt looks like you are trying to build a library crate. \
                    You either need to run `cargo hot` from inside a binary crate or build a specific example with the `--example` flag. \
                    Available examples are:\n{}", target_of_kind(&TargetKind::ExampleBin))
                }
            })?
            .clone();

        let profile = match args.get_one::<String>("profile") {
            Some(profile) => profile.to_owned(),
            None if args.flag("release") => "release".to_string(),
            None => "dev".to_string(),
        };

        let triple = match args.get_one::<String>("target") {
            Some(target) => target.parse().expect("parse target"),
            None => target_lexicon::HOST,
        };

        // Determine the --package we'll pass to cargo.
        // todo: I think this might be wrong - we don't want to use main_package necessarily...
        let package = args
            .get_one("package")
            .cloned()
            .unwrap_or_else(|| main_package.name().to_string());

        let cargo_config = cargo_config2::Config::load().unwrap();

        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| cargo_config.build.target_dir.clone())
            .map(Filesystem::new)
            .unwrap_or_else(|| workspace.target_dir());

        let mut custom_linker = cargo_config.linker(triple.to_string())?;
        if let Some(linker) = custom_linker.as_ref()
            && (linker == "rust-lld" || linker == "rust-lld.exe")
            && cfg!(windows)
        {
            // When using "rust-lld.exe" as linker on windows, it still needs to have a flavor
            // given to it. rustc appears to be passing `-flavor "link"` when none is set by the
            // user. If no flavor is given, it fails with 'lld is a generic driver'.
            // We already use the existing lld-link by default on windows, so we can simply set the
            // `custom_linker` to `None` in these cases, since we end up using "lld-link" anyway
            // which is the same as "rust-lld.exe -flavor link".
            custom_linker = None;
        }

        let exe_args = args
            .get_many("args")
            .into_iter()
            .flatten()
            .cloned()
            .collect();

        let extra_cargo_args = vec![]; // TODO

        // TODO
        let mut extra_rustc_args = cargo_config
            .rustflags(triple.to_string())
            .unwrap_or_default()
            .unwrap_or_default()
            .flags;

        // Full debug info makes the linker write a PDB of hundreds of megabytes on every patch.
        // Line tables keep file and line numbers in backtraces, and make patches ~40% faster.
        // These args only reach the hot reloaded crate, and patches reuse them.
        if !args.flag("full-debuginfo") {
            extra_rustc_args.push("-Cdebuginfo=line-tables-only".to_owned());
        }

        let crate_dir = main_package.manifest_path().parent().unwrap().to_path_buf();
        let crate_src_dir = src_dir(&crate_target).context("Get source path")?;
        let workspace_crates = workspace_crates(&workspace, main_package);
        let workspace_dir = workspace.root_manifest().parent().unwrap().to_path_buf();

        // Redirecting writes x64 jumps into the app.
        let redirect = !args.flag("no-redirect")
            && triple.operating_system == OperatingSystem::Windows
            && triple.architecture == Architecture::X86_64;

        Ok(Self {
            gctx,
            sysroot: PathBuf::from(sysroot),
            crate_target,
            crate_dir,
            crate_src_dir,
            workspace_crates,
            workspace_dir,
            profile,
            triple,
            package,
            features: args
                .get_many("features")
                .map(|features| features.into_iter().cloned().collect())
                .unwrap_or_default(),
            exe_args,
            extra_cargo_args,
            extra_rustc_args,
            verbose: args.get_count("verbose"),
            auto_restart: args.flag("auto-restart"),
            redirect,
            full_debuginfo: args.flag("full-debuginfo"),
            no_default_features: args.get_flag("no-default-features"),
            target_dir,
            custom_linker,
        })
    }

    async fn run(self) -> Result<ExitCode> {
        use notify::Watcher;

        std::fs::create_dir_all(self.exe_dir())?;

        for file in [
            self.link_args_file(),
            self.link_err_file(),
            self.windows_command_file(),
        ] {
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(file)?;
        }

        let (sender, mut receiver) = mpsc::channel(1);

        let handler = move |event: notify::Result<notify::Event>| {
            let Ok(event) = event else {
                return;
            };

            let is_allowed_notify_event = match event.kind {
                notify::EventKind::Modify(notify::event::ModifyKind::Data(_)) => true,
                notify::EventKind::Modify(notify::event::ModifyKind::Name(_)) => true,
                // The primary modification event on WSL's poll watcher.
                notify::EventKind::Modify(notify::event::ModifyKind::Metadata(
                    notify::event::MetadataKind::WriteTime,
                )) => true,
                // Catch-all for unknown event types (windows)
                notify::EventKind::Modify(notify::event::ModifyKind::Any) => true,
                notify::EventKind::Modify(notify::event::ModifyKind::Metadata(_)) => false,
                // Don't care about anything else.
                notify::EventKind::Create(_) => true,
                notify::EventKind::Remove(_) => true,
                _ => false,
            };

            if is_allowed_notify_event {
                let _ = sender.blocking_send(event);
            }
        };

        let mut watcher: Box<dyn Watcher> = if is_wsl() {
            // On wsl, we need to poll the filesystem for changes
            Box::new(notify::PollWatcher::new(
                handler,
                notify::Config::default().with_poll_interval(Duration::from_secs(2)),
            )?)
        } else {
            // Otherwise we can use the recommended watcher
            Box::new(notify::recommended_watcher(handler)?)
        };

        let mut connections = accept_connections(server::Server::bind().await?);

        // Read before building, so they are what the app is built from.
        let mut sources = layout::Sources::read(self.source_dirs());

        let mut build = self.build(BuildMode::Fat).await?;
        let mut executable = self.spawn_executable()?;

        for dir in self.source_dirs() {
            watcher.watch(dir, notify::RecursiveMode::Recursive)?;
        }

        let mut connection = None;
        let mut patch_state = PatchState::default();
        let mut needs_restart = false;
        let mut crashed = false;
        let mut started = Instant::now();
        let mut buffer = Vec::new();

        loop {
            let changed_files: BTreeSet<PathBuf> = tokio::select! {
                n = read_batch(&mut receiver, &mut buffer, Duration::from_millis(100)) => {
                    if n == 0 {
                        return Err(anyhow!("file notifier failed"));
                    }

                    let changed_files: BTreeSet<PathBuf> = buffer
                        .drain(..n)
                        .filter_map(|event| {
                            let path = event.paths.first()?;

                            if path != &path.with_extension("rs") {
                                return None;
                            }

                            Some(event.paths)
                        })
                        .flatten()
                        .collect();

                    if changed_files.is_empty() {
                        continue;
                    }

                    changed_files
                }
                Some(new_connection) = connections.recv() => {
                    connection = Some(new_connection);
                    continue;
                }
                status = executable.wait(), if !crashed => {
                    let status = status?;

                    if !status.success() {
                        log::error!("The app exited with {status}");
                    }

                    if !self.auto_restart || status.success() {
                        return Ok(status.code().map(|code| ExitCode::from(code as u8)).unwrap_or_default());
                    }

                    crashed = true;
                    connection = None;

                    // A crash right after starting would crash again on restart.
                    if started.elapsed() < CRASH_LOOP_WINDOW {
                        log::error!(
                            "The app crashed right after starting ({status}). \
                            Waiting for a change to restart it."
                        );
                        continue;
                    }

                    log::warn!("The app crashed ({status}). Restarting.");
                    BTreeSet::new()
                }
            };

            // Until a restart succeeds, patching would build against libraries the running
            // app does not have.
            let changed_crates = self.changed_crates(&changed_files);
            needs_restart = needs_restart || crashed || changed_crates.is_none();

            let layout_start = Instant::now();
            let changed_definition = (!needs_restart)
                .then(|| sources.changed_definition(&changed_files))
                .flatten();
            log::debug!(
                "Checked the changed definitions in {:.2?}",
                layout_start.elapsed()
            );

            if let Some(definition) = changed_definition {
                log::info!(
                    "Restarting, since patched code would read the app's data with the \
                    layout of the new {definition}"
                );
                needs_restart = true;
            }

            let changed_sources = layout::Sources::read_files(&changed_files);

            let start = Instant::now();

            if needs_restart {
                let _ = self.gctx.shell().status(
                    "Rebuilding",
                    format!(
                        "{} ({})",
                        self.crate_target.name(),
                        self.crate_dir.display(),
                    ),
                );

                let new_sources = layout::Sources::read(self.source_dirs());

                match self.restart(&mut executable).await {
                    Ok(Some(new_build)) => {
                        sources = new_sources;
                        build = new_build;
                        connection = None;
                        patch_state = PatchState::default();
                        started = Instant::now();
                    }
                    // Nothing changed since the last build, so the app on disk is current.
                    Ok(None) if crashed => {
                        executable = self.spawn_executable()?;
                        started = Instant::now();
                    }
                    Ok(None) => {}
                    Err(error) if crashed => {
                        log::error!("Rebuild failed, waiting for a change to retry: {error:#}");
                        continue;
                    }
                    Err(error) => {
                        log::error!(
                            "Rebuild failed, the app keeps running the old code: {error:#}"
                        );
                        continue;
                    }
                }

                crashed = false;
                needs_restart = false;
                self.status_finished(start);
                continue;
            }

            let Some(active_connection) = connection.as_mut() else {
                log::warn!(
                    "Skipping patch: the app is not connected. \
                    Is it built with hot reloading enabled (e.g. iced's `hot` feature)?"
                );
                continue;
            };

            let _ = self.gctx.shell().status(
                "Patching",
                format!(
                    "{} ({})",
                    self.crate_target.name(),
                    self.crate_dir.display(),
                ),
            );

            let patch = match self
                .patch(
                    &build,
                    changed_files,
                    changed_crates.unwrap_or_default(),
                    &mut patch_state,
                    active_connection.aslr_reference() as u64,
                )
                .await
            {
                Ok(patch) => patch,
                Err(error) => {
                    log::error!("Patch failed: {error:#}");
                    continue;
                }
            };

            if let Some(base) = patch_state.patches.last().map(|patch| patch.base)
                && !patch.detours.is_empty()
                && let Err(error) = self
                    .redirect(
                        &executable,
                        patch.jump_table.lib.clone(),
                        base,
                        patch.detours,
                    )
                    .await
            {
                // A patch loaded elsewhere reads garbage instead of the app's data.
                log::error!("Restarting, since the patch could not be loaded: {error:#}");
                needs_restart = true;
                continue;
            }

            match active_connection.patch(&patch.jump_table).await {
                Ok(()) => {
                    sources.remember(changed_sources);
                    self.status_finished(start);
                }
                Err(error) => {
                    log::error!("Lost connection to the app: {error:#}");
                    connection = None;
                }
            }
        }
    }

    /// The source directories of the target and of the workspace libraries it depends on.
    fn source_dirs(&self) -> impl Iterator<Item = &Path> {
        std::iter::once(self.crate_src_dir.as_path()).chain(
            self.workspace_crates
                .iter()
                .map(|krate| krate.src_dir.as_path()),
        )
    }

    /// The workspace libraries the changed files belong to, or `None` if a change can only be
    /// applied with a restart. Changes of the target's own sources need no library.
    fn changed_crates(&self, changed_files: &BTreeSet<PathBuf>) -> Option<BTreeSet<String>> {
        let mut changed_crates = BTreeSet::new();

        for file in changed_files {
            // Source directories can nest, so the innermost one owns the file.
            let owner = self
                .workspace_crates
                .iter()
                .filter(|krate| file.starts_with(&krate.src_dir))
                .max_by_key(|krate| krate.src_dir.components().count());

            let in_target = file.starts_with(&self.crate_src_dir);

            let Some(krate) = owner else {
                if in_target {
                    continue;
                }

                log::debug!("{} is not part of a crate the app uses", file.display());
                return None;
            };

            // A library and the target can share their source directory.
            if in_target
                && self.crate_src_dir.components().count() > krate.src_dir.components().count()
            {
                continue;
            }

            // A proc macro runs inside of the compiler, not the app.
            if krate.is_proc_macro {
                log::debug!("{} changed, which is a proc macro", krate.crate_name);
                return None;
            }

            if !self.rustc_args_file(&krate.crate_name, true).exists() {
                log::warn!(
                    "Restarting, since the arguments to compile {} were not recorded",
                    krate.crate_name
                );
                return None;
            }

            let _ = changed_crates.insert(krate.crate_name.clone());
        }

        Some(changed_crates)
    }

    fn status_finished(&self, start: Instant) {
        let _ = self.gctx.shell().status(
            "Finished",
            format!(
                "`{}` profile target(s) in {:.2}s",
                self.profile,
                start.elapsed().as_millis() as f32 / 1_000.0
            ),
        );
    }

    /// Remove the files of earlier patches that `should_remove` picks.
    ///
    /// A patch is never needed again once the app that loaded it is gone, and its PDB alone can
    /// be hundreds of megabytes.
    fn remove_patch_files(&self, should_remove: impl Fn(&Path) -> bool) {
        let prefix = format!("lib{}-patch-", self.executable_name());

        let Ok(entries) = std::fs::read_dir(self.exe_dir()) else {
            return;
        };

        for path in entries.flatten().map(|entry| entry.path()) {
            let is_patch = path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with(&prefix));

            if is_patch && should_remove(&path) {
                _ = std::fs::remove_file(path);
            }
        }
    }

    /// `-Csave-temps` keeps the LLVM bitcode of every build next to its object files. Nothing
    /// reads it, and it adds up to gigabytes over a session.
    fn remove_saved_bitcode(&self, object_dir: &Path) {
        let prefix = format!("{}.", self.crate_target.crate_name());

        let Ok(entries) = std::fs::read_dir(object_dir) else {
            return;
        };

        for path in entries.flatten().map(|entry| entry.path()) {
            let is_bitcode = path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.starts_with(&prefix) && name.ends_with(".bc")
            });

            if is_bitcode {
                _ = std::fs::remove_file(path);
            }
        }
    }

    fn spawn_executable(&self) -> Result<process::Child> {
        // None of our apps is running, so no patch is loaded anymore.
        self.remove_patch_files(|_| true);

        Ok(process::Command::new(self.main_exe())
            .args(&self.exe_args)
            .kill_on_drop(true)
            .spawn()?)
    }

    /// Run a full build and, if anything changed, replace the running app with the new one.
    ///
    /// The old app keeps running while compiling, so a failed build leaves it untouched.
    async fn restart(&self, executable: &mut process::Child) -> Result<Option<Build>> {
        let mut build = self.cargo_build(&BuildMode::Fat).await?;

        if build.is_fresh {
            return Ok(None);
        }

        executable
            .kill()
            .await
            .context("Failed to stop the running app")?;

        self.write_executable(&build.exe)
            .await
            .context("Failed to write main executable")?;

        build.patch_cache = Some(Arc::new(self.create_patch_cache(&build.exe).await?));
        *executable = self.spawn_executable()?;

        Ok(Some(build))
    }

    async fn build(&self, mode: BuildMode) -> Result<Build> {
        // Run the cargo build to produce our artifacts
        let compile_start = Instant::now();
        let mut build = self.cargo_build(&mode).await?;
        log::debug!("Compiled in {:.2?}", compile_start.elapsed());

        // Write the build artifacts to the bundle on the disk
        match &mode {
            BuildMode::Thin {
                aslr_reference,
                cache,
                rustc_args,
                workspace_rlibs,
                patch_base,
                ..
            } => {
                let link_start = Instant::now();
                self.write_patch(
                    *aslr_reference,
                    &mut build,
                    cache,
                    rustc_args,
                    workspace_rlibs,
                    *patch_base,
                )
                .await?;
                log::debug!("Linked patch in {:.2?}", link_start.elapsed());
            }

            BuildMode::Fat => {
                if !build.is_fresh {
                    self.write_executable(&build.exe)
                        .await
                        .context("Failed to write main executable")?;

                    log::debug!("Binary created at {}", self.build_dir().display());
                }
            }
        }

        // Populate the patch cache if we're in fat mode
        if matches!(mode, BuildMode::Fat) {
            build.patch_cache = Some(Arc::new(self.create_patch_cache(&build.exe).await?));
        }

        Ok(build)
    }

    async fn patch(
        &self,
        build: &Build,
        changed_files: BTreeSet<PathBuf>,
        changed_crates: BTreeSet<String>,
        state: &mut PatchState,
        aslr_reference: u64,
    ) -> Result<Patch> {
        self.compile_workspace_crates(changed_crates, state).await?;

        let workspace_rlibs = state
            .modified
            .iter()
            .map(|crate_name| self.rlib(crate_name))
            .collect::<Result<_>>()?;

        let cache = build.patch_cache.clone().unwrap();
        let patch_base = self.patch_base(&cache, state, aslr_reference)?;

        let patch = self
            .build(BuildMode::Thin {
                rustc_args: build.direct_rustc.clone(),
                workspace_rlibs,
                patch_base,
                changed_files: changed_files.into_iter().collect(),
                aslr_reference,
                cache: build.patch_cache.clone().unwrap(),
            })
            .await?;

        // The next patch goes after this one, which stays loaded.
        if let Some(base) = patch_base {
            let size = hotpatch::pe_image_size(&self.patch_exe(patch.time_start))?;
            state.next_patch_base = Some(align_up(base + size, PATCH_ALIGNMENT));
        }

        let jump_table_start = Instant::now();
        let jump_table = hotpatch::create_jump_table(
            &self.patch_exe(patch.time_start),
            &self.triple,
            build.patch_cache.as_ref().unwrap(),
        )?;
        log::debug!(
            "Created jump table ({} entries) in {:.2?}",
            jump_table.map.len(),
            jump_table_start.elapsed()
        );

        let detours = match patch_base {
            Some(base) if self.redirect => self.detours(
                &cache,
                state,
                &self.patch_exe(patch.time_start),
                base,
                aslr_reference,
            )?,
            _ => Vec::new(),
        };

        // Older patches stay loaded, but only the newest one runs and is worth debugging.
        let current_pdb = self.patch_exe(patch.time_start).with_extension("pdb");
        self.remove_patch_files(|path| {
            match path.extension().and_then(|extension| extension.to_str()) {
                Some("lib" | "exp" | "map") => true,
                Some("pdb") => path != current_pdb,
                _ => false,
            }
        });

        Ok(Patch {
            jump_table,
            detours,
        })
    }

    /// The jumps that redirect the old versions of the functions in the patch at `module`, in
    /// the executable and in earlier patches, to the patch.
    ///
    /// Closures and async bodies are left alone: values created by the old code, like stored
    /// callbacks and futures in flight, hold the captures of the old body, which the new body
    /// may lay out differently.
    fn detours(
        &self,
        cache: &hotpatch::Cache,
        state: &mut PatchState,
        module: &Path,
        base: u64,
        aslr_reference: u64,
    ) -> Result<Vec<Detour>> {
        let start = Instant::now();
        let functions =
            hotpatch::patch_functions(&module.with_extension("map"), &self.crate_names())?;

        let exe_base = aslr_reference
            - cache
                .symbol_table
                .get("main")
                .context("No `main` in the executable")?
                .address;

        let old_images = std::iter::once((exe_base, &cache.functions)).chain(
            state
                .patches
                .iter()
                .map(|patch| (patch.base, &patch.functions)),
        );

        let mut detours = Vec::new();

        for (old_base, old_functions) in old_images {
            for (name, function) in &functions {
                let Some(old) = old_functions.get(name) else {
                    continue;
                };

                // Too small to hold the jump without overwriting the next function.
                if old.size < JUMP_SIZE {
                    continue;
                }

                log::trace!(
                    "Redirecting {name} from {:#x} to {:#x}",
                    old_base + old.rva,
                    base + function.rva
                );

                detours.push(Detour {
                    from: old_base + old.rva,
                    to: base + function.rva,
                });
            }
        }

        // Demangling is the slow part, so only the redirected functions are checked.
        let is_body = |name: &str| {
            rustc_demangle::try_demangle(name)
                .map(|demangled| format!("{demangled:#}").contains("::{"))
                .unwrap_or(true)
        };

        let targets: HashSet<u64> = detours.iter().map(|detour| detour.to).collect();

        let bodies: HashSet<u64> = functions
            .iter()
            .filter(|(_, function)| targets.contains(&(base + function.rva)))
            .filter(|(name, _)| is_body(name))
            .map(|(_, function)| base + function.rva)
            .collect();

        detours.retain(|detour| !bodies.contains(&detour.to));

        log::debug!(
            "Found {} functions to redirect, of {} in the patch, in {:.2?}",
            detours.len(),
            functions.len(),
            start.elapsed()
        );

        state.patches.push(LoadedPatch { base, functions });

        Ok(detours)
    }

    /// Load the patch at `module` into the app and redirect the app's old code to it.
    #[cfg(windows)]
    async fn redirect(
        &self,
        executable: &process::Child,
        module: PathBuf,
        base: u64,
        detours: Vec<Detour>,
    ) -> Result<()> {
        use cargo_hot::detour;

        let pid = executable.id().context("The app is not running")?;

        let start = Instant::now();
        let count = detours.len();

        tokio::task::spawn_blocking(move || {
            let mut detours: Vec<detour::Detour> = detours
                .into_iter()
                .map(|Detour { from, to }| detour::Detour { from, to })
                .collect();

            detour::load(pid, &module, base)?;
            detour::apply(pid, &mut detours)
        })
        .await??;

        log::debug!("Redirected {count} functions in {:.2?}", start.elapsed());

        Ok(())
    }

    #[cfg(not(windows))]
    async fn redirect(
        &self,
        _executable: &process::Child,
        _module: PathBuf,
        _base: u64,
        _detours: Vec<Detour>,
    ) -> Result<()> {
        Ok(())
    }

    /// The names of the crates whose code patches replace.
    fn crate_names(&self) -> Vec<String> {
        std::iter::once(self.crate_target.crate_name())
            .chain(
                self.workspace_crates
                    .iter()
                    .map(|krate| krate.crate_name.clone()),
            )
            .collect()
    }

    /// Where to load the next patch on Windows: right after the app's executable, or after the
    /// previous patch.
    ///
    /// x64 code reaches data with 32-bit offsets, also the data of the executable that a patch
    /// uses. Those offsets are fixed when linking, so the patch must be loaded where it was
    /// linked for, and within 2 GiB of the executable. Windows would otherwise load it at a
    /// random address, usually much further away.
    fn patch_base(
        &self,
        cache: &hotpatch::Cache,
        state: &PatchState,
        aslr_reference: u64,
    ) -> Result<Option<u64>> {
        if self.linker_flavor() != LinkerFlavor::Msvc {
            return Ok(None);
        }

        let main = cache
            .symbol_table
            .get("main")
            .context("No `main` in the executable")?;

        let exe_base = aslr_reference - main.address;
        let exe_end = exe_base + hotpatch::pe_image_size(&self.main_exe())?;
        let base = state
            .next_patch_base
            .unwrap_or_else(|| align_up(exe_end, PATCH_ALIGNMENT));

        // Patches are rarely bigger than this, and the last byte must still be in reach.
        const MAX_PATCH_SIZE: u64 = 256 << 20;

        if base + MAX_PATCH_SIZE - exe_base > 2 << 30 {
            log::warn!(
                "The patches loaded so far fill the memory next to the app, \
                restart it if patching starts to fail"
            );
        }

        Ok(Some(base))
    }

    /// Recompile the changed workspace libraries and every library depending on them, in place.
    ///
    /// The target compiles against the libraries on disk, and rustc refuses to mix a recompiled
    /// library with dependents compiled against its old version. Their code is part of the
    /// patch, so the generic and inlined code they took from a changed library is updated too.
    ///
    /// Crates whose dependencies are all compiled are compiled in parallel.
    async fn compile_workspace_crates(
        &self,
        changed_crates: BTreeSet<String>,
        state: &mut PatchState,
    ) -> Result<()> {
        // `workspace_crates` lists dependencies first, so one pass finds every dependent.
        let mut stale = changed_crates;
        stale.append(&mut state.pending);

        for krate in &self.workspace_crates {
            if krate
                .dependencies
                .iter()
                .any(|dependency| stale.contains(dependency))
            {
                let _ = stale.insert(krate.crate_name.clone());
            }
        }

        if stale.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let mut compiling = tokio::task::JoinSet::new();
        let mut waiting: Vec<&WorkspaceCrate> = self
            .workspace_crates
            .iter()
            .filter(|krate| stale.contains(&krate.crate_name))
            .collect();

        let result = loop {
            let (ready, blocked): (Vec<_>, Vec<_>) = waiting.into_iter().partition(|krate| {
                !krate
                    .dependencies
                    .iter()
                    .any(|dependency| stale.contains(dependency))
            });

            waiting = blocked;

            for krate in ready {
                let crate_name = krate.crate_name.clone();
                let command = self.workspace_crate_command(&crate_name);

                let _ = compiling.spawn(async move {
                    let result = match command {
                        Ok(command) => compile_workspace_crate(&crate_name, command).await,
                        Err(error) => Err(error),
                    };

                    (crate_name, result)
                });
            }

            let Some(compiled) = compiling.join_next().await else {
                break Ok(());
            };

            let (crate_name, result) = compiled?;

            if let Err(error) = result {
                break Err(error);
            }

            let _ = stale.remove(&crate_name);
            let _ = state.modified.insert(crate_name);
        };

        // A dependent compiled after a failure would build against outdated libraries, so
        // whatever has not finished is compiled again next time.
        compiling.shutdown().await;
        state.pending = stale;

        result?;

        log::debug!("Compiled workspace crates in {:.2?}", start.elapsed());

        Ok(())
    }

    /// Replay the recorded rustc invocation of a workspace library.
    fn workspace_crate_command(&self, crate_name: &str) -> Result<process::Command> {
        let args = self
            .read_rustc_args(crate_name, true)
            .with_context(|| format!("The arguments to compile {crate_name} were not recorded"))?;

        let mut command = process::Command::new("rustc");

        let _ = command
            .current_dir(&self.workspace_dir)
            .env_clear()
            .envs(
                args.envs
                    .iter()
                    .filter(|(key, _)| {
                        ![
                            "RUSTC_WRAPPER",
                            "RUSTC_WORKSPACE_WRAPPER",
                            rustc::DX_RUSTC_WRAPPER_ENV_VAR,
                            rustc::WORKSPACE_DIR_ENV_VAR,
                        ]
                        .contains(&key.as_str())
                            && !key.starts_with("DX_LINK")
                    })
                    .cloned(),
            )
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        // Like the target, the libraries' code ends up in every patch, and full debug info
        // makes linking much slower.
        if self.full_debuginfo {
            let _ = command.args(&args.args[1..]);
        } else {
            // Other flags invalidate the incremental cache of cargo's builds, so patches keep
            // their own. Otherwise, every patch after a restart and every restart after a patch
            // would compile the crate from scratch.
            let mut replay_args = args.args[1..].to_vec();

            for arg in &mut replay_args {
                for prefix in ["incremental=", "-Cincremental="] {
                    if let Some(dir) = arg.strip_prefix(prefix) {
                        *arg = format!("{prefix}{dir}-cargo-hot");
                    }
                }
            }

            let _ = command
                .args(replay_args)
                .arg("-Cdebuginfo=line-tables-only");
        }

        Ok(command)
    }

    /// The rlib a workspace library compiles to, as named by its recorded arguments.
    fn rlib(&self, crate_name: &str) -> Result<PathBuf> {
        let args = self
            .read_rustc_args(crate_name, true)
            .with_context(|| format!("The arguments to compile {crate_name} were not recorded"))?;

        let out_dir = args
            .value("--out-dir")
            .with_context(|| format!("No `--out-dir` in the arguments of {crate_name}"))?;

        let extra_filename = args.codegen_option("extra-filename").unwrap_or_default();

        Ok(self
            .workspace_dir
            .join(out_dir)
            .join(format!("lib{crate_name}{extra_filename}.rlib")))
    }

    async fn create_patch_cache(&self, exe: &Path) -> Result<hotpatch::Cache> {
        // TODO: Wasm
        let exe = exe.to_path_buf();
        let start = Instant::now();
        let cache = hotpatch::Cache::new(&exe, &self.triple, &self.crate_names())?;

        log::debug!(
            "Read {} symbols of the app in {:.2?}",
            cache.symbol_table.len(),
            start.elapsed()
        );

        Ok(cache)
    }

    async fn write_executable(&self, exe: &Path) -> Result<()> {
        // TODO: Wasm
        let _ = std::fs::copy(exe, self.main_exe())?;

        Ok(())
    }

    /// Run the cargo build by assembling the build command and executing it.
    ///
    /// This method needs to be very careful with processing output since errors being swallowed will
    /// be very confusing to the user.
    async fn cargo_build(&self, mode: &BuildMode) -> Result<Build> {
        use tokio::io::AsyncBufReadExt;

        let time_start = SystemTime::now();

        if matches!(mode, BuildMode::Fat) {
            self.reset_unrecorded_fingerprints();
        }

        let mut cmd = self.build_command(mode)?;

        log::debug!("Executing cargo for {}", self.triple);

        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("Failed to spawn cargo build")?;

        let stdout = tokio::io::BufReader::new(child.stdout.take().unwrap());
        let stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
        let mut output_location: Option<PathBuf> = None;
        let mut stdout = stdout.lines();
        let mut stderr = stderr.lines();
        let mut emitting_error = false;
        let mut has_compiled = false;

        // TODO
        // let mut units_compiled = 0;

        loop {
            use cargo_metadata::Message;
            use cargo_metadata::diagnostic::Diagnostic;

            let line = tokio::select! {
                Ok(Some(line)) = stdout.next_line() => line,
                Ok(Some(line)) = stderr.next_line() => line,
                else => break,
            };

            let Some(Ok(message)) = Message::parse_stream(std::io::Cursor::new(line)).next() else {
                continue;
            };

            match message {
                Message::BuildScriptExecuted(_) => {
                    // TODO
                    // units_compiled += 1;
                }
                Message::CompilerMessage(msg) => println!("{}", msg.message),
                Message::TextLine(line) => {
                    // Handle the case where we're getting lines directly from rustc.
                    // These are in a different format than the normal cargo output, though I imagine
                    // this parsing code is quite fragile/sensitive to changes in cargo, cargo_metadata, rustc, etc.
                    #[derive(Deserialize)]
                    struct RustcArtifact {
                        artifact: PathBuf,
                        emit: String,
                    }

                    // These outputs look something like:
                    //
                    // { "artifact":"target/debug/deps/libdioxus_core-4f2a0b3c1e5f8b7c.rlib", "emit":"link" }
                    //
                    // There are other outputs like depinfo that we might be interested in in the future.
                    if let Ok(artifact) = serde_json::from_str::<RustcArtifact>(&line) {
                        if artifact.emit == "link" {
                            output_location = Some(artifact.artifact);
                        }

                        continue;
                    }

                    // Handle direct rustc diagnostics
                    if let Ok(diag) = serde_json::from_str::<Diagnostic>(&line) {
                        if let Some(rendered) = diag.rendered {
                            println!("{rendered}");
                        }

                        continue;
                    }

                    // For whatever reason, if there's an error while building, we still receive the TextLine
                    // instead of an "error" message. However, the following messages *also* tend to
                    // be the error message, and don't start with "error:". So we'll check if we've already
                    // emitted an error message and if so, we'll emit all following messages as errors too.
                    //
                    // todo: This can lead to some really ugly output though, so we might want to look
                    // into a more reliable way to detect errors propagating out of the compiler. If
                    // we always wrapped rustc, then we could store this data somewhere in a much more
                    // reliable format.
                    if line.trim_start().starts_with("error:") {
                        emitting_error = true;
                    }

                    // Note that previous text lines might have set emitting_error to true
                    match emitting_error {
                        true => eprintln!("{line}"),
                        false => println!("{line}"),
                    }
                }
                Message::CompilerArtifact(artifact) => {
                    // TODO
                    // units_compiled += 1;
                    has_compiled = has_compiled || !artifact.fresh;
                    output_location = artifact.executable.map(Into::into);
                }
                // todo: this can occasionally swallow errors, so we should figure out what exactly is going wrong
                //       since that is a really bad user experience.
                Message::BuildFinished(finished) => {
                    if !finished.success {
                        return Err(anyhow::anyhow!(
                            "Cargo build failed, signaled by the compiler. Rerun with `-v` for more information."
                        ));
                    }
                }
                _ => {}
            }
        }

        // The target's rustc args, as recorded by the wrapper.
        let mut direct_rustc = self
            .read_rustc_args(&self.crate_target.crate_name(), false)
            .unwrap_or_default();

        // If there's any warnings from the linker, we should print them out
        if let Ok(linker_warnings) = std::fs::read_to_string(self.link_err_file())
            && !linker_warnings.is_empty()
        {
            if output_location.is_none() {
                log::error!("Linker warnings: {linker_warnings}");
            } else {
                log::debug!("Linker warnings: {linker_warnings}");
            }
        }

        // Collect the linker args from the and update the rustc args
        direct_rustc.link_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let exe = output_location.context(
            "Cargo build failed - no output location. Rerun with `-v` for more information.",
        )?;

        // Fat builds need to be linked with the fat linker. Would also like to link here for thin builds
        if matches!(mode, BuildMode::Fat) && has_compiled {
            let link_start = SystemTime::now();
            self.run_fat_link(&exe, &direct_rustc).await?;

            log::debug!(
                "Fat linking completed in {}us",
                SystemTime::now()
                    .duration_since(link_start)
                    .unwrap()
                    .as_micros()
            );
        }

        let time_end = SystemTime::now();

        log::debug!(
            "Build completed successfully in {}us: {:?}",
            time_end.duration_since(time_start).unwrap().as_micros(),
            exe
        );

        Ok(Build {
            exe,
            is_fresh: !has_compiled,
            direct_rustc,
            time_start,
            patch_cache: None,
        })
    }

    async fn write_patch(
        &self,
        aslr_reference: u64,
        build: &mut Build,
        cache: &Arc<hotpatch::Cache>,
        rustc_args: &rustc::Args,
        workspace_rlibs: &[PathBuf],
        patch_base: Option<u64>,
    ) -> anyhow::Result<()> {
        log::debug!(
            "Original builds for patch: {}",
            self.link_args_file().display()
        );

        let raw_args = std::fs::read_to_string(self.link_args_file())
            .context("Failed to read link args from file")?;

        let args = raw_args.lines().collect::<Vec<_>>();

        // Extract out the incremental object files.
        //
        // This is sadly somewhat of a hack, but it might be a moderately reliable hack.
        //
        // When rustc links your project, it passes the args as how a linker would expect, but with
        // a somewhat reliable ordering. These are all internal details to cargo/rustc, so we can't
        // rely on them *too* much, but the *are* fundamental to how rust compiles your projects, and
        // linker interfaces probably won't change drastically for another 40 years.
        //
        // We need to tear apart this command and only pass the args that are relevant to our thin link.
        // Mainly, we don't want any rlibs to be linked. Occasionally some libraries like objc_exception
        // export a folder with their artifacts - unsure if we actually need to include them. Generally
        // you can err on the side that most *libraries* don't need to be linked here since dlopen
        // satisfies those symbols anyways when the binary is loaded.
        //
        // Many args are passed twice, too, which can be confusing, but generally don't have any real
        // effect. Note that on macos/ios, there's a special macho header that needs to be set, otherwise
        // dyld will complain.
        //
        // Also, some flags in darwin land might become deprecated, need to be super conservative:
        // - https://developer.apple.com/forums/thread/773907
        //
        // The format of this command roughly follows:
        // ```
        // clang
        //     /dioxus/target/debug/subsecond-cli
        //     /var/folders/zs/gvrfkj8x33d39cvw2p06yc700000gn/T/rustcAqQ4p2/symbols.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.05stnb4bovskp7a00wyyf7l9s.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.08rgcutgrtj2mxoogjg3ufs0g.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.0941bd8fa2bydcv9hfmgzzne9.rcgu.o
        //     /dioxus/target/subsecond-dev/deps/libbincode-c215feeb7886f81b.rlib
        //     /dioxus/target/subsecond-dev/deps/libanyhow-e69ac15c094daba6.rlib
        //     /dioxus/target/subsecond-dev/deps/libratatui-c3364579b86a1dfc.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libstd-019f0f6ae6e6562b.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libpanic_unwind-7387d38173a2eb37.rlib
        //     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libobject-2b03cf6ece171d21.rlib
        //     -framework AppKit
        //     -lc
        //     -framework Foundation
        //     -framework Carbon
        //     -lSystem
        //     -framework CoreFoundation
        //     -lobjc
        //     -liconv
        //     -lm
        //     -arch arm64
        //     -mmacosx-version-min=11.0.0
        //     -L /dioxus/target/subsecond-dev/build/objc_exception-dc226cad0480ea65/out
        //     -o /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa
        //     -nodefaultlibs
        //     -Wl,-all_load
        // ```
        let mut dylibs = vec![];
        let temp_objects = args
            .iter()
            .filter(|arg| arg.ends_with(".rcgu.o"))
            .sorted()
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        let sharing = if self.triple.operating_system == OperatingSystem::Windows {
            let start = Instant::now();
            let sharing = statics::share(
                cache,
                &temp_objects,
                workspace_rlibs,
                &self.shared_objects_dir(),
            )?;

            log::debug!(
                "Shared {} statics with the app in {:.2?}",
                sharing.count,
                start.elapsed()
            );

            sharing
        } else {
            statics::Sharing::default()
        };

        let mut object_files = temp_objects.clone();
        object_files.extend(sharing.objects.iter().cloned());

        // On non-wasm platforms, we generate a special shim object file which converts symbols from
        // fat binary into direct addresses from the running process.
        //
        // Our wasm approach is quite specific to wasm. We don't need to resolve any missing symbols
        // there since wasm is relocatable, but there is considerable pre and post processing work to
        // satisfy undefined symbols that we do by munging the binary directly.
        //
        // todo: can we adjust our wasm approach to also use a similar system?
        // todo: don't require the aslr reference and just patch the got when loading.
        //
        // Requiring the ASLR offset here is necessary but unfortunately might be flakey in practice.
        // Android apps can take a long time to open, and a hot patch might've been issued in the interim,
        // making this hotpatch a failure.
        if self.triple.architecture != Architecture::Wasm32
            && self.triple.architecture != Architecture::Wasm64
        {
            let stub_start = Instant::now();
            let stub_bytes = hotpatch::create_undefined_symbol_stub(
                cache,
                &[object_files.as_slice(), workspace_rlibs].concat(),
                &sharing.replaced_members,
                sharing.thread_locals,
                &self.triple,
                aslr_reference,
            )
            .context("Failed to resolve patch symbols")?;
            log::debug!(
                "Resolved symbols of {} object files in {:.2?}",
                object_files.len() + workspace_rlibs.len(),
                stub_start.elapsed()
            );

            // Currently we're dropping stub.o in the exe dir, but should probably just move to a tempfile?
            let patch_file = self.main_exe().with_file_name("stub.o");
            std::fs::write(&patch_file, stub_bytes)?;
            object_files.push(patch_file);

            // Add the dylibs/sos to the linker args
            // Make sure to use the one in the bundle, not the ones in the target dir or system.
            for arg in &rustc_args.link_args {
                if arg.ends_with(".dylib") || arg.ends_with(".so") {
                    let path = PathBuf::from(arg);
                    dylibs.push(self.frameworks_folder().join(path.file_name().unwrap()));
                }
            }
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;
        let out_exe = self.patch_exe(build.time_start);
        let out_arg = match self.triple.operating_system {
            OperatingSystem::Windows => vec![format!("/OUT:{}", out_exe.display())],
            _ => vec!["-o".to_string(), out_exe.display().to_string()],
        };

        log::trace!("Linking with {linker:?} using args: {object_files:#?}");

        let mut out_args: Vec<OsString> = vec![];
        out_args.extend(object_files.iter().map(Into::into));

        // The recompiled workspace libraries are linked as archives, so only the code that the
        // target reaches is pulled in. They come after the stub: the linker pulls in an archive
        // member as soon as it defines a symbol that is undefined so far, and the stub defines
        // the statics shared with the app, which members define too. Their rlibs are kept,
        // unlike the target's objects.
        out_args.extend(workspace_rlibs.iter().map(Into::into));
        out_args.extend(dylibs.iter().map(Into::into));
        out_args.extend(self.thin_link_args(&args)?.iter().map(Into::into));

        if let Some(base) = patch_base {
            // A fixed base keeps Windows from moving the patch.
            out_args.push(format!("/BASE:{base:#x}").into());
            out_args.push("/DYNAMICBASE:NO".into());
            // For the functions to redirect to the patch.
            out_args.push(format!("/MAP:{}", out_exe.with_extension("map").display()).into());
        }
        out_args.extend(out_arg.iter().map(Into::into));

        if cfg!(windows) {
            let cmd_contents: String = out_args
                .iter()
                .map(|s| format!("\"{}\"", s.to_string_lossy()))
                .join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display()).into()];
        }

        // Run the linker directly!
        //
        // We dump its output directly into the patch exe location which is different than how rustc
        // does it since it uses llvm-objcopy into the `target/debug/` folder.
        let linker_start = Instant::now();
        let res = tokio::process::Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(rustc_args.envs.iter().map(|(k, v)| (k, v)))
            .output()
            .await?;
        log::debug!("Ran the linker in {:.2?}", linker_start.elapsed());

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            if !self.patch_exe(build.time_start).exists() || !res.status.success() {
                log::error!("Failed to generate patch: {}", errs.trim());
            } else {
                log::trace!("Linker output during thin linking: {}", errs.trim());
            }
        }

        // For some really weird reason that I think is because of dlopen caching, future loads of the
        // jump library will fail if we don't remove the original fat file. I think this could be
        // because of library versioning and namespaces, but really unsure.
        //
        // The errors if you forget to do this are *extremely* cryptic - missing symbols that never existed.
        //
        // Fortunately, this binary exists in two places - the deps dir and the target out dir. We
        // can just remove the one in the deps dir and the problem goes away.
        if let Some(idx) = args.iter().position(|arg| *arg == "-o") {
            _ = std::fs::remove_file(PathBuf::from(args[idx + 1]));
        }

        // Clean up the temps manually
        // todo: we might want to keep them around for debugging purposes
        if let Some(object_dir) = temp_objects.first().and_then(|file| file.parent()) {
            self.remove_saved_bitcode(object_dir);
        }

        for file in temp_objects {
            _ = std::fs::remove_file(file);
        }

        _ = std::fs::remove_file(self.main_exe().with_file_name("stub.o"));

        Ok(())
    }

    /// Patches are stored in the same directory as the main executable, but with a name based on the
    /// time the patch started compiling.
    ///
    /// - lib{name}-patch-{time}.(so/dll/dylib) (next to the main exe)
    ///
    /// Note that weirdly enough, the name of dylibs can actually matter. In some environments, libs
    /// can override each other with symbol interposition.
    ///
    /// Also, on Android - and some Linux, we *need* to start the lib name with `lib` for the dynamic
    /// loader to consider it a shared library.
    ///
    /// todo: the time format might actually be problematic if two platforms share the same build folder.
    fn patch_exe(&self, time_start: SystemTime) -> PathBuf {
        let path = self.main_exe().with_file_name(format!(
            "lib{}-patch-{}",
            self.executable_name(),
            time_start
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|f| f.as_millis())
                .unwrap_or(0),
        ));

        let extension = match self.linker_flavor() {
            LinkerFlavor::Darwin => "dylib",
            LinkerFlavor::Gnu => "so",
            LinkerFlavor::WasmLld => "wasm",
            LinkerFlavor::Msvc => "dll",
            LinkerFlavor::Unsupported => "",
        };

        path.with_extension(extension)
    }

    /// Take the original args passed to the "fat" build and then create the "thin" variant.
    ///
    /// This is basically just stripping away the rlibs and other libraries that will be satisfied
    /// by our stub step.
    fn thin_link_args(&self, original_args: &[&str]) -> Result<Vec<String>> {
        let mut out_args = vec![];

        match self.linker_flavor() {
            // wasm32-unknown-unknown -> use wasm-ld (gnu-lld)
            //
            // We need to import a few things - namely the memory and ifunc table.
            //
            // We can safely export everything, I believe, though that led to issues with the "fat"
            // binaries that also might lead to issues here too. wasm-bindgen chokes on some symbols
            // and the resulting JS has issues.
            //
            // We turn on both --pie and --experimental-pic but I think we only need --pie.
            //
            // We don't use *any* of the original linker args since they do lots of custom exports
            // and other things that we don't need.
            //
            // The trickiest one here is -Crelocation-model=pic, which forces data symbols
            // into a GOT, making it possible to import them from the main module.
            //
            // I think we can make relocation-model=pic work for non-wasm platforms, enabling
            // fully relocatable modules with no host coordination in lieu of sending out
            // the aslr slide at runtime.
            LinkerFlavor::WasmLld => {
                out_args.extend([
                    "--fatal-warnings".to_string(),
                    "--verbose".to_string(),
                    "--import-memory".to_string(),
                    "--import-table".to_string(),
                    "--growable-table".to_string(),
                    "--export".to_string(),
                    "main".to_string(),
                    "--allow-undefined".to_string(),
                    "--no-demangle".to_string(),
                    "--no-entry".to_string(),
                    "--pie".to_string(),
                    "--experimental-pic".to_string(),
                ]);

                // retain exports so post-processing has hooks to work with
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "--export" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }
                }
            }

            // This uses "cc" and these args need to be ld compatible
            //
            // Most importantly, we want to pass `-dylib` to both CC and the linker to indicate that
            // we want to generate the shared library instead of an executable.
            LinkerFlavor::Darwin => {
                out_args.extend(["-Wl,-dylib".to_string()]);

                // Preserve the original args. We only preserve:
                // -framework
                // -arch
                // -lxyz
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-framework" || *arg == "-arch" || *arg == "-L" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-Wl,-fuse-ld")
                        || arg.starts_with("-fuse-ld")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            // android/linux need to be compatible with lld
            //
            // android currently drags along its own libraries and other zany flags
            LinkerFlavor::Gnu => {
                out_args.extend([
                    "-shared".to_string(),
                    "-Wl,--eh-frame-hdr".to_string(),
                    "-Wl,-z,noexecstack".to_string(),
                    "-Wl,-z,relro,-z,now".to_string(),
                    "-nodefaultlibs".to_string(),
                    "-Wl,-Bdynamic".to_string(),
                ]);

                // Preserve the original args. We only preserve:
                // -L <path>
                // -arch
                // -lxyz
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-L" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-Wl,--target=")
                        || arg.starts_with("-Wl,-fuse-ld")
                        || arg.starts_with("-fuse-ld")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            LinkerFlavor::Msvc => {
                out_args.extend([
                    "shlwapi.lib".to_string(),
                    "kernel32.lib".to_string(),
                    "advapi32.lib".to_string(),
                    "ntdll.lib".to_string(),
                    "userenv.lib".to_string(),
                    "ws2_32.lib".to_string(),
                    "dbghelp.lib".to_string(),
                    "/defaultlib:msvcrt".to_string(),
                    "/DLL".to_string(),
                    "/DEBUG".to_string(),
                    "/PDBALTPATH:%_PDB%".to_string(),
                    "/EXPORT:main".to_string(),
                    "/HIGHENTROPYVA:NO".to_string(),
                ]);
            }

            LinkerFlavor::Unsupported => {
                return Err(anyhow::anyhow!("Unsupported platform for thin linking"));
            }
        }

        let extract_value = |arg: &str| -> Option<String> {
            original_args
                .iter()
                .position(|a| *a == arg)
                .map(|i| original_args[i + 1].to_string())
        };

        if let Some(vale) = extract_value("-target") {
            out_args.push("-target".to_string());
            out_args.push(vale);
        }

        if let Some(vale) = extract_value("-isysroot") {
            out_args.push("-isysroot".to_string());
            out_args.push(vale);
        }

        Ok(out_args)
    }

    fn main_exe(&self) -> PathBuf {
        self.exe_dir().join(self.platform_exe_name())
    }

    fn executable_name(&self) -> &str {
        self.crate_target.name()
    }

    fn platform_exe_name(&self) -> String {
        if self.triple.operating_system == OperatingSystem::Windows {
            return format!("{}.exe", self.executable_name());
        }

        if self.triple.architecture == Architecture::Wasm32
            || self.triple.architecture == Architecture::Wasm64
        {
            // this will be wrong, I think, but not important?
            return format!("{}_bg.wasm", self.executable_name());
        }

        self.executable_name().to_string()
    }

    fn exe_dir(&self) -> PathBuf {
        self.build_dir()
    }

    fn build_dir(&self) -> PathBuf {
        self.internal_out_dir()
            .join(self.executable_name())
            .join(&self.profile)
    }

    fn internal_out_dir(&self) -> PathBuf {
        self.target_dir.as_path_unlocked().join("cargo-hot")
    }

    /// When we link together the fat binary, we need to make sure every `.o` file in *every* rlib
    /// is taken into account. This is the same work that the rust compiler does when assembling
    /// staticlibs.
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_codegen_ssa/src/back/link.rs#L448>
    ///
    /// Since we're going to be passing these to the linker, we need to make sure and not provide any
    /// weird files (like the rmeta) file that rustc generates.
    ///
    /// We discovered the need for this after running into issues with wasm-ld not being able to
    /// handle the rmeta file.
    ///
    /// <https://github.com/llvm/llvm-project/issues/55786>
    ///
    /// Also, crates might not drag in all their dependent code. The monorphizer won't lift trait-based generics:
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_monomorphize/src/collector.rs>
    ///
    /// When Rust normally handles this, it uses the +whole-archive directive which adjusts how the rlib
    /// is written to disk.
    ///
    /// Since creating this object file can be a lot of work, we cache it in the target dir by hashing
    /// the names of the rlibs in the command and storing it in the target dir. That way, when we run
    /// this command again, we can just used the cached object file.
    ///
    /// In theory, we only need to do this for every crate accessible by the current crate, but that's
    /// hard acquire without knowing the exported symbols from each crate.
    ///
    /// todo: I think we can traverse our immediate dependencies and inspect their symbols, unless they `pub use` a crate
    /// todo: we should try and make this faster with memmapping
    pub(crate) async fn run_fat_link(&self, exe: &Path, rustc_args: &rustc::Args) -> Result<()> {
        use uuid::Uuid;

        // Filter out the rlib files from the arguments
        let rlibs = rustc_args
            .link_args
            .iter()
            .filter(|arg| arg.ends_with(".rlib"))
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        // Acquire a hash from the rlib names, sizes, modified times, and dx's git commit hash
        // This ensures that any changes in dx or the rlibs will cause a new hash to be generated
        // The hash relies on both dx and rustc hashes, so it should be thoroughly unique. Keep it
        // short to avoid long file names.
        let hash_id = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            rlibs
                .iter()
                .map(|p| {
                    format!(
                        "{}-{}-{}-{}",
                        p.file_name().unwrap().to_string_lossy(),
                        p.metadata().map(|m| m.len()).unwrap_or_default(),
                        p.metadata()
                            .ok()
                            .and_then(|m| m.modified().ok())
                            .and_then(|f| f
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .map(|f| f.as_secs())
                                .ok())
                            .unwrap_or_default(),
                        env!("CARGO_PKG_VERSION"),
                    )
                })
                .collect::<String>()
                .as_bytes(),
        )
        .to_string()
        .chars()
        .take(8)
        .collect::<String>();

        // Check if we already have a cached object file
        let out_ar_path = exe.with_file_name(format!("libdeps-{hash_id}.a",));
        let out_rlibs_list = exe.with_file_name(format!("rlibs-{hash_id}.txt"));
        let mut archive_has_contents = out_ar_path.exists();

        // Use the rlibs list if it exists
        let mut compiler_rlibs = std::fs::read_to_string(&out_rlibs_list)
            .ok()
            .map(|s| s.lines().map(PathBuf::from).collect::<Vec<_>>())
            .unwrap_or_default();

        // Create it by dumping all the rlibs into it
        // This will include the std rlibs too, which can severely bloat the size of the archive
        //
        // The nature of this process involves making extremely fat archives, so we should try and
        // speed up the future linking process by caching the archive.
        //
        // Since we're using the git hash for the CLI entropy, debug builds should always regenerate
        // the archive since their hash might not change, but the logic might.
        if !archive_has_contents || cfg!(debug_assertions) {
            compiler_rlibs.clear();

            let mut bytes = vec![];
            let mut out_ar = ar::Builder::new(&mut bytes);
            for rlib in &rlibs {
                // Skip compiler rlibs since they're missing bitcode
                //
                // https://github.com/rust-lang/rust/issues/94232#issuecomment-1048342201
                //
                // if the rlib is not in the target directory, we skip it.
                if !rlib.starts_with(&self.workspace_dir) {
                    compiler_rlibs.push(rlib.clone());
                    log::trace!("Skipping rlib: {rlib:?}");
                    continue;
                }

                log::trace!("Adding rlib to staticlib: {rlib:?}");

                let rlib_contents = std::fs::read(rlib)?;
                let mut reader = ar::Archive::new(std::io::Cursor::new(rlib_contents));
                let mut keep_linker_rlib = false;
                while let Some(Ok(object_file)) = reader.next_entry() {
                    let name = std::str::from_utf8(object_file.header().identifier()).unwrap();
                    if name.ends_with(".rmeta") {
                        continue;
                    }

                    if object_file.header().size() == 0 {
                        continue;
                    }

                    // rlibs might contain dlls/sos/lib files which we don't want to include
                    //
                    // This catches .dylib, .so, .dll, .lib, .o, etc files that are not compatible with
                    // our "fat archive" linking process.
                    //
                    // We only trust `.rcgu.o` files to make it into the --all_load archive.
                    // This is a temporary stopgap to prevent issues with libraries that generate
                    // object files that are not compatible with --all_load.
                    // see https://github.com/DioxusLabs/dioxus/issues/4237
                    if !(name.ends_with(".rcgu.o") || name.ends_with(".obj")) {
                        keep_linker_rlib = true;
                        continue;
                    }

                    archive_has_contents = true;
                    out_ar
                        .append(&object_file.header().clone(), object_file)
                        .context("Failed to add object file to archive")?;
                }

                // Some rlibs contain weird artifacts that we don't want to include in the fat archive.
                // However, we still want them around in the linker in case the regular linker can handle them.
                if keep_linker_rlib {
                    compiler_rlibs.push(rlib.clone());
                }
            }

            let bytes = out_ar.into_inner().context("Failed to finalize archive")?;
            std::fs::write(&out_ar_path, bytes).context("Failed to write archive")?;
            log::debug!("Wrote fat archive to {out_ar_path:?}");

            // Run the ranlib command to index the archive. This slows down this process a bit,
            // but is necessary for some linkers to work properly.
            // We ignore its error in case it doesn't recognize the architecture
            if self.linker_flavor() == LinkerFlavor::Darwin
                && let Some(ranlib) = select_ranlib()
            {
                _ = tokio::process::Command::new(ranlib)
                    .arg(&out_ar_path)
                    .output()
                    .await;
            }
        }

        compiler_rlibs.dedup();

        // We're going to replace the first rlib in the args with our fat archive
        // And then remove the rest of the rlibs
        //
        // We also need to insert the -force_load flag to force the linker to load the archive
        let mut args = rustc_args.link_args.clone();

        if let Some(last_object) = args.iter().rposition(|arg| arg.ends_with(".o"))
            && archive_has_contents
        {
            match self.linker_flavor() {
                LinkerFlavor::WasmLld => {
                    args.insert(last_object, "--whole-archive".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.insert(last_object + 2, "--no-whole-archive".to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 3, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Gnu => {
                    args.insert(last_object, "-Wl,--whole-archive".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.insert(last_object + 2, "-Wl,--no-whole-archive".to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 3, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Darwin => {
                    args.insert(last_object, "-Wl,-force_load".to_string());
                    args.insert(last_object + 1, out_ar_path.display().to_string());
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 2, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Msvc => {
                    args.insert(
                        last_object,
                        format!("/WHOLEARCHIVE:{}", out_ar_path.display()),
                    );
                    args.retain(|arg| !arg.ends_with(".rlib"));
                    for rlib in compiler_rlibs.iter().rev() {
                        args.insert(last_object + 1, rlib.display().to_string());
                    }
                }
                LinkerFlavor::Unsupported => {
                    log::error!("Unsupported platform for fat linking");
                }
            };
        }

        // Add custom args to the linkers
        match self.linker_flavor() {
            LinkerFlavor::Gnu => {
                // Export `main` so subsecond can use it for a reference point
                args.push("-Wl,--export-dynamic-symbol,main".to_string());
            }
            LinkerFlavor::Darwin => {
                // `-all_load` is an extra step to ensure that all symbols are loaded (different than force_load)
                args.push("-Wl,-all_load".to_string());
            }
            LinkerFlavor::Msvc => {
                // Prevent alsr from overflowing 32 bits
                args.push("/HIGHENTROPYVA:NO".to_string());

                // Unlike the PDB, the map lists the statics and functions that are local to
                // their object file, which patches share with or redirect from the executable.
                args.push(format!("/MAP:{}", exe.with_extension("map").display()));

                // Redirecting a function would redirect every function folded into it.
                args.push("/OPT:NOICF".to_string());

                // Export `main` so subsecond can use it for a reference point
                args.push("/EXPORT:main".to_string());
            }
            LinkerFlavor::WasmLld | LinkerFlavor::Unsupported => {}
        }

        // We also need to remove the `-o` flag since we want the linker output to end up in the
        // rust exe location, not in the deps dir as it normally would.
        if let Some(idx) = args
            .iter()
            .position(|arg| *arg == "-o" || *arg == "--output")
        {
            let _ = args.remove(idx + 1);
            let _ = args.remove(idx);
        }

        // same but windows support
        if let Some(idx) = args.iter().position(|arg| arg.starts_with("/OUT")) {
            let _ = args.remove(idx);
        }

        // TODO
        // We want to go through wasm-ld directly, so we need to remove the -flavor flag
        // if self.platform == Platform::Web {
        //     let flavor_idx = args.iter().position(|arg| *arg == "-flavor").unwrap();
        //     args.remove(flavor_idx + 1);
        //     args.remove(flavor_idx);
        // }

        // Set the output file
        match self.triple.operating_system {
            OperatingSystem::Windows => args.push(format!("/OUT:{}", exe.display())),
            _ => args.extend(["-o".to_string(), exe.display().to_string()]),
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;

        log::trace!("Fat linking with args: {:?} {:#?}", linker, args);
        log::trace!("Fat linking with env:");
        for e in rustc_args.envs.iter() {
            log::trace!("  {}={}", e.0, e.1);
        }

        // Handle windows command files
        let mut out_args = args.clone();
        if cfg!(windows) {
            let cmd_contents: String = out_args
                .iter()
                .skip(1)
                .map(|f| format!("\"{f}\""))
                .join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display())];
        }

        // Run the linker directly!
        let res = tokio::process::Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(rustc_args.envs.iter().map(|(k, v)| (k, v)))
            .output()
            .await?;

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            if !res.status.success() {
                log::error!("Failed to generate fat binary: {}", errs.trim());
            } else {
                log::trace!("Warnings during fat linking: {}", errs.trim());
            }
        }

        if !res.stdout.is_empty() {
            let out = String::from_utf8_lossy(&res.stdout);
            log::trace!("Output from fat linking: {}", out.trim());
        }

        // Clean up the temps manually
        if let Some(object_dir) = args
            .iter()
            .find(|arg| arg.ends_with(".rcgu.o"))
            .and_then(|file| Path::new(file).parent())
        {
            self.remove_saved_bitcode(object_dir);
        }

        for f in args.iter().filter(|arg| arg.ends_with(".rcgu.o")) {
            _ = std::fs::remove_file(f);
        }

        // Cache the rlibs list
        _ = std::fs::write(
            &out_rlibs_list,
            compiler_rlibs
                .into_iter()
                .map(|s| s.display().to_string())
                .join("\n"),
        );

        Ok(())
    }

    fn linker_flavor(&self) -> LinkerFlavor {
        if let Some(custom) = self.custom_linker.as_ref() {
            let name = custom.file_name().unwrap().to_ascii_lowercase();
            match name.to_str() {
                Some("lld-link") => return LinkerFlavor::Msvc,
                Some("lld-link.exe") => return LinkerFlavor::Msvc,
                Some("wasm-ld") => return LinkerFlavor::WasmLld,
                Some("ld64.lld") => return LinkerFlavor::Darwin,
                Some("ld.lld") => return LinkerFlavor::Gnu,
                Some("ld.gold") => return LinkerFlavor::Gnu,
                Some("mold") => return LinkerFlavor::Gnu,
                Some("sold") => return LinkerFlavor::Gnu,
                Some("wild") => return LinkerFlavor::Gnu,
                _ => {}
            }
        }

        match self.triple.environment {
            target_lexicon::Environment::Gnu
            | target_lexicon::Environment::Gnuabi64
            | target_lexicon::Environment::Gnueabi
            | target_lexicon::Environment::Gnueabihf
            | target_lexicon::Environment::GnuLlvm => LinkerFlavor::Gnu,
            target_lexicon::Environment::Musl => LinkerFlavor::Gnu,
            target_lexicon::Environment::Android => LinkerFlavor::Gnu,
            target_lexicon::Environment::Msvc => LinkerFlavor::Msvc,
            target_lexicon::Environment::Macabi => LinkerFlavor::Darwin,
            _ => match self.triple.operating_system {
                OperatingSystem::Darwin(_) => LinkerFlavor::Darwin,
                OperatingSystem::IOS(_) => LinkerFlavor::Darwin,
                OperatingSystem::MacOSX(_) => LinkerFlavor::Darwin,
                OperatingSystem::Linux => LinkerFlavor::Gnu,
                OperatingSystem::Windows => LinkerFlavor::Msvc,
                _ => match self.triple.architecture {
                    target_lexicon::Architecture::Wasm32 => LinkerFlavor::WasmLld,
                    target_lexicon::Architecture::Wasm64 => LinkerFlavor::WasmLld,
                    _ => LinkerFlavor::Unsupported,
                },
            },
        }
    }

    fn frameworks_folder(&self) -> PathBuf {
        self.build_dir()
    }

    /// Select the linker to use for this platform.
    ///
    /// We prefer to use the rust-lld linker when we can since it's usually there.
    /// On macos, we use the system linker since macho files can be a bit finicky.
    ///
    /// This means we basically ignore the linker flavor that the user configured, which could
    /// cause issues with a custom linker setup. In theory, rust translates most flags to the right
    /// linker format.
    fn select_linker(&self) -> Result<PathBuf> {
        // Use a custom linker for non-crosscompile and crosscompile targets
        if matches!(
            self.triple.operating_system,
            OperatingSystem::Darwin(_) | OperatingSystem::Linux | OperatingSystem::Windows
        ) && let Ok(linker) = std::env::var("DX_HOST_LINKER")
        {
            return Ok(PathBuf::from(linker));
        }

        if let Ok(linker) = std::env::var("DX_LINKER") {
            return Ok(PathBuf::from(linker));
        }

        if let Some(linker) = self.custom_linker.clone() {
            return Ok(linker);
        }

        let cc = match self.linker_flavor() {
            LinkerFlavor::WasmLld => self.wasm_ld(),

            // On macOS, we use the system linker since it's usually there.
            // We could also use `lld` here, but it might not be installed by default.
            //
            // Note that this is *clang*, not `lld`.
            LinkerFlavor::Darwin => self.cc(),

            // On Linux, we use the system linker since it's usually there.
            LinkerFlavor::Gnu => self.cc(),

            // On windows, instead of trying to find the system linker, we just go with the lld.link
            // that rustup provides. It's faster and more stable then reyling on link.exe in path.
            LinkerFlavor::Msvc => self.lld_link(),

            // The rest of the platforms use `cc` as the linker which should be available in your path,
            // provided you have build-tools setup. On mac/linux this is the default, but on Windows
            // it requires msvc or gnu downloaded, which is a requirement to use rust anyways.
            //
            // The default linker might actually be slow though, so we could consider using lld or rust-lld
            // since those are shipping by default on linux as of 1.86. Window's linker is the really slow one.
            //
            // https://blog.rust-lang.org/2024/05/17/enabling-rust-lld-on-linux.html
            //
            // Note that "cc" is *not* a linker. It's a compiler! The arguments we pass need to be in
            // the form of `-Wl,<args>` for them to make it to the linker. This matches how rust does it
            // which is confusing.
            LinkerFlavor::Unsupported => self.cc(),
        };

        Ok(cc)
    }

    /// Return the path to the `cc` compiler
    ///
    /// This is used for the patching system to run the linker.
    /// We could also just use lld given to us by rust itself.
    pub fn cc(&self) -> PathBuf {
        PathBuf::from("cc")
    }

    /// The windows linker
    pub fn lld_link(&self) -> PathBuf {
        self.gcc_ld_dir().join("lld-link")
    }

    pub fn wasm_ld(&self) -> PathBuf {
        self.gcc_ld_dir().join("wasm-ld")
    }

    fn gcc_ld_dir(&self) -> PathBuf {
        self.sysroot
            .join("lib")
            .join("rustlib")
            .join(Triple::host().to_string())
            .join("bin")
            .join("gcc-ld")
    }

    fn build_command(&self, mode: &BuildMode) -> Result<tokio::process::Command> {
        match mode {
            // We're assembling rustc directly, so we need to be *very* careful. Cargo sets rustc's
            // env up very particularly, and we want to match it 1:1 but with some changes.
            //
            // To do this, we reset the env completely, and then pass every env var that the original
            // rustc process had 1:1.
            //
            // We need to unset a few things, like the RUSTC wrappers and then our special env var
            // indicating that dx itself is the compiler. If we forget to do this, then the compiler
            // ends up doing some recursive nonsense and dx is trying to link instead of compiling.
            //
            // todo: maybe rustc needs to be found on the FS instead of using the one in the path?
            BuildMode::Thin { rustc_args, .. } => {
                let mut cmd = tokio::process::Command::new("rustc");

                let _ = cmd
                    .current_dir(&self.workspace_dir)
                    .env_clear()
                    .args(rustc_args.args[1..].iter())
                    .env_remove("RUSTC_WORKSPACE_WRAPPER")
                    .env_remove("RUSTC_WRAPPER")
                    .env_remove(rustc::DX_RUSTC_WRAPPER_ENV_VAR)
                    .envs(self.cargo_build_env_vars(mode)?)
                    .arg(format!("-Clinker={}", path_to_me()?.display()));

                // TODO
                // if self.platform == Platform::Web {
                //     cmd.arg("-Crelocation-model=pic");
                // }

                log::debug!("Direct rustc: {cmd:#?}");

                let _ = cmd.envs(rustc_args.envs.iter().cloned());

                // tracing::trace!("Setting env vars: {:#?}", rustc_args.envs);

                Ok(cmd)
            }

            // For Base and Fat builds, we use a regular cargo setup, but we might need to intercept
            // rustc itself in case we're hot-patching and need a reliable rustc environment to
            // continuously recompile the top-level crate with.
            //
            // In the future, when we support hot-patching *all* workspace crates, we will need to
            // make use of the RUSTC_WORKSPACE_WRAPPER environment variable instead of RUSTC_WRAPPER
            // and then keep track of env and args on a per-crate basis.
            //
            // We've also had a number of issues with incorrect canonicalization when passing paths
            // through envs on windows, hence the frequent use of dunce::canonicalize.
            _ => {
                let mut cmd = tokio::process::Command::new("cargo");

                let _ = cmd
                    .arg("rustc")
                    .current_dir(&self.crate_dir)
                    .arg("--message-format")
                    .arg("json-diagnostic-rendered-ansi")
                    .arg("--color")
                    .arg("always")
                    .args(self.cargo_build_arguments(mode))
                    .envs(self.cargo_build_env_vars(mode)?);

                if mode == &BuildMode::Fat {
                    std::fs::create_dir_all(self.rustc_args_dir())?;

                    let _ = cmd
                        .env(
                            rustc::DX_RUSTC_WRAPPER_ENV_VAR,
                            dunce::canonicalize(self.rustc_args_dir())?
                                .display()
                                .to_string(),
                        )
                        .env(rustc::WORKSPACE_DIR_ENV_VAR, &self.workspace_dir)
                        .env("RUSTC_WRAPPER", path_to_me()?.display().to_string());
                }

                log::debug!("Cargo: {cmd:#?}");

                Ok(cmd)
            }
        }
    }

    /// Create a list of arguments for cargo builds
    ///
    /// We always use `cargo rustc` *or* `rustc` directly. This means we can pass extra flags like
    /// `-C` arguments directly to the compiler.
    #[allow(clippy::vec_init_then_push)]
    fn cargo_build_arguments(&self, mode: &BuildMode) -> Vec<String> {
        let mut cargo_args = Vec::with_capacity(4);

        // Add required profile flags. --release overrides any custom profiles.
        cargo_args.push("--profile".to_string());
        cargo_args.push(self.profile.to_string());

        // Pass the appropriate target to cargo. We *always* specify a target which is somewhat helpful for preventing thrashing
        cargo_args.push("--target".to_string());
        cargo_args.push(self.triple.to_string());

        // We always run in verbose since the CLI itself is the one doing the presentation
        if self.verbose > 0 {
            cargo_args.push(format!("-{}", "v".repeat(usize::from(self.verbose))));
        }

        if self.no_default_features {
            cargo_args.push("--no-default-features".to_string());
        }

        if !self.features.is_empty() {
            cargo_args.push("--features".to_string());
            cargo_args.push(self.features.join(" "));
        }

        // We *always* set the package since that's discovered from cargo metadata
        cargo_args.push(String::from("-p"));
        cargo_args.push(self.package.clone());

        // Set the executable
        match self.crate_target.kind() {
            TargetKind::Bin => cargo_args.push("--bin".to_string()),
            TargetKind::Lib(_) => cargo_args.push("--lib".to_string()),
            TargetKind::ExampleBin => cargo_args.push("--example".to_string()),
            _ => {}
        };
        cargo_args.push(self.executable_name().to_string());

        // Merge in extra args. Order shouldn't really matter.
        cargo_args.extend(self.extra_cargo_args.clone());
        cargo_args.push("--".to_string());
        cargo_args.extend(self.extra_rustc_args.clone());

        // The Windows subsystem is left to the app (`#![windows_subsystem]`), like `cargo run` does.
        // Forcing `/SUBSYSTEM:WINDOWS` would hide the output of console apps.

        // TODO
        // The bundle splitter needs relocation data to create a call-graph.
        // This will automatically be erased by wasm-opt during the optimization step.
        // if self.platform == Platform::Web && self.wasm_split {
        //     cargo_args.push("-Clink-args=--emit-relocs".to_string());
        // }

        // dx *always* links android and thin builds
        if self.custom_linker.is_some() || matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            cargo_args.push(format!(
                "-Clinker={}",
                path_to_me().expect("can't find dx").display()
            ));
        }

        // TODO
        // for debuggability, we need to make sure android studio can properly understand our build
        // https://stackoverflow.com/questions/68481401/debugging-a-prebuilt-shared-library-in-android-studio
        // if self.platform == Platform::Android {
        //     cargo_args.push("-Clink-arg=-Wl,--build-id=sha1".to_string());
        // }

        // Handle frameworks/dylibs by setting the rpath
        // This is dependent on the bundle structure - in this case, appimage and appbundle for mac/linux
        // todo: we need to figure out what to do for windows
        match self.triple.operating_system {
            OperatingSystem::Darwin(_) | OperatingSystem::IOS(_) => {
                cargo_args.push("-Clink-arg=-Wl,-rpath,@executable_path/../Frameworks".to_string());
                cargo_args.push("-Clink-arg=-Wl,-rpath,@executable_path".to_string());
            }
            OperatingSystem::Linux => {
                cargo_args.push("-Clink-arg=-Wl,-rpath,$ORIGIN/../lib".to_string());
                cargo_args.push("-Clink-arg=-Wl,-rpath,$ORIGIN".to_string());
            }
            _ => {}
        }

        // Our fancy hot-patching engine needs a lot of customization to work properly.
        //
        // These args are mostly intended to be passed when *fat* linking but are generally fine to
        // pass for both fat and thin linking.
        //
        // We need save-temps and no-dead-strip in both cases though. When we run `cargo rustc` with
        // these args, they will be captured and re-ran for the fast compiles in the future, so whatever
        // we set here will be set for all future hot patches too.
        if matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            // rustc gives us some portable flags required:
            // - link-dead-code: prevents rust from passing -dead_strip to the linker since that's the default.
            // - save-temps=true: keeps the incremental object files around, which we need for manually linking.
            cargo_args.extend_from_slice(&[
                "-Csave-temps=true".to_string(),
                "-Clink-dead-code".to_string(),
            ]);

            // TODO
            // We need to set some extra args that ensure all symbols make it into the final output
            // and that the linker doesn't strip them out.
            //
            // This basically amounts of -all_load or --whole-archive, depending on the linker.
            // We just assume an ld-like interface on macos and a gnu-ld interface elsewhere.
            //
            // macOS/iOS use ld64 but through the `cc` interface.
            // cargo_args.push("-Clink-args=-Wl,-all_load".to_string());
            //
            // Linux and Android fit under this umbrella, both with the same clang-like entrypoint
            // and the gnu-ld interface.
            //
            // cargo_args.push("-Clink-args=-Wl,--whole-archive".to_string());
            //
            // If windows -Wl,--whole-archive is required since it follows gnu-ld convention.
            // There might be other flags on windows - we haven't tested windows thoroughly.
            //
            // cargo_args.push("-Clink-args=-Wl,--whole-archive".to_string());
            // https://learn.microsoft.com/en-us/cpp/build/reference/wholearchive-include-all-library-object-files?view=msvc-170
            //
            // ------------------------------------------------------------
            //
            // if web, -Wl,--whole-archive is required since it follows gnu-ld convention.
            //
            // We also use --no-gc-sections and --export-table and --export-memory  to push
            // said symbols into the export table.
            //
            // We use --emit-relocs to build up a solid call graph.
            //
            // rust uses its own wasm-ld linker which can be found here (it's just gcc-ld with a `-target wasm` flag):
            // - ~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld
            // - ~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld/wasm-ld
            //
            // Note that we can't use --export-all, unfortunately, since some symbols are internal
            // to wasm-bindgen and exporting them causes the JS generation to fail.
            //
            // We are basically replicating what emscripten does here with its dynamic linking
            // approach where the MAIN_MODULE is very "fat" and exports the necessary arguments
            // for the side modules to be linked in. This guide is really helpful:
            //
            // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md
            //
            // The tricky one is -Ctarget-cpu=mvp, which prevents rustc from generating externref
            // entries.
            //
            // https://blog.rust-lang.org/2024/09/24/webassembly-targets-change-in-default-target-features/#disabling-on-by-default-webassembly-proposals
            //
            // It's fine that these exist in the base module but not in the patch.
            // if self.platform == Platform::Web
            //     || self.triple.operating_system == OperatingSystem::Wasi
            // {
            //     cargo_args.push("-Ctarget-cpu=mvp".into());
            //     cargo_args.push("-Clink-arg=--no-gc-sections".into());
            //     cargo_args.push("-Clink-arg=--growable-table".into());
            //     cargo_args.push("-Clink-arg=--export-table".into());
            //     cargo_args.push("-Clink-arg=--export-memory".into());
            //     cargo_args.push("-Clink-arg=--emit-relocs".into());
            //     cargo_args.push("-Clink-arg=--export=__stack_pointer".into());
            //     cargo_args.push("-Clink-arg=--export=__heap_base".into());
            //     cargo_args.push("-Clink-arg=--export=__data_end".into());
            // }
        }

        cargo_args
    }

    fn cargo_build_env_vars(&self, mode: &BuildMode) -> Result<Vec<(&'static str, String)>> {
        let mut env_vars = vec![];

        // TODO
        // Make sure to set all the crazy android flags. Cross-compiling is hard, man.
        // if self.platform == Platform::Android {
        //     env_vars.extend(self.android_env_vars()?);
        // };

        // If we're either zero-linking or using a custom linker, make `dx` itself do the linking.
        if self.custom_linker.is_some() || matches!(mode, BuildMode::Thin { .. } | BuildMode::Fat) {
            link::LinkAction {
                triple: self.triple.clone(),
                linker: self.custom_linker.clone(),
                link_err_file: dunce::canonicalize(self.link_err_file())?,
                link_args_file: dunce::canonicalize(self.link_args_file())?,
            }
            .write_env_vars(&mut env_vars)?;
        }

        // TODO
        // Disable reference types on wasm when using hotpatching
        // https://blog.rust-lang.org/2024/09/24/webassembly-targets-change-in-default-target-features/#disabling-on-by-default-webassembly-proposals
        // if self.platform == Platform::Web
        //     && matches!(ctx.mode, BuildMode::Thin { .. } | BuildMode::Fat)
        // {
        //     env_vars.push(("RUSTFLAGS", {
        //         let mut rust_flags = std::env::var("RUSTFLAGS").unwrap_or_default();
        //         rust_flags.push_str(" -Ctarget-cpu=mvp");
        //         rust_flags
        //     }));
        // }

        Ok(env_vars)
    }

    fn link_args_file(&self) -> PathBuf {
        self.exe_dir().join("link_args.txt")
    }

    fn link_err_file(&self) -> PathBuf {
        self.exe_dir().join("link_err.txt")
    }

    /// Where the rustc wrapper records the arguments of each workspace crate.
    ///
    /// Each set of features gets its own directory: cargo keeps the artifacts of every set it
    /// built, and does not recompile, so does not record again, a crate that it has artifacts
    /// for.
    fn rustc_args_dir(&self) -> PathBuf {
        use std::hash::{DefaultHasher, Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (&self.features, self.no_default_features, &self.package).hash(&mut hasher);

        self.exe_dir()
            .join(format!("rustc-args-{:016x}", hasher.finish()))
    }

    fn rustc_args_file(&self, crate_name: &str, is_lib: bool) -> PathBuf {
        self.rustc_args_dir()
            .join(rustc::args_file_name(crate_name, is_lib))
    }

    fn read_rustc_args(&self, crate_name: &str, is_lib: bool) -> Option<rustc::Args> {
        let file = std::fs::read_to_string(self.rustc_args_file(crate_name, is_lib)).ok()?;

        serde_json::from_str(&file).ok()
    }

    /// Make cargo recompile the workspace crates whose arguments were never recorded, and the
    /// target itself, which has to be linked again to record its link arguments.
    ///
    /// Cargo skips crates that are up to date, so a crate built before `cargo hot` ran would
    /// otherwise never pass through the wrapper. Removing a crate's fingerprint is what
    /// `cargo clean -p` does too, and only that crate and its dependents are recompiled.
    fn reset_unrecorded_fingerprints(&self) {
        let mut packages: HashSet<&str> = self
            .workspace_crates
            .iter()
            .filter(|krate| !krate.is_proc_macro)
            .filter(|krate| !self.rustc_args_file(&krate.crate_name, true).exists())
            .map(|krate| krate.package.as_str())
            .collect();

        if !self
            .rustc_args_file(&self.crate_target.crate_name(), false)
            .exists()
        {
            let _ = packages.insert(&self.package);
        }

        if packages.is_empty() {
            return;
        }

        log::debug!("Recompiling to record their arguments: {packages:?}");

        let profile_dir = match self.profile.as_str() {
            "dev" | "test" => "debug",
            "bench" => "release",
            profile => profile,
        };

        let fingerprints = self
            .target_dir
            .as_path_unlocked()
            .join(self.triple.to_string())
            .join(profile_dir)
            .join(".fingerprint");

        for entry in std::fs::read_dir(fingerprints)
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name();

            let is_reset = name
                .to_str()
                .and_then(|name| name.rsplit_once('-'))
                .is_some_and(|(package, _hash)| packages.contains(package));

            if is_reset {
                _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    /// Where the copies of rlib members that share statics with the app go.
    fn shared_objects_dir(&self) -> PathBuf {
        self.exe_dir().join("shared-objects")
    }

    fn windows_command_file(&self) -> PathBuf {
        self.exe_dir().join("windows_command.txt")
    }
}

fn src_dir(target: &Target) -> Option<PathBuf> {
    Some(target.src_path().path()?.parent()?.to_path_buf())
}

/// The libraries of every workspace member the main package depends on, directly or through
/// other members, ordered so that each crate comes after its dependencies. Dev and build
/// dependencies are skipped, since they are not part of the app.
///
/// The main package's own library is included, but not its binary.
fn workspace_crates(workspace: &Workspace<'_>, main_package: &Package) -> Vec<WorkspaceCrate> {
    let members: HashMap<_, _> = workspace
        .members()
        .map(|package| (package.name(), package))
        .collect();

    let library = |package: &Package| {
        package
            .targets()
            .iter()
            .find(|target| target.is_lib())
            .cloned()
    };

    let dependencies = |package: &Package| -> Vec<&Package> {
        package
            .dependencies()
            .iter()
            .filter(|dependency| dependency.kind() == DepKind::Normal)
            .filter_map(|dependency| members.get(&dependency.package_name()).copied())
            .collect()
    };

    // Depth first, adding a crate once all of its dependencies are in.
    fn visit<'a>(
        package: &'a Package,
        dependencies: &dyn Fn(&'a Package) -> Vec<&'a Package>,
        seen: &mut HashSet<&'a str>,
        order: &mut Vec<&'a Package>,
    ) {
        if !seen.insert(package.name().as_str()) {
            return;
        }

        for dependency in dependencies(package) {
            visit(dependency, dependencies, seen, order);
        }

        order.push(package);
    }

    let mut order = Vec::new();
    visit(main_package, &dependencies, &mut HashSet::new(), &mut order);

    order
        .into_iter()
        .filter_map(|package| {
            let target = library(package)?;

            Some(WorkspaceCrate {
                crate_name: target.crate_name(),
                src_dir: src_dir(&target)?,
                is_proc_macro: target.proc_macro(),
                dependencies: dependencies(package)
                    .into_iter()
                    .filter_map(|dependency| Some(library(dependency)?.crate_name()))
                    .collect(),
                package: package.name().to_string(),
            })
        })
        .collect()
}

/// Run a workspace library's compiler, showing its diagnostics.
async fn compile_workspace_crate(crate_name: &str, mut command: process::Command) -> Result<()> {
    use cargo_metadata::diagnostic::{Diagnostic, DiagnosticLevel};
    use tokio::io::AsyncBufReadExt;

    let start = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("Failed to run rustc for {crate_name}"))?;

    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();

    loop {
        let line = tokio::select! {
            Ok(Some(line)) = stdout.next_line() => line,
            Ok(Some(line)) = stderr.next_line() => line,
            else => break,
        };

        // Cargo asks for JSON diagnostics, which come with a rendered version. Other JSON
        // lines announce the artifacts.
        match serde_json::from_str::<Diagnostic>(&line) {
            Ok(diagnostic) => {
                // The summary ("aborting due to ...") is left to the error below.
                if diagnostic.spans.is_empty() && diagnostic.level != DiagnosticLevel::Warning {
                    continue;
                }

                if let Some(rendered) = diagnostic.rendered {
                    eprint!("{rendered}");
                }
            }
            Err(_) if line.starts_with('{') => {}
            Err(_) => eprintln!("{line}"),
        }
    }

    let status = child.wait().await?;

    if !status.success() {
        return Err(anyhow!("Failed to compile {crate_name}"));
    }

    log::debug!("Compiled {crate_name} in {:.2?}", start.elapsed());

    Ok(())
}

fn select_ranlib() -> Option<PathBuf> {
    // prefer the modern llvm-ranlib if they have it
    which::which("llvm-ranlib")
        .or_else(|_| which::which("ranlib"))
        .ok()
}

fn path_to_me() -> Result<PathBuf> {
    dunce::canonicalize(std::env::current_exe().context("Failed to find cargo-hot")?)
        .context("Failed to find cargo-hot")
}

/// Accept app connections in the background, so a restarted or reconnecting app
/// can always reach us, and waiting for one never blocks the main loop.
fn accept_connections(mut server: server::Server) -> mpsc::Receiver<server::Connection> {
    let (sender, receiver) = mpsc::channel(1);

    drop(tokio::spawn(async move {
        loop {
            match server.accept().await {
                Ok(connection) => {
                    log::debug!("App connected");

                    if sender.send(connection).await.is_err() {
                        return;
                    }
                }
                Err(error) => log::warn!("Failed to accept app connection: {error:#}"),
            }
        }
    }));

    receiver
}

async fn read_batch<T>(
    receiver: &mut mpsc::Receiver<T>,
    buffer: &mut Vec<T>,
    duration: Duration,
) -> usize {
    use tokio::time;

    let Some(item) = receiver.recv().await else {
        return 0;
    };

    buffer.push(item);

    let mut n = 1;

    loop {
        let Ok(Some(item)) = time::timeout(duration, receiver.recv()).await else {
            break;
        };

        buffer.push(item);
        n += 1;
    }

    n
}

/// Detects if `dx` is being ran in a WSL environment.
///
/// We determine this based on whether the keyword `microsoft` or `wsl` is contained within the `WSL_1` or `WSL_2` files.
/// This may fail in the future as it isn't guaranteed by Microsoft.
/// See <https://github.com/microsoft/WSL/issues/423#issuecomment-221627364>
fn is_wsl() -> bool {
    const WSL_1: &str = "/proc/sys/kernel/osrelease";
    const WSL_2: &str = "/proc/version";
    const WSL_KEYWORDS: [&str; 2] = ["microsoft", "wsl"];

    // Test 1st File
    if let Ok(content) = std::fs::read_to_string(WSL_1) {
        let lowercase = content.to_lowercase();
        for keyword in WSL_KEYWORDS {
            if lowercase.contains(keyword) {
                return true;
            }
        }
    }

    // Test 2nd File
    if let Ok(content) = std::fs::read_to_string(WSL_2) {
        let lowercase = content.to_lowercase();
        for keyword in WSL_KEYWORDS {
            if lowercase.contains(keyword) {
                return true;
            }
        }
    }

    false
}
