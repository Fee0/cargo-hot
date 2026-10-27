//! Sharing the statics of the running app with its patches, on Windows.
//!
//! A patch is linked from the full object files of the code it replaces, which define the
//! statics of that code once more. Without care, the patched code uses fresh copies: counters
//! start over, caches are empty and whatever the app set up is gone.
//!
//! So before linking, every static that the executable has too is turned into a reference, which
//! the symbol stub then resolves to the executable's copy. Only new statics get storage in the
//! patch.
//!
//! Thread locals are shared the same way, if the patch defines no new ones: the patch then reads
//! them through the executable's TLS slot, at the offsets of the executable's TLS block. New
//! thread locals have no place in that block, so a patch that defines one keeps a TLS block of
//! its own, where all of its thread locals start over.
use crate::hotpatch::Cache;

use anyhow::{Context, bail};
use object::read::archive::ArchiveFile;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// How a patch shares the app's statics.
#[derive(Debug, Default)]
pub struct Sharing {
    /// Copies of rlib members defining shared statics, to link in place of the originals.
    pub objects: Vec<PathBuf>,
    /// The members of each rlib that `objects` replace.
    pub replaced_members: HashMap<PathBuf, HashSet<Vec<u8>>>,
    /// Whether the patch uses the app's thread locals.
    pub thread_locals: bool,
    /// The number of shared statics and thread locals, for logging.
    pub count: usize,
}

/// Turn the statics of `objects` and of the members of `rlibs` that the app has into references.
///
/// `objects` are rewritten in place, rlib members are copied to `out_dir` when they define a
/// shared static.
pub fn share(
    cache: &Cache,
    objects: &[PathBuf],
    rlibs: &[PathBuf],
    out_dir: &Path,
) -> anyhow::Result<Sharing> {
    let _ = std::fs::remove_dir_all(out_dir);
    std::fs::create_dir_all(out_dir)?;

    let loose = objects
        .iter()
        .map(|path| Ok((std::fs::read(path)?, path.clone(), None)));

    let mut members = Vec::new();

    for rlib in rlibs {
        let bytes = std::fs::read(rlib)?;
        let archive = ArchiveFile::parse(bytes.as_slice())?;

        for (index, member) in archive.members().enumerate() {
            let member = member?;

            if !member.name().ends_with(b".o") {
                continue;
            }

            let name = member.name().to_vec();
            let file_name = format!(
                "{}-{index}-{}",
                rlib.file_stem().unwrap_or_default().to_string_lossy(),
                String::from_utf8_lossy(&name),
            );

            members.push(Ok((
                member.data(bytes.as_slice())?.to_vec(),
                out_dir.join(file_name),
                Some((rlib.clone(), name)),
            )));
        }
    }

    let mut files = loose
        .chain(members)
        .map(|file: anyhow::Result<_>| {
            let (bytes, path, member) = file?;
            let coff = Coff::parse(&bytes).with_context(|| format!("Failed to read {path:?}"))?;

            Ok((bytes, path, member, coff))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Thread locals can only be shared if the executable has all of them.
    let new_thread_local = files.iter().find_map(|(bytes, _, _, coff)| {
        coff.statics(bytes)
            .find(|symbol| {
                symbol.is_thread_local && !cache.thread_locals.contains_key(&symbol.name)
            })
            .map(|symbol| symbol.name)
    });

    if let Some(name) = &new_thread_local {
        log::info!(
            "The patch defines a new thread local ({name}), \
            so the thread locals of patched code start over"
        );
    }

    let mut sharing = Sharing {
        thread_locals: new_thread_local.is_none() && !cache.thread_locals.is_empty(),
        ..Sharing::default()
    };

    for (bytes, path, member, coff) in &mut files {
        let shared: Vec<_> = coff
            .statics(bytes)
            .filter(|symbol| {
                if symbol.is_thread_local {
                    sharing.thread_locals && cache.thread_locals.contains_key(&symbol.name)
                } else {
                    cache.statics.contains_key(&symbol.name)
                }
            })
            .map(|symbol| (symbol.offset, symbol.section))
            .collect();

        if shared.is_empty() {
            continue;
        }

        sharing.count += shared.len();

        for (offset, section) in shared {
            coff.make_undefined(bytes, offset, section);
        }

        std::fs::write(&*path, &*bytes)?;

        if let Some((rlib, name)) = member.take() {
            sharing.objects.push(path.clone());
            let _ = sharing
                .replaced_members
                .entry(rlib)
                .or_default()
                .insert(name);
        }
    }

    Ok(sharing)
}

/// The layout of a COFF object file, in its regular or its "big object" format.
struct Coff {
    symbol_table: usize,
    symbol_count: usize,
    symbol_size: usize,
    /// The header offset, characteristics and name of each section, by its 1-based number.
    sections: Vec<(usize, u32, [u8; 8])>,
}

/// A static defined by an object file.
struct Static {
    name: String,
    /// The offset of its symbol table entry.
    offset: usize,
    /// The number of the section holding it.
    section: usize,
    is_thread_local: bool,
}

const IMAGE_SCN_LNK_COMDAT: u32 = 0x1000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;
const IMAGE_SYM_DTYPE_FUNCTION: u16 = 0x20;

/// `ANON_OBJECT_HEADER_BIGOBJ::ClassID`, which marks the big object format.
const BIGOBJ_CLASS_ID: [u8; 16] = [
    0xC7, 0xA1, 0xBA, 0xD1, 0xEE, 0xBA, 0xA9, 0x4B, 0xAF, 0x20, 0xFA, 0xF6, 0x6A, 0xA4, 0xDC, 0xB8,
];

impl Coff {
    fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        let u16_at = |offset: usize| read::<2>(bytes, offset).map(u16::from_le_bytes);
        let u32_at = |offset: usize| read::<4>(bytes, offset).map(u32::from_le_bytes);

        let is_bigobj = u16_at(0)? == 0
            && u16_at(2)? == 0xFFFF
            && bytes.get(12..28) == Some(BIGOBJ_CLASS_ID.as_slice());

        let (section_count, symbol_table, symbol_count, sections_start, symbol_size) = if is_bigobj
        {
            (u32_at(44)? as usize, u32_at(48)?, u32_at(52)?, 56, 20)
        } else {
            let optional_header = u16_at(16)? as usize;
            (
                u16_at(2)? as usize,
                u32_at(8)?,
                u32_at(12)?,
                20 + optional_header,
                18,
            )
        };

        let sections = (0..section_count)
            .map(|index| {
                let header = sections_start + index * 40;
                Ok((header, u32_at(header + 36)?, read::<8>(bytes, header)?))
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(Self {
            symbol_table: symbol_table as usize,
            symbol_count: symbol_count as usize,
            symbol_size,
            sections,
        })
    }

    /// The statics defined in writable sections, which are not shared between modules
    /// otherwise. Constants are left alone, since copies of them do no harm.
    fn statics<'a>(&'a self, bytes: &'a [u8]) -> impl Iterator<Item = Static> + 'a {
        let strings = self.symbol_table + self.symbol_count * self.symbol_size;
        let mut index = 0;

        std::iter::from_fn(move || {
            while index < self.symbol_count {
                let offset = self.symbol_table + index * self.symbol_size;
                let entry = bytes.get(offset..offset + self.symbol_size)?;
                let section_size = self.symbol_size - 16;

                let aux_count = entry[self.symbol_size - 1] as usize;
                let class = entry[self.symbol_size - 2];
                let kind = u16::from_le_bytes([entry[12 + section_size], entry[13 + section_size]]);
                let section = match section_size {
                    4 => i32::from_le_bytes(entry[12..16].try_into().unwrap()),
                    _ => i32::from(i16::from_le_bytes([entry[12], entry[13]])),
                };

                index += 1 + aux_count;

                // Section symbols carry an auxiliary record, plain symbols do not.
                if section <= 0
                    || aux_count != 0
                    || kind & 0xF0 == IMAGE_SYM_DTYPE_FUNCTION
                    || !matches!(class, IMAGE_SYM_CLASS_EXTERNAL | IMAGE_SYM_CLASS_STATIC)
                {
                    continue;
                }

                let Some(&(_, characteristics, section_name)) =
                    self.sections.get(section as usize - 1)
                else {
                    continue;
                };

                let is_thread_local = section_name.starts_with(b".tls");

                if characteristics & IMAGE_SCN_MEM_WRITE == 0 && !is_thread_local {
                    continue;
                }

                let name = if entry[..4] == [0; 4] {
                    let start =
                        strings + u32::from_le_bytes(entry[4..8].try_into().unwrap()) as usize;
                    let end = bytes[start..].iter().position(|byte| *byte == 0)? + start;
                    &bytes[start..end]
                } else {
                    let end = entry[..8].iter().position(|byte| *byte == 0).unwrap_or(8);
                    &entry[..end]
                };

                return Some(Static {
                    name: String::from_utf8_lossy(name).into_owned(),
                    offset,
                    section: section as usize,
                    is_thread_local,
                });
            }

            None
        })
    }

    /// Turn the symbol at `offset` into a reference to a symbol of the same name, defined
    /// elsewhere. Relocations point at symbols by index, so the code using it follows.
    ///
    /// Each static has a COMDAT section of its own, which the linker would keep or drop with
    /// the symbol leading it. Without the symbol, it becomes a plain section that nothing uses.
    fn make_undefined(&self, bytes: &mut [u8], offset: usize, section: usize) {
        let section_size = self.symbol_size - 16;
        let entry = &mut bytes[offset..offset + self.symbol_size];

        // Value
        entry[8..12].fill(0);
        // Section number: undefined
        entry[12..12 + section_size].fill(0);
        entry[self.symbol_size - 2] = IMAGE_SYM_CLASS_EXTERNAL;

        let (header, characteristics, _) = self.sections[section - 1];
        let characteristics = characteristics & !IMAGE_SCN_LNK_COMDAT;
        bytes[header + 36..header + 40].copy_from_slice(&characteristics.to_le_bytes());
    }
}

fn read<const N: usize>(bytes: &[u8], offset: usize) -> anyhow::Result<[u8; N]> {
    match bytes.get(offset..offset + N) {
        Some(slice) => Ok(slice.try_into().unwrap()),
        None => bail!("Truncated object file"),
    }
}
