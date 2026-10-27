//! Changes that a patch cannot apply safely.
//!
//! A patch replaces code, not data: the running app keeps the values it created, laid out as the
//! old code defined them. New code reading them with a changed layout reads garbage, so types,
//! traits (whose methods are called through vtables of the old layout) and the macros that may
//! define them have to stay the same. A change of any of them needs a restart.
//!
//! The check compares definitions token by token, so it also asks for a restart when only a
//! derive or a field's visibility changed. It cannot see layouts that change indirectly, like a
//! `const` used as an array length.
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::Visit;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The source files as the running app was built from them.
#[derive(Debug, Default)]
pub struct Sources {
    files: HashMap<PathBuf, String>,
}

impl Sources {
    /// Remember every Rust source file in `dirs`.
    pub fn read<'a>(dirs: impl IntoIterator<Item = &'a Path>) -> Self {
        let mut files = HashMap::new();
        let mut pending: Vec<PathBuf> = dirs.into_iter().map(Path::to_path_buf).collect();

        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();

                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|extension| extension == "rs")
                    && let Ok(source) = std::fs::read_to_string(&path)
                {
                    let _ = files.insert(path, source);
                }
            }
        }

        Self { files }
    }

    /// The first definition changed in `changed_files` since they were remembered, if any.
    pub fn changed_definition<'a>(
        &self,
        changed_files: impl IntoIterator<Item = &'a PathBuf>,
    ) -> Option<String> {
        for file in changed_files {
            let Some(old) = self.files.get(file) else {
                // A new file defines new types, which no data has yet.
                continue;
            };

            let Ok(new) = std::fs::read_to_string(file) else {
                continue;
            };

            if *old == new {
                continue;
            }

            // A file that does not parse does not compile either, which the patch reports.
            let (Some(old), Some(new)) = (definitions(old), definitions(&new)) else {
                continue;
            };

            if let Some(name) = old
                .iter()
                .find(|(name, tokens)| new.get(*name).is_some_and(|new| new != *tokens))
                .map(|(name, _)| name)
            {
                return Some(format!("{name} in {}", file.display()));
            }
        }

        None
    }

    /// The current contents of `files`, to [`Sources::remember`] once the app runs them.
    pub fn read_files<'a>(
        files: impl IntoIterator<Item = &'a PathBuf>,
    ) -> Vec<(PathBuf, Option<String>)> {
        files
            .into_iter()
            .map(|file| (file.clone(), std::fs::read_to_string(file).ok()))
            .collect()
    }

    /// Remember the contents of files that the app now runs.
    pub fn remember(&mut self, files: Vec<(PathBuf, Option<String>)>) {
        for (file, source) in files {
            match source {
                Some(source) => {
                    let _ = self.files.insert(file, source);
                }
                None => {
                    let _ = self.files.remove(&file);
                }
            }
        }
    }
}

/// The definitions in `source` that data depends on, by their path in the file, as tokens
/// without doc comments.
fn definitions(source: &str) -> Option<HashMap<String, String>> {
    let file = syn::parse_file(source).ok()?;
    let mut collector = Collector::default();

    collector.visit_file(&file);

    Some(collector.definitions)
}

#[derive(Default)]
struct Collector {
    /// The modules and functions around the item being visited.
    path: Vec<String>,
    definitions: HashMap<String, String>,
}

impl Collector {
    fn add(&mut self, kind: &str, name: String, item: &impl ToTokens) {
        let mut key = format!("{kind} `{}{name}`", self.prefix());

        // Functions can define items with the same name.
        let mut index = 1;
        while self.definitions.contains_key(&key) {
            index += 1;
            key = format!("{kind} `{}{name}` #{index}", self.prefix());
        }

        let tokens = without_docs(item.to_token_stream()).to_string();
        let _ = self.definitions.insert(key, tokens);
    }

    fn prefix(&self) -> String {
        self.path.iter().map(|part| format!("{part}::")).collect()
    }

    fn nested(&mut self, name: String, visit: impl FnOnce(&mut Self)) {
        self.path.push(name);
        visit(self);
        let _ = self.path.pop();
    }
}

impl<'ast> Visit<'ast> for Collector {
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.add("struct", item.ident.to_string(), item);
    }

    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.add("enum", item.ident.to_string(), item);
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        self.add("union", item.ident.to_string(), item);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.add("type", item.ident.to_string(), item);
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        // Default method bodies are code, only the list of items shapes the vtable.
        let mut signature = item.clone();

        for trait_item in &mut signature.items {
            if let syn::TraitItem::Fn(function) = trait_item {
                function.default = None;
            }
        }

        self.add("trait", item.ident.to_string(), &signature);
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        // `macro_rules!` definitions and item macros like `thread_local!` or `bitflags!`.
        let name = item
            .ident
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| item.mac.path.to_token_stream().to_string());

        self.add("macro", name, item);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        self.nested(item.ident.to_string(), |collector| {
            syn::visit::visit_item_mod(collector, item);
        });
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.nested(item.sig.ident.to_string(), |collector| {
            syn::visit::visit_item_fn(collector, item);
        });
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.nested(item.sig.ident.to_string(), |collector| {
            syn::visit::visit_impl_item_fn(collector, item);
        });
    }
}

/// `tokens` without `#[doc = "..."]` attributes, which doc comments turn into.
fn without_docs(tokens: TokenStream) -> TokenStream {
    let mut output = Vec::new();
    let mut tokens = tokens.into_iter().peekable();

    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(punct) if punct.as_char() == '#' => {
                let is_doc = matches!(
                    tokens.peek(),
                    Some(TokenTree::Group(group))
                        if group.stream().into_iter().next().is_some_and(
                            |first| matches!(first, TokenTree::Ident(ident) if ident == "doc")
                        )
                );

                if is_doc {
                    let _ = tokens.next();
                } else {
                    output.push(TokenTree::Punct(punct));
                }
            }
            TokenTree::Group(group) => {
                let mut stripped =
                    proc_macro2::Group::new(group.delimiter(), without_docs(group.stream()));
                stripped.set_span(group.span());
                output.push(TokenTree::Group(stripped));
            }
            token => output.push(token),
        }
    }

    output.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changed(old: &str, new: &str) -> bool {
        let old = definitions(old).unwrap();
        let new = definitions(new).unwrap();

        old.iter()
            .any(|(name, tokens)| new.get(name).is_some_and(|new| new != tokens))
    }

    #[test]
    fn code_changes_are_fine() {
        assert!(!changed(
            "struct A { x: u32 } fn f() -> u32 { 1 }",
            "struct A { x: u32 } fn f() -> u32 { 2 }",
        ));
        assert!(!changed(
            "/// Old docs.\nstruct A { /// Field.\n x: u32 }",
            "/// New docs.\nstruct A { /// Other.\n x: u32 }",
        ));
        assert!(!changed(
            "trait T { fn f(&self) -> u32 { 1 } }",
            "trait T { fn f(&self) -> u32 { 2 } }",
        ));
        assert!(!changed("struct A;", "struct A; struct B { y: f32 }"));
    }

    #[test]
    fn layout_changes_are_caught() {
        assert!(changed(
            "struct A { x: u32 }",
            "struct A { x: u32, y: u32 }"
        ));
        assert!(changed("enum E { A, B }", "enum E { A, B, C }"));
        assert!(changed(
            "mod m { struct A(u8); }",
            "mod m { struct A(u16); }"
        ));
        assert!(changed(
            "trait T { fn f(&self); }",
            "trait T { fn f(&self); fn g(&self); }"
        ));
        assert!(changed(
            "fn f() { struct Local(u8); }",
            "fn f() { struct Local(u64); }",
        ));
        assert!(changed(
            "thread_local! { static X: u8 = 0; }",
            "thread_local! { static X: u64 = 0; }",
        ));
    }
}
