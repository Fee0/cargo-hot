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
//!
//! Changes in the bodies of plain functions are also told apart from others: those functions
//! are compiled in their own crate only, so a patch redirecting the old functions to the new ones
//! needs nothing else recompiled.
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::visit::Visit;

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// How functions changed since the app last loaded code, which tells how to patch them.
#[derive(Debug, Default)]
pub struct Changes {
    /// See [`Sources::only_bodies_changed`].
    pub only_bodies: bool,
    /// See [`Sources::changed_signatures`].
    pub signatures: BTreeSet<String>,
}

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

    /// Whether all changes in `changed_files` since they were remembered are in the bodies of
    /// functions that are compiled in their own crate only: not generic, not `#[inline]`, not
    /// `const` and not `async` (whose body is a type). Other crates call such functions, but
    /// have none of their code.
    pub fn only_bodies_changed<'a>(
        &self,
        changed_files: impl IntoIterator<Item = &'a PathBuf>,
    ) -> bool {
        changed_files.into_iter().all(|file| {
            let (Some(old), Ok(new)) = (self.files.get(file), std::fs::read_to_string(file)) else {
                return false;
            };

            if *old == new {
                return true;
            }

            let (Some(old), Some(new)) = (items(old), items(&new)) else {
                return false;
            };

            new.iter().all(|(key, new)| match (old.get(key), new) {
                (Some(Item::Other(old)), Item::Other(new)) => old == new,
                (
                    Some(Item::Function {
                        signature: old_signature,
                        body: old_body,
                        ..
                    }),
                    Item::Function {
                        signature,
                        body,
                        is_local,
                    },
                ) => old_signature == signature && (old_body == body || *is_local),
                // A new function is new code.
                (None, Item::Function { is_local, .. }) => *is_local,
                _ => false,
            }) && old.keys().all(|key| new.contains_key(key))
        })
    }

    /// The names of the functions whose signature changed in `changed_files` since they were
    /// remembered. Old code calls them with the old arguments, which the new versions would
    /// read wrongly.
    pub fn changed_signatures<'a>(
        &self,
        changed_files: impl IntoIterator<Item = &'a PathBuf>,
    ) -> BTreeSet<String> {
        let mut names = BTreeSet::new();

        for file in changed_files {
            let (Some(old), Ok(new)) = (self.files.get(file), std::fs::read_to_string(file)) else {
                continue;
            };

            if *old == new {
                continue;
            }

            let (Some(old), Some(new)) = (items(old), items(&new)) else {
                continue;
            };

            for (key, new) in &new {
                if let (
                    Some(Item::Function {
                        signature: old_signature,
                        ..
                    }),
                    Item::Function { signature, .. },
                ) = (old.get(key), new)
                    && old_signature != signature
                {
                    // Keys end with the function's name, and maybe a number telling apart
                    // functions of the same path.
                    let path = key.split(" #").next().unwrap_or(key);
                    let name = path.rsplit("::").next().unwrap_or(path);

                    let _ = names.insert(name.to_owned());
                }
            }
        }

        names
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

/// An item of a source file, as tokens without doc comments.
#[derive(Debug, PartialEq)]
enum Item {
    Function {
        signature: String,
        body: String,
        /// Whether the function is compiled in its crate only (see [`Sources::only_bodies_changed`]).
        is_local: bool,
    },
    Other(String),
}

/// The items of `source`, by their path in the file.
fn items(source: &str) -> Option<HashMap<String, Item>> {
    let file = syn::parse_file(source).ok()?;
    let mut items = HashMap::new();

    collect_items(&file.items, "", &mut items);

    Some(items)
}

fn collect_items(source: &[syn::Item], prefix: &str, items: &mut HashMap<String, Item>) {
    let add = |key: String, item: Item, items: &mut HashMap<String, Item>| {
        let mut unique = key.clone();
        let mut index = 1;

        while items.contains_key(&unique) {
            index += 1;
            unique = format!("{key} #{index}");
        }

        let _ = items.insert(unique, item);
    };

    for item in source {
        match item {
            syn::Item::Fn(function) => add(
                format!("{prefix}{}", function.sig.ident),
                self::function(
                    &function.attrs,
                    &function.vis,
                    &function.sig,
                    &function.block,
                    false,
                ),
                items,
            ),
            syn::Item::Mod(module) if module.content.is_some() => {
                let mut header = module.clone();
                header.content = None;
                add(
                    format!("{prefix}mod {}", module.ident),
                    other(&header),
                    items,
                );

                collect_items(
                    &module.content.as_ref().unwrap().1,
                    &format!("{prefix}{}::", module.ident),
                    items,
                );
            }
            syn::Item::Impl(implementation) => {
                let is_generic = has_type_parameters(&implementation.generics);
                let name = format!(
                    "{prefix}impl {}{}",
                    implementation
                        .trait_
                        .as_ref()
                        .map(|(_, path, _)| format!("{} for ", path.to_token_stream()))
                        .unwrap_or_default(),
                    implementation.self_ty.to_token_stream()
                );

                let mut header = implementation.clone();
                header.items.clear();
                add(name.clone(), other(&header), items);

                for impl_item in &implementation.items {
                    match impl_item {
                        syn::ImplItem::Fn(method) => add(
                            format!("{name}::{}", method.sig.ident),
                            function(
                                &method.attrs,
                                &method.vis,
                                &method.sig,
                                &method.block,
                                is_generic,
                            ),
                            items,
                        ),
                        impl_item => add(format!("{name}::item"), other(impl_item), items),
                    }
                }
            }
            item => add(format!("{prefix}item"), other(item), items),
        }
    }
}

fn function(
    attrs: &[syn::Attribute],
    visibility: &syn::Visibility,
    signature: &syn::Signature,
    body: &syn::Block,
    in_generic_impl: bool,
) -> Item {
    let is_inline = attrs.iter().any(|attr| attr.path().is_ident("inline"));
    let takes_impl_trait = signature
        .inputs
        .to_token_stream()
        .into_iter()
        .any(|token| matches!(token, TokenTree::Ident(ident) if ident == "impl"));

    let mut tokens = TokenStream::new();
    for attr in attrs {
        attr.to_tokens(&mut tokens);
    }
    visibility.to_tokens(&mut tokens);
    signature.to_tokens(&mut tokens);

    Item::Function {
        signature: without_docs(tokens).to_string(),
        body: body.to_token_stream().to_string(),
        is_local: !in_generic_impl
            && !has_type_parameters(&signature.generics)
            && !takes_impl_trait
            && !is_inline
            && signature.constness.is_none()
            && signature.asyncness.is_none(),
    }
}

fn other(item: &impl ToTokens) -> Item {
    Item::Other(without_docs(item.to_token_stream()).to_string())
}

/// Whether `generics` has type or const parameters, which make code generic, unlike lifetimes.
fn has_type_parameters(generics: &syn::Generics) -> bool {
    generics
        .params
        .iter()
        .any(|param| !matches!(param, syn::GenericParam::Lifetime(_)))
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

    fn only_bodies(old: &str, new: &str) -> bool {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("lib.rs");

        std::fs::write(&file, old).unwrap();
        let sources = Sources::read([directory.path()]);
        std::fs::write(&file, new).unwrap();

        sources.only_bodies_changed([&file])
    }

    #[test]
    fn plain_function_bodies_are_local() {
        assert!(only_bodies("fn f() -> u32 { 1 }", "fn f() -> u32 { 2 }"));
        assert!(only_bodies(
            "struct A; impl A { pub fn f(&self) -> u32 { 1 } }",
            "struct A; impl A { pub fn f(&self) -> u32 { 2 } }",
        ));
        assert!(only_bodies(
            "mod m { fn f<'a>(x: &'a str) -> &'a str { x } }",
            "mod m { fn f<'a>(x: &'a str) -> &'a str { &x[1..] } }",
        ));
        assert!(only_bodies("fn f() {}", "fn f() {} fn g() {}"));
        assert!(only_bodies("/// Old.\nfn f() {}", "/// New.\nfn f() {}"));
    }

    #[test]
    fn other_changes_are_not() {
        assert!(!only_bodies(
            "fn f<T>() -> u32 { 1 }",
            "fn f<T>() -> u32 { 2 }"
        ));
        assert!(!only_bodies(
            "fn f(x: impl Into<u32>) -> u32 { 1 }",
            "fn f(x: impl Into<u32>) -> u32 { 2 }",
        ));
        assert!(!only_bodies(
            "struct A<T>(T); impl<T> A<T> { fn f(&self) -> u32 { 1 } }",
            "struct A<T>(T); impl<T> A<T> { fn f(&self) -> u32 { 2 } }",
        ));
        assert!(!only_bodies(
            "#[inline] fn f() -> u32 { 1 }",
            "#[inline] fn f() -> u32 { 2 }"
        ));
        assert!(!only_bodies(
            "const fn f() -> u32 { 1 }",
            "const fn f() -> u32 { 2 }"
        ));
        assert!(!only_bodies(
            "async fn f() -> u32 { 1 }",
            "async fn f() -> u32 { 2 }"
        ));
        assert!(!only_bodies("fn f(x: u8) {}", "fn f(x: u16) {}"));
        assert!(!only_bodies("const X: u32 = 1;", "const X: u32 = 2;"));
        assert!(!only_bodies("fn f() {} fn g() {}", "fn f() {}"));
        assert!(!only_bodies(
            "trait T { fn f(&self) -> u32 { 1 } }",
            "trait T { fn f(&self) -> u32 { 2 } }",
        ));
    }

    fn signatures(old: &str, new: &str) -> Vec<String> {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("lib.rs");

        std::fs::write(&file, old).unwrap();
        let sources = Sources::read([directory.path()]);
        std::fs::write(&file, new).unwrap();

        sources.changed_signatures([&file]).into_iter().collect()
    }

    #[test]
    fn changed_signatures_are_named() {
        assert_eq!(
            signatures(
                "fn f(a: u8) {} fn g() {}",
                "fn f(a: u8, b: u8) {} fn g() {}"
            ),
            ["f"]
        );
        assert_eq!(
            signatures(
                "struct A; impl A { fn f(&self) -> u8 { 1 } }",
                "struct A; impl A { fn f(&self) -> u16 { 1 } }",
            ),
            ["f"]
        );
        assert_eq!(
            signatures("mod m { fn f() {} }", "mod m { fn f(x: u8) {} }"),
            ["f"]
        );
        assert!(signatures("fn f() -> u8 { 1 }", "fn f() -> u8 { 2 }").is_empty());
        assert!(signatures("fn f() {}", "fn f() {} fn g(x: u8) {}").is_empty());
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
