//! The page's decision table, written from the kernel.
//!
//! The page offers a form per decision, and a form needs the same numbers the
//! kernel reads: which byte names the decision, how long its request is, and
//! where each of its fields lies. Those numbers are in the kernel's own
//! modules, as the offsets and widths its answer reads at, and the route's
//! tag for each. Writing them here a second time would be a second source for
//! the one thing that must not differ, so they are taken from there.
//!
//! What a field is *called*, and what a sensible value for it is, is not in
//! the model and cannot be: `site/decision-names.json` holds that, by tag and
//! offset, and this joins the two.
//!
//! The result goes two places, both generated: `site/decisions.json`, which
//! the gate reads, and a block inside `site/card.html`, which the page reads
//! without fetching anything. The page's first paint has to be complete —
//! the frame inlines the card into the HTML and measures what happens after —
//! so the chooser's options and the first decision's fields are written there
//! too.

use std::path::Path;

use crate::Fail;

/// Between these, in `site/card.html`, is written rather than kept.
const OPEN: &str = "<!-- written by `cargo xtask page --write`; do not edit -->";
const CLOSE: &str = "<!-- end -->";

/// One field of one decision, as the kernel reads it.
#[derive(Debug, PartialEq, Eq)]
struct Field {
    at: usize,
    width: usize,
}

/// One decision, as the kernel routes and reads it.
#[derive(Debug)]
struct Decision {
    tag: u8,
    module: String,
    length: usize,
    fields: Vec<Field>,
}

/// Write the table, or check what is written is what the kernel says.
pub fn page(root: &Path, write: bool) -> Result<(), Fail> {
    let decisions = read_kernel(root)?;
    let names: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("site/decision-names.json"),
    )?)
    .map_err(|e| format!("site/decision-names.json: {e}"))?;

    let table = table(&decisions, &names)?;
    let json = format!(
        "{}\n",
        serde_json::to_string_pretty(&table).map_err(|e| e.to_string())?
    );
    let card = card(root, &table, &json)?;

    let json_path = root.join("site/decisions.json");
    let card_path = root.join("site/card.html");
    if write {
        std::fs::write(&json_path, &json)?;
        std::fs::write(&card_path, &card)?;
        println!("page: wrote site/decisions.json and site/card.html from the kernel");
        return Ok(());
    }
    for (path, want) in [(json_path, json), (card_path, card)] {
        let have = std::fs::read_to_string(&path)?;
        if have != want {
            return Err(format!(
                "R1: {} is stale: it disagrees with the kernel's own modules. \
                 Run `cargo xtask page --write`.",
                path.display()
            )
            .into());
        }
    }
    println!("page: site/decisions.json and site/card.html are the kernel's");
    Ok(())
}

/// Every decision the kernel routes, with the length and the fields its answer
/// reads.
fn read_kernel(root: &Path) -> Result<Vec<Decision>, Fail> {
    let entry = std::fs::read_to_string(root.join("app/src/Kernel.lex.tex"))?;
    let mut decisions = Vec::new();
    // The route says which tag reaches which module; the module says the rest.
    for (tag, module) in routed(&entry) {
        let source = std::fs::read_to_string(
            root.join("app/src/Kernel")
                .join(format!("{module}.lex.tex")),
        )?;
        let length = after(&source, "(Foundation.Bytes::byteLength value) =? ")
            .and_then(number)
            .ok_or_else(|| format!("{module}: its answer checks no request length"))?;
        let mut fields: Vec<Field> = Vec::new();
        // The widths the wire format has: a flag, a number, and a label, which
        // is read by a helper of its own rather than by the shared one.
        for (needle, width) in [("Kernel.Wire::fieldAt value ", 0), ("(labelAt value ", 32)] {
            let mut rest = source.as_str();
            while let Some(found) = rest.find(needle) {
                rest = &rest[found + needle.len()..];
                let Some(at) = number(rest) else { continue };
                let width = if width == 0 {
                    let after_at = rest
                        .split_once(' ')
                        .map(|(_, tail)| tail)
                        .unwrap_or_default();
                    match number(after_at) {
                        Some(width) => width,
                        None => continue,
                    }
                } else {
                    width
                };
                if !fields.iter().any(|field| field.at == at) {
                    fields.push(Field { at, width });
                }
            }
        }
        fields.sort_by_key(|field| field.at);
        decisions.push(Decision {
            tag,
            module,
            length,
            fields,
        });
    }
    decisions.sort_by_key(|decision| decision.tag);
    Ok(decisions)
}

/// The tag and module of each `(if (tag =? N) then (Kernel.M::` in the route.
fn routed(entry: &str) -> Vec<(u8, String)> {
    let mut out = Vec::new();
    let mut rest = entry;
    while let Some(found) = rest.find("(if (tag =? ") {
        rest = &rest[found + "(if (tag =? ".len()..];
        let Some(tag) = number(rest).and_then(|tag| u8::try_from(tag).ok()) else {
            continue;
        };
        let Some((_, after_then)) = rest.split_once("then (Kernel.") else {
            continue;
        };
        let Some((module, _)) = after_then.split_once("::") else {
            continue;
        };
        out.push((tag, module.to_string()));
    }
    out
}

/// The text after `needle`, if it is there.
fn after<'a>(haystack: &'a str, needle: &str) -> Option<&'a str> {
    haystack
        .find(needle)
        .map(|at| &haystack[at + needle.len()..])
}

/// The number `text` begins with.
fn number(text: &str) -> Option<usize> {
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The table the page reads: the kernel's numbers, with the names beside them.
fn table(decisions: &[Decision], names: &serde_json::Value) -> Result<serde_json::Value, Fail> {
    let mut out = Vec::new();
    for decision in decisions {
        let named = &names["decisions"][decision.tag.to_string()];
        if named.is_null() {
            return Err(format!(
                "site/decision-names.json: decision {} is routed by the kernel and has no name here",
                decision.tag
            )
            .into());
        }
        let mut fields = Vec::new();
        for field in &decision.fields {
            let named = &named["fields"][field.at.to_string()];
            if named.is_null() {
                return Err(format!(
                    "site/decision-names.json: decision {}'s field at byte {} has no name here",
                    decision.tag, field.at
                )
                .into());
            }
            let mut entry = serde_json::Map::new();
            entry.insert("at".into(), field.at.into());
            entry.insert("width".into(), field.width.into());
            for (key, value) in named.as_object().into_iter().flatten() {
                entry.insert(key.clone(), value.clone());
            }
            fields.push(serde_json::Value::Object(entry));
        }
        out.push(serde_json::json!({
            "tag": decision.tag,
            "module": decision.module,
            "name": named["name"],
            "asks": named["asks"],
            "length": decision.length,
            "fields": fields,
        }));
    }
    Ok(serde_json::json!({
        "comment": "Written by `cargo xtask page --write` from the kernel's own modules \
                    and site/decision-names.json. The offsets, widths, lengths and tags are \
                    the kernel's; the names are the companion file's.",
        "decisions": out,
    }))
}

/// `site/card.html`, with its written part written.
fn card(root: &Path, table: &serde_json::Value, json: &str) -> Result<String, Fail> {
    let existing = std::fs::read_to_string(root.join("site/card.html"))?;
    let (before, rest) = existing
        .split_once(OPEN)
        .ok_or("site/card.html: the written part is not marked")?;
    let (_, after) = rest
        .split_once(CLOSE)
        .ok_or("site/card.html: the written part is not closed")?;

    let decisions = table["decisions"].as_array().cloned().unwrap_or_default();
    let first = decisions.first().cloned().unwrap_or_default();
    let mut written = String::from("\n");
    // The chooser, complete before any script runs.
    written.push_str("  <select id=\"decision\">\n");
    for (index, decision) in decisions.iter().enumerate() {
        written.push_str(&format!(
            "    <option value=\"{index}\">{}. {}</option>\n",
            decision["tag"],
            escape(decision["name"].as_str().unwrap_or_default())
        ));
    }
    written.push_str("  </select>\n");
    // The first decision's question and fields, so the page is whole at once.
    written.push_str(&format!(
        "  <p id=\"asks\" class=\"asks\">{}</p>\n  <div id=\"fields\" class=\"fields\">\n",
        escape(first["asks"].as_str().unwrap_or_default())
    ));
    for field in first["fields"].as_array().into_iter().flatten() {
        written.push_str(&control(field));
    }
    written.push_str("  </div>\n");
    // And the table itself, in the page rather than a request away.
    written.push_str(&format!(
        "  <script type=\"application/json\" id=\"decisions\">{}</script>\n  ",
        json.trim()
    ));
    Ok(format!("{before}{OPEN}{written}{CLOSE}{after}"))
}

/// One field's control, as the page's script would build it.
fn control(field: &serde_json::Value) -> String {
    let name = escape(field["name"].as_str().unwrap_or_default());
    let label = if field["kind"].as_str() == Some("label") {
        " class=\"wide\""
    } else {
        ""
    };
    let mut out = format!("    <label{label}><span class=\"k\">{name}</span>");
    if let Some(choice) = field["choice"].as_array() {
        out.push_str("<select>");
        for pair in choice {
            let text = escape(pair[0].as_str().unwrap_or_default());
            let code = &pair[1];
            let chosen = if *code == field["value"] {
                " selected"
            } else {
                ""
            };
            out.push_str(&format!(
                "<option value=\"{code}\"{chosen}>{text} ({code})</option>"
            ));
        }
        out.push_str("</select>");
    } else {
        let value = match &field["value"] {
            serde_json::Value::String(text) => escape(text),
            other => other.to_string(),
        };
        let kind = if field["kind"].as_str() == Some("label") {
            " class=\"label-in\" spellcheck=\"false\""
        } else {
            " inputmode=\"numeric\""
        };
        out.push_str(&format!(
            "<input autocomplete=\"off\"{kind} value=\"{value}\" />"
        ));
    }
    out.push_str("</label>\n");
    out
}

/// Text that cannot close a tag it is inside.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
