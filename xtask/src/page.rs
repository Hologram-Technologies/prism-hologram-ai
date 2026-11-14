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

/// How long a request is, as its module says.
///
/// Three forms, because the kernel has three. A decision with a list in it
/// cannot have one length, and a decision whose shape depends on a format
/// cannot either; both say so in the same place, the length its answer
/// accepts.
#[derive(Debug, PartialEq, Eq)]
enum Length {
    /// `=? 25`
    Fixed(usize),
    /// `=? (7 + (prim multiply 4 (Kernel.Wire::fieldAt value 6 1) : Nat))`
    Counted {
        base: usize,
        stride: usize,
        count_at: usize,
    },
    /// Several branches, each a length for one value of one field.
    Chosen {
        at: usize,
        lengths: Vec<(usize, usize)>,
    },
}

/// One decision, as the kernel routes and reads it.
#[derive(Debug)]
struct Decision {
    tag: u8,
    module: String,
    length: Length,
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
        let length = length_of(&source)
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

/// One length check, and the field and values of the branch it stands in.
type Branch = (usize, Option<(usize, Vec<usize>)>);

/// How long a request is, from every place the module checks its length.
///
/// One check is a fixed length. One check of the counted form is a list. More
/// than one check, each beside a test of the same field, is a length the
/// format chooses.
fn length_of(source: &str) -> Option<Length> {
    const CHECK: &str = "(Foundation.Bytes::byteLength value) =? ";
    let mut found: Vec<Branch> = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find(CHECK) {
        let before = &rest[..at];
        rest = &rest[at + CHECK.len()..];
        if let Some(fixed) = number(rest) {
            // What value of what field this branch is for, if it is one of
            // several: the test stands just before the length check.
            found.push((fixed, chosen_for(before)));
            continue;
        }
        if let Some(counted) = counted(rest) {
            return Some(counted);
        }
    }
    match found.as_slice() {
        [] => None,
        [(fixed, _)] => Some(Length::Fixed(*fixed)),
        several => {
            let at = several
                .iter()
                .find_map(|(_, chosen)| chosen.as_ref().map(|(at, _)| *at))?;
            let mut lengths = Vec::new();
            for (length, chosen) in several {
                let (field, values) = chosen.as_ref()?;
                if *field != at {
                    return None;
                }
                // One branch may be for several values, written as a choice
                // between them: each of them has this length.
                for value in values {
                    lengths.push((*value, *length));
                }
            }
            lengths.sort_unstable();
            Some(Length::Chosen { at, lengths })
        }
    }
}

/// `(B + (prim multiply S (Kernel.Wire::fieldAt value A W) : Nat))`.
fn counted(text: &str) -> Option<Length> {
    let text = text.strip_prefix('(')?;
    let base = number(text)?;
    let rest = after(text, " + (prim multiply ")?;
    let stride = number(rest)?;
    let count_at = number(after(rest, "(Kernel.Wire::fieldAt value ")?)?;
    Some(Length::Counted {
        base,
        stride,
        count_at,
    })
}

/// The field a branch tests, and every value of it this branch is for.
///
/// A branch may be for one value or for a choice between several, written as
/// `((f =? 2) || (f =? 3))`, and both are one length. The branch is taken to
/// begin at the nearest `if` before the length check, so a test from an
/// earlier branch is not read as part of this one.
fn chosen_for(before: &str) -> Option<(usize, Vec<usize>)> {
    const READ: &str = "(Kernel.Wire::fieldAt value ";
    let branch = before.rfind("if (").map_or(before, |at| &before[at..]);
    let mut field = None;
    let mut values = Vec::new();
    let mut rest = branch;
    while let Some(at) = rest.find(READ) {
        rest = &rest[at + READ.len()..];
        let Some(read) = number(rest) else { continue };
        let Some(value) = after(rest, ") =? ").and_then(number) else {
            continue;
        };
        // Only the field this branch chooses on: a test of another field in
        // the same branch is a condition, not the choice.
        match field {
            None => field = Some(read),
            Some(field) if field == read => {}
            Some(_) => continue,
        }
        values.push(value);
    }
    field.filter(|_| !values.is_empty()).map(|at| (at, values))
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
        let length = match &decision.length {
            Length::Fixed(length) => serde_json::json!(length),
            Length::Counted {
                base,
                stride,
                count_at,
            } => serde_json::json!({ "base": base, "stride": stride, "countAt": count_at }),
            Length::Chosen { at, lengths } => {
                let mut by_value = serde_json::Map::new();
                for (value, length) in lengths {
                    by_value.insert(value.to_string(), serde_json::json!(length));
                }
                serde_json::json!({ "at": at, "lengths": by_value })
            }
        };
        let mut entry = serde_json::Map::new();
        entry.insert("tag".into(), decision.tag.into());
        entry.insert("module".into(), decision.module.clone().into());
        entry.insert("name".into(), named["name"].clone());
        entry.insert("asks".into(), named["asks"].clone());
        entry.insert("length".into(), length);
        entry.insert("fields".into(), fields.into());
        // What repeats, and what the format chooses, are the companion file's
        // to say: the module's length expression says there is a list, not
        // what is in it. The gate checks the two agree.
        for key in ["repeat", "shapes"] {
            if !named[key].is_null() {
                entry.insert(key.into(), named[key].clone());
            }
        }
        out.push(serde_json::Value::Object(entry));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The length checks of decisions this repository already carries, and of
    /// the six a browser engine adds, quoted from their modules. A length this
    /// parser cannot read is a decision the page cannot draw, so each shape is
    /// here before the decision that needs it arrives.
    #[test]
    fn every_length_the_kernel_writes_is_read() {
        let fixed = "(if (((Foundation.Bytes::byteLength value) =? 25) && ((Kernel.Wire::fieldAt \
                     value 1 8) <=? 1)) then";
        assert_eq!(length_of(fixed), Some(Length::Fixed(25)));

        // A list: a base, a stride and the field holding the count.
        let pick = "(if ((((Foundation.Bytes::byteLength value) =? (7 + (prim multiply 4 \
                    (Kernel.Wire::fieldAt value 6 1) : Nat))) && ((Kernel.Wire::fieldAt value 1 \
                    1) <? (Kernel.Wire::fieldAt value 6 1))) then";
        assert_eq!(
            length_of(pick),
            Some(Length::Counted {
                base: 7,
                stride: 4,
                count_at: 6
            })
        );
        let penalty = "(if (((Foundation.Bytes::byteLength value) =? (8 + (prim multiply 4 \
                       (Kernel.Wire::fieldAt value 7 1) : Nat))) then";
        assert_eq!(
            length_of(penalty),
            Some(Length::Counted {
                base: 8,
                stride: 4,
                count_at: 7
            })
        );
        let speculate = "(if ((((Foundation.Bytes::byteLength value) =? (7 + (prim multiply 8 \
                         (Kernel.Wire::fieldAt value 1 1) : Nat))) && ((1 <=? \
                         (Kernel.Wire::fieldAt value 2 1)) && ((Kernel.Wire::fieldAt value 2 1) \
                         <=? ((Kernel.Wire::fieldAt value 1 1) + 1)))) then";
        assert_eq!(
            length_of(speculate),
            Some(Length::Counted {
                base: 7,
                stride: 8,
                count_at: 1
            })
        );
    }

    /// A length the request's own first byte chooses, with one branch standing
    /// for two formats.
    #[test]
    fn a_length_a_format_chooses_is_read() {
        let quant = "(if (((Kernel.Wire::fieldAt value 1 1) =? 0) && (((Foundation.Bytes::\
                     byteLength value) =? 7) && ((Kernel.Wire::fieldAt value 2 1) <? 8))) then \
                     (a) else (if (((Kernel.Wire::fieldAt value 1 1) =? 1) && \
                     (((Foundation.Bytes::byteLength value) =? 15) && ((Kernel.Wire::fieldAt \
                     value 2 1) <? 32))) then (b) else (if ((((Kernel.Wire::fieldAt value 1 1) \
                     =? 2) || ((Kernel.Wire::fieldAt value 1 1) =? 3)) && \
                     (((Foundation.Bytes::byteLength value) =? 7) && ((Kernel.Wire::fieldAt \
                     value 2 1) <? 16))) then (c) else (if (((Kernel.Wire::fieldAt value 1 1) \
                     =? 4) && (((Foundation.Bytes::byteLength value) =? 5) && \
                     ((Kernel.Wire::fieldAt value 2 1) <? 8))) then (d) else (e))))";
        assert_eq!(
            length_of(quant),
            Some(Length::Chosen {
                at: 1,
                lengths: vec![(0, 7), (1, 15), (2, 7), (3, 7), (4, 5)]
            }),
            "the format chooses the length, and one branch stands for two formats"
        );
    }

    /// A module whose answer checks no length at all is a module this cannot
    /// describe, and says so rather than guessing one.
    #[test]
    fn a_module_with_no_length_check_is_no_decision() {
        assert_eq!(length_of("(if (something else) then (a) else (b))"), None);
    }
}
