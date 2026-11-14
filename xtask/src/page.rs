//! The page's decision table, written from the kernel.
//!
//! The page offers a form per decision, and a form needs the same numbers the
//! kernel reads: which byte names the decision, how long its request is, and
//! where each of its fields lies. Those numbers are in the kernel's own
//! modules, as the offsets and widths its answer reads at, and the route's
//! tag for each. Writing them here a second time would be a second source for
//! the one thing that must not differ, so they are taken from there.
//!
//! Three shapes of request have to be described, because the kernel has three.
//! A fixed one is its fields and its length. One with a list in it is a fixed
//! part followed by groups of equal items, as many as a field of the request
//! says; where each group begins and how many items it holds are read from the
//! definitions that index into it. One whose format chooses its shape is
//! several forms in one, each the fields of the branch that format takes.
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

/// How the shared reader is written, wherever a field is read.
const READ: &str = "(Kernel.Wire::fieldAt value ";

/// One field of one decision, as the kernel reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Field {
    at: usize,
    width: usize,
}

/// One form of a request whose own format chooses its shape.
#[derive(Debug, PartialEq, Eq)]
struct Shape {
    /// The values of the choosing field this form is for: one branch may
    /// stand for several.
    values: Vec<usize>,
    length: usize,
    fields: Vec<Field>,
}

/// One run of equal items inside a request with a list in it.
///
/// Its first item is at `from + scaled * count`, each item is `width` bytes
/// wide, and it holds `per * count + plus` of them. All of that is read from
/// the definition that indexes into it and from the length its answer accepts.
#[derive(Debug, PartialEq, Eq)]
struct Group {
    reader: String,
    from: usize,
    scaled: usize,
    width: usize,
    per: usize,
    plus: usize,
}

/// How long a request is, as its module says.
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
    /// Several branches, each a shape for one value of one field.
    Chosen { at: usize, shapes: Vec<Shape> },
}

/// One decision, as the kernel routes and reads it.
#[derive(Debug)]
struct Decision {
    tag: u8,
    module: String,
    length: Length,
    fields: Vec<Field>,
    /// The groups of a request with a list in it, in the order they lie.
    groups: Vec<Group>,
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

/// Every decision the kernel routes, with the length, the fields and the
/// groups its answer reads.
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
        for field in reads(&source) {
            if !fields.iter().any(|seen| seen.at == field.at) {
                fields.push(field);
            }
        }
        fields.sort_by_key(|field| field.at);
        // A request with a list in it is drawn from the definitions that index
        // into the list: that is the only place its groups are written down.
        let groups = match &length {
            Length::Counted { base, stride, .. } => {
                groups_of(&source, *base, *stride).ok_or_else(|| {
                    format!(
                        "{module}: its answer accepts a counted length, and what repeats cannot \
                         be read from the definitions that index into it"
                    )
                })?
            }
            _ => Vec::new(),
        };
        decisions.push(Decision {
            tag,
            module,
            length,
            fields,
            groups,
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

/// Every field `text` reads at a fixed offset, in the order it reads them.
///
/// An offset written as an expression rather than a number is an index into a
/// list, which is no field of the form; `groups_of` reads those.
fn reads(text: &str) -> Vec<Field> {
    let mut out = Vec::new();
    // The widths the wire format has: a flag, a number, and a label, which is
    // read by a helper of its own rather than by the shared one.
    for (needle, fixed) in [(READ, 0), ("(labelAt value ", 32)] {
        let mut rest = text;
        while let Some(found) = rest.find(needle) {
            rest = &rest[found + needle.len()..];
            let Some(at) = number(rest) else { continue };
            let width = if fixed == 0 {
                let after_at = rest
                    .split_once(' ')
                    .map(|(_, tail)| tail)
                    .unwrap_or_default();
                match number(after_at) {
                    Some(width) => width,
                    None => continue,
                }
            } else {
                fixed
            };
            out.push(Field { at, width });
        }
    }
    out
}

/// The parenthesised group `text` begins with, its parentheses included.
fn group(text: &str) -> Option<&str> {
    if !text.starts_with('(') {
        return None;
    }
    let mut depth = 0usize;
    for (at, byte) in text.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[..=at]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The condition and the taken branch of the `(if C then T else E)` that
/// `text` begins with.
fn if_parts(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("(if ")?;
    let condition = group(rest)?;
    let taken = rest[condition.len()..].strip_prefix(" then ")?;
    let taken = group(taken).or_else(|| taken.split(" else ").next())?;
    Some((condition, taken))
}

/// One length check: the length, the field and the values of the branch it
/// stands in, and every field that branch reads.
type Branch = (usize, Option<(usize, Vec<usize>)>, Vec<Field>);

/// How long a request is, from every branch of the module that checks it.
///
/// One branch is a fixed length, unless the length it checks is of the counted
/// form, in which case the request has a list in it. Several branches, each
/// beside a test of the same field, are one shape per value of that field.
fn length_of(source: &str) -> Option<Length> {
    const CHECK: &str = "(Foundation.Bytes::byteLength value) =? ";
    let mut found: Vec<Branch> = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find("(if ") {
        let here = &rest[at..];
        rest = &rest[at + "(if ".len()..];
        let Some((condition, taken)) = if_parts(here) else {
            continue;
        };
        let Some(after_check) = after(condition, CHECK) else {
            continue;
        };
        if let Some(counted) = counted(after_check) {
            return Some(counted);
        }
        let Some(fixed) = number(after_check) else {
            continue;
        };
        // What this branch reads is the form the page draws for it: the
        // condition's own reads as well as the branch's, since a request
        // carries every byte the kernel looks at to answer it.
        let mut fields: Vec<Field> = Vec::new();
        for field in reads(condition).into_iter().chain(reads(taken)) {
            if !fields.iter().any(|seen| seen.at == field.at) {
                fields.push(field);
            }
        }
        fields.sort_by_key(|field| field.at);
        found.push((fixed, chosen_in(condition), fields));
    }
    match found.as_slice() {
        [] => None,
        [(fixed, _, _)] => Some(Length::Fixed(*fixed)),
        several => {
            let at = several
                .iter()
                .find_map(|(_, chosen, _)| chosen.as_ref().map(|(at, _)| *at))?;
            let mut shapes = Vec::new();
            for (length, chosen, fields) in several {
                let (field, values) = chosen.as_ref()?;
                if *field != at {
                    return None;
                }
                let mut values = values.clone();
                values.sort_unstable();
                shapes.push(Shape {
                    values,
                    length: *length,
                    fields: fields.clone(),
                });
            }
            shapes.sort_by_key(|shape| shape.values.first().copied().unwrap_or_default());
            Some(Length::Chosen { at, shapes })
        }
    }
}

/// `(B + (prim multiply S (Kernel.Wire::fieldAt value A W) : Nat))`.
fn counted(text: &str) -> Option<Length> {
    let text = text.strip_prefix('(')?;
    let base = number(text)?;
    let rest = after(text, " + (prim multiply ")?;
    let stride = number(rest)?;
    let count_at = number(after(rest, READ)?)?;
    Some(Length::Counted {
        base,
        stride,
        count_at,
    })
}

/// The field a branch tests, and every value of it this branch is for.
///
/// A branch may be for one value or for a choice between several, written as
/// `((f =? 2) || (f =? 3))`, and both are one shape.
fn chosen_in(condition: &str) -> Option<(usize, Vec<usize>)> {
    let mut field = None;
    let mut values = Vec::new();
    let mut rest = condition;
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

/// The groups of a request with a list in it, from the definitions that index
/// into it.
///
/// A definition whose body is `fieldAt value (F + (prim multiply W i : Nat)) W`
/// reads the items of a group that begins at `F`, and one whose body is
/// `fieldAt value ((F + (prim multiply S c : Nat)) + (prim multiply W i : Nat)) W`
/// reads a group that begins after `c` items of an earlier one. How many items
/// a group holds is then the distance to the next group, or to the end of the
/// request, which is the length the answer accepts.
fn groups_of(source: &str, base: usize, stride: usize) -> Option<Vec<Group>> {
    let mut starts: Vec<(String, usize, usize, usize)> = Vec::new();
    for block in source.split("definition{").skip(1) {
        let Some(reader) = after(block, "name: ").and_then(|rest| rest.split(',').next()) else {
            continue;
        };
        let Some(body) = after(block, "body: ").and_then(group) else {
            continue;
        };
        let Some(rest) = body.strip_prefix(READ) else {
            continue;
        };
        // A field read at a fixed offset is a field of the form, not a list.
        let Some(offset) = group(rest) else { continue };
        let Some(width) = number(rest[offset.len()..].trim_start()) else {
            continue;
        };
        let Some(from) = number(offset.trim_start_matches('(')) else {
            continue;
        };
        let mut steps = Vec::new();
        let mut rest = offset;
        while let Some(found) = after(rest, "(prim multiply ") {
            let Some(step) = number(found) else { break };
            steps.push(step);
            rest = found;
        }
        let (scaled, step) = match steps.as_slice() {
            [step] => (0, *step),
            [scaled, step] => (*scaled, *step),
            _ => continue,
        };
        // A group whose step is not its item's width is one the page cannot
        // tile: its items would overlap, or leave a byte nothing writes.
        if step != width {
            return None;
        }
        starts.push((reader.trim().to_string(), from, scaled, width));
    }
    starts.sort_by_key(|(_, from, scaled, _)| (*scaled, *from));
    let mut groups = Vec::new();
    for (index, (reader, from, scaled, width)) in starts.iter().enumerate() {
        // Where this group ends: the next one begins there, and the last ends
        // where the request does.
        let next = starts
            .get(index + 1)
            .map(|(_, from, scaled, _)| (*from, *scaled))
            .unwrap_or((base, stride));
        let plus = next.0.checked_sub(*from)?;
        let per = next.1.checked_sub(*scaled)?;
        if plus % width != 0 || per % width != 0 {
            return None;
        }
        groups.push(Group {
            reader: reader.clone(),
            from: *from,
            scaled: *scaled,
            width: *width,
            per: per / width,
            plus: plus / width,
        });
    }
    (!groups.is_empty()).then_some(groups)
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
        let tag = decision.tag;
        let named = &names["decisions"][tag.to_string()];
        if named.is_null() {
            return Err(format!(
                "site/decision-names.json: decision {tag} is routed by the kernel and has no \
                 name here"
            )
            .into());
        }
        let mut fields = Vec::new();
        for field in &decision.fields {
            fields.push(described(tag, field, named, &[])?);
        }
        let length = match &decision.length {
            Length::Fixed(length) => serde_json::json!(length),
            Length::Counted {
                base,
                stride,
                count_at,
            } => serde_json::json!({ "base": base, "stride": stride, "countAt": count_at }),
            Length::Chosen { at, shapes } => {
                let mut by_value = serde_json::Map::new();
                for shape in shapes {
                    for value in &shape.values {
                        by_value.insert(value.to_string(), serde_json::json!(shape.length));
                    }
                }
                serde_json::json!({ "at": at, "lengths": by_value })
            }
        };
        let mut entry = serde_json::Map::new();
        entry.insert("tag".into(), tag.into());
        entry.insert("module".into(), decision.module.clone().into());
        entry.insert("name".into(), named["name"].clone());
        entry.insert("asks".into(), named["asks"].clone());
        entry.insert("length".into(), length);
        entry.insert("fields".into(), fields.into());
        // A form per format, each the branch's own fields: the same byte may
        // be a word of codes to one format and a byte of signs to another, so
        // one list of fields could not draw them both.
        if let Length::Chosen { shapes, .. } = &decision.length {
            let mut forms = Vec::new();
            for shape in shapes {
                let mut fields = Vec::new();
                for field in &shape.fields {
                    fields.push(described(tag, field, named, &shape.values)?);
                }
                forms.push(serde_json::json!({
                    "values": shape.values,
                    "length": shape.length,
                    "fields": fields,
                }));
            }
            entry.insert("shapes".into(), forms.into());
        }
        // And the list, group by group, as the kernel indexes into it.
        if let Length::Counted { count_at, .. } = &decision.length {
            let mut groups = Vec::new();
            for group in &decision.groups {
                let named = &named["list"][&group.reader];
                if named.is_null() {
                    return Err(format!(
                        "site/decision-names.json: decision {tag} repeats what \
                         `{}` reads, and that has no name here",
                        group.reader
                    )
                    .into());
                }
                let mut item = serde_json::Map::new();
                item.insert("reader".into(), group.reader.clone().into());
                item.insert("from".into(), group.from.into());
                item.insert("scaled".into(), group.scaled.into());
                item.insert("width".into(), group.width.into());
                item.insert("per".into(), group.per.into());
                item.insert("plus".into(), group.plus.into());
                for (key, value) in named.as_object().into_iter().flatten() {
                    item.insert(key.clone(), value.clone());
                }
                groups.push(serde_json::Value::Object(item));
            }
            entry.insert(
                "list".into(),
                serde_json::json!({ "countAt": count_at, "groups": groups }),
            );
        }
        out.push(serde_json::Value::Object(entry));
    }
    Ok(serde_json::json!({
        "comment": "Written by `cargo xtask page --write` from the kernel's own modules \
                    and site/decision-names.json. The offsets, widths, lengths, groups and \
                    tags are the kernel's; the names are the companion file's.",
        "decisions": out,
    }))
}

/// One field, as the kernel reads it and the companion file names it.
///
/// `formats` are the values of the chooser this field is drawn for, if it
/// belongs to one form of a request whose format chooses its shape: a byte
/// that two formats read differently is named by each of them.
fn described(
    tag: u8,
    field: &Field,
    named: &serde_json::Value,
    formats: &[usize],
) -> Result<serde_json::Value, Fail> {
    let mut entry = serde_json::Map::new();
    entry.insert("at".into(), field.at.into());
    entry.insert("width".into(), field.width.into());
    let mut found = false;
    for source in std::iter::once(&named["fields"][field.at.to_string()]).chain(
        formats
            .iter()
            .map(|format| &named["shapes"][format.to_string()][field.at.to_string()]),
    ) {
        for (key, value) in source.as_object().into_iter().flatten() {
            entry.insert(key.clone(), value.clone());
            found = true;
        }
    }
    if !found {
        return Err(format!(
            "site/decision-names.json: decision {tag}'s field at byte {} has no name here",
            field.at
        )
        .into());
    }
    Ok(serde_json::Value::Object(entry))
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

    /// A group, written as the kernel's own modules write it.
    fn group_of(reader: &str, from: usize, scaled: usize, per: usize, plus: usize) -> Group {
        Group {
            reader: reader.to_string(),
            from,
            scaled,
            width: 4,
            per,
            plus,
        }
    }

    /// The length checks of decisions this repository already carries, and of
    /// the six a browser engine adds, quoted from their modules. A length this
    /// parser cannot read is a decision the page cannot draw, so each shape is
    /// here before the decision that needs it arrives.
    #[test]
    fn every_length_the_kernel_writes_is_read() {
        let fixed = "(if (((Foundation.Bytes::byteLength value) =? 25) && \
                     ((Kernel.Wire::fieldAt value 1 8) <=? 1)) then (a) else (b))";
        assert_eq!(length_of(fixed), Some(Length::Fixed(25)));

        // A list: a base, a stride and the field holding the count.
        let pick = "(if (((Foundation.Bytes::byteLength value) =? (7 + (prim multiply 4 \
                    (Kernel.Wire::fieldAt value 6 1) : Nat))) && ((Kernel.Wire::fieldAt value 1 \
                    1) <? (Kernel.Wire::fieldAt value 6 1))) then (a) else (b))";
        assert_eq!(
            length_of(pick),
            Some(Length::Counted {
                base: 7,
                stride: 4,
                count_at: 6
            })
        );
        let penalty = "(if ((Foundation.Bytes::byteLength value) =? (8 + (prim multiply 4 \
                       (Kernel.Wire::fieldAt value 7 1) : Nat))) then (a) else (b))";
        assert_eq!(
            length_of(penalty),
            Some(Length::Counted {
                base: 8,
                stride: 4,
                count_at: 7
            })
        );
        let speculate = "(if (((Foundation.Bytes::byteLength value) =? (7 + (prim multiply 8 \
                         (Kernel.Wire::fieldAt value 1 1) : Nat))) && ((1 <=? \
                         (Kernel.Wire::fieldAt value 2 1)) && ((Kernel.Wire::fieldAt value 2 1) \
                         <=? ((Kernel.Wire::fieldAt value 1 1) + 1)))) then (a) else (b))";
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
    /// for two formats, and a form per format: the fields each branch reads,
    /// which are not the same fields.
    #[test]
    fn a_shape_a_format_chooses_is_read() {
        let quant = "(if (((Kernel.Wire::fieldAt value 1 1) =? 0) && (((Foundation.Bytes::\
                     byteLength value) =? 7) && ((Kernel.Wire::fieldAt value 2 1) <? 8))) then \
                     (signedText (nibbleAt (Kernel.Wire::fieldAt value 3 4) (Kernel.Wire::\
                     fieldAt value 2 1)) 8) else (if (((Kernel.Wire::fieldAt value 1 1) =? 1) && \
                     (((Foundation.Bytes::byteLength value) =? 15) && ((Kernel.Wire::fieldAt \
                     value 2 1) <? 32))) then (fieldCode (Kernel.Wire::fieldAt value 3 4) \
                     (Kernel.Wire::fieldAt value 7 4) (Kernel.Wire::fieldAt value 11 4)) else \
                     (if ((((Kernel.Wire::fieldAt value 1 1) =? 2) || ((Kernel.Wire::fieldAt \
                     value 1 1) =? 3)) && (((Foundation.Bytes::byteLength value) =? 7) && \
                     ((Kernel.Wire::fieldAt value 2 1) <? 16))) then (pairAt (Kernel.Wire::\
                     fieldAt value 3 4)) else (if (((Kernel.Wire::fieldAt value 1 1) =? 4) && \
                     (((Foundation.Bytes::byteLength value) =? 5) && ((Kernel.Wire::fieldAt \
                     value 2 1) <? 8))) then (ternaryCode (bitAt (Kernel.Wire::fieldAt value 3 \
                     1)) (bitAt (Kernel.Wire::fieldAt value 4 1))) else (invalid))))";
        let read = length_of(quant).expect("the format chooses the shape");
        let Length::Chosen { at, shapes } = read else {
            panic!("a length one field chooses is not fixed or counted: {read:?}");
        };
        assert_eq!(at, 1);
        assert_eq!(
            shapes
                .iter()
                .map(|s| (&s.values, s.length))
                .collect::<Vec<_>>(),
            vec![
                (&vec![0], 7usize),
                (&vec![1], 15),
                (&vec![2, 3], 7),
                (&vec![4], 5)
            ],
            "one branch stands for two formats, and each shape has its own length"
        );
        // The same byte is four bytes of codes to one format and one byte of
        // signs to another, which is why a form per format is read.
        assert_eq!(
            shapes[0].fields,
            vec![
                Field { at: 1, width: 1 },
                Field { at: 2, width: 1 },
                Field { at: 3, width: 4 }
            ]
        );
        assert_eq!(
            shapes[3].fields,
            vec![
                Field { at: 1, width: 1 },
                Field { at: 2, width: 1 },
                Field { at: 3, width: 1 },
                Field { at: 4, width: 1 }
            ]
        );
    }

    /// One group, from the definition that indexes into it: every item of it
    /// lies where the kernel reads it, and there are as many as the length
    /// leaves room for.
    #[test]
    fn one_group_is_read_from_the_definition_that_indexes_it() {
        let pick = "definition{\n  name: scoreAt,\n  parameters: [value : Bytes, position : \
                    Nat],\n  body: (Kernel.Wire::fieldAt value (7 + (prim multiply 4 position : \
                    Nat)) 4),\n}\n";
        assert_eq!(
            groups_of(pick, 7, 4),
            Some(vec![group_of("scoreAt", 7, 0, 1, 0)]),
            "one group of as many items as the count"
        );
    }

    /// Two groups, the second after as many items as the first holds, and one
    /// item longer: the extra item is the base the length leaves over.
    #[test]
    fn two_groups_follow_each_other_through_the_count() {
        let speculate = "definition{\n  name: draftAt,\n  body: (Kernel.Wire::fieldAt value (3 + \
                         (prim multiply 4 position : Nat)) 4),\n}\ndefinition{\n  name: \
                         verifiedAt,\n  body: (Kernel.Wire::fieldAt value ((3 + (prim multiply 4 \
                         window : Nat)) + (prim multiply 4 position : Nat)) 4),\n}\n";
        assert_eq!(
            groups_of(speculate, 7, 8),
            Some(vec![
                group_of("draftAt", 3, 0, 1, 0),
                group_of("verifiedAt", 3, 4, 1, 1)
            ])
        );
    }

    /// A list whose items do not account for the length the answer accepts is
    /// one the page cannot write, and says so rather than guessing.
    #[test]
    fn a_list_that_does_not_fill_the_request_is_no_list() {
        let pick = "definition{\n  name: scoreAt,\n  body: (Kernel.Wire::fieldAt value (7 + (prim \
                    multiply 4 position : Nat)) 4),\n}\n";
        assert_eq!(
            groups_of(pick, 7, 6),
            None,
            "a stride of six, items of four"
        );
        assert_eq!(
            groups_of(pick, 3, 4),
            None,
            "a base before the group starts"
        );
        assert_eq!(groups_of("definition{\n  name: nothing,\n}\n", 7, 4), None);
    }

    /// A module whose answer checks no length at all is a module this cannot
    /// describe, and says so rather than guessing one.
    #[test]
    fn a_module_with_no_length_check_is_no_decision() {
        assert_eq!(length_of("(if (something else) then (a) else (b))"), None);
    }

    /// And the kernel this repository carries, read whole: every decision it
    /// routes has a length, a field for every byte of its fixed part, and, if
    /// it has a list in it, groups that account for the rest.
    #[test]
    fn the_kernel_in_this_repository_reads() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is one below the root");
        let decisions = read_kernel(root).expect("the kernel's own modules");
        assert!(
            decisions.len() >= 17,
            "the kernel routes {} decisions",
            decisions.len()
        );
        for decision in &decisions {
            let Length::Counted { base, stride, .. } = decision.length else {
                continue;
            };
            let first = decision
                .groups
                .first()
                .expect("a counted length has a list");
            let fixed = decision
                .fields
                .iter()
                .map(|field| field.at + field.width)
                .max()
                .unwrap_or_default();
            assert_eq!(
                (first.from, first.scaled),
                (fixed, 0),
                "{}: its list does not start where its fields end",
                decision.module
            );
            let constant: usize = decision
                .groups
                .iter()
                .map(|group| group.width * group.plus)
                .sum();
            let counted: usize = decision
                .groups
                .iter()
                .map(|group| group.width * group.per)
                .sum();
            assert_eq!(
                (first.from + constant, counted),
                (base, stride),
                "{}: its groups do not fill the length its answer accepts",
                decision.module
            );
        }
    }
}
