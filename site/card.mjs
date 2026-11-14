// The app's own surface: pick a decision, fill in its numbers, and the
// verified guest answers.
//
// Every layout comes from the kernel itself: `cargo xtask page --write` reads
// the offsets, widths, lengths, groups and tags out of `app/src/Kernel` and
// writes them into this page and into site/decisions.json, which `EN-14`
// checks against the kernel's own examples. A field at the wrong offset or of
// the wrong width fails the gate rather than quietly asking the kernel
// something else. Nothing in this file decides anything; it writes the request
// the kernel reads and shows the answer the kernel gives.
//
// A request has one of three shapes, because the kernel reads three. Most are
// a fixed set of fields. Some carry a list: a fixed part, then groups of equal
// items, as many as one field of the request says. One has a shape per format:
// its own second byte says which form the rest of it takes, and the forms do
// not read the same bytes.

const BYTE = 256n;

/// The most items of one group this form draws. The kernel takes as many as a
/// byte can count; a form of 255 boxes is not a form, so the page says what it
/// offers rather than writing a request nobody filled in.
const MOST = 64;

/**
 * Which form of `decision` the values on the page ask for: how long the
 * request is, which fields it carries, and the items of its list.
 *
 * Exported with `request` so `EN-14` can run both over the kernel's own
 * numbers: the page's reading of the table is the thing being checked.
 */
export function shapeOf(decision, values) {
  const length = decision.length;
  if (typeof length === "number") {
    return { length, fields: decision.fields, items: [] };
  }
  // A list: the request is a base plus a stride for every item counted.
  if (length.base !== undefined) {
    const count = whole(decision, values, length.countAt);
    if (count > MOST) {
      throw new Error(`${named(decision, length.countAt)}: from 0 to ${MOST} on this page`);
    }
    return {
      length: length.base + length.stride * count,
      fields: decision.fields,
      items: items(decision, count),
      drives: length.countAt,
    };
  }
  // A shape its own format chooses: one form per value of one field.
  const format = whole(decision, values, length.at);
  const shape = (decision.shapes ?? []).find((shape) => shape.values.includes(format));
  if (!shape) {
    throw new Error(`${named(decision, length.at)}: the kernel reads no request of format ${format}`);
  }
  return { length: shape.length, fields: shape.fields, items: [], drives: length.at };
}

/** Every item of every group of `decision`'s list, where the kernel reads it. */
function items(decision, count) {
  const out = [];
  for (const group of decision.list?.groups ?? []) {
    const many = group.per * count + group.plus;
    for (let index = 0; index < many; index += 1) {
      out.push({
        group: group.reader,
        name: `${group.name} ${index}`,
        label: String(index),
        at: group.from + group.scaled * count + group.width * index,
        width: group.width,
        value: group.values?.[index] ?? group.value,
      });
    }
  }
  return out;
}

/** The field of `decision` at byte `at`. */
function field(decision, at) {
  const found = decision.fields.find((field) => field.at === at);
  if (!found) throw new Error(`the request has no field at byte ${at}`);
  return found;
}

/** What that field is called. */
function named(decision, at) {
  return field(decision, at).name;
}

/** The value on the page for the field at byte `at`, as a whole number. */
function whole(decision, values, at) {
  const found = field(decision, at);
  const given = values.get(found.name);
  // An empty box is a zero, here and in `request`: were the two to read it
  // differently, the form drawn would not be the form written.
  const text = given === undefined ? String(found.value) : String(given).trim();
  const number = Number(text === "" ? "0" : text);
  if (!Number.isSafeInteger(number) || number < 0) {
    throw new Error(`${found.name}: a whole number`);
  }
  return number;
}

/**
 * The request bytes for `decision`, from the values on the page.
 *
 * Exported so `EN-14` can run it over the kernel's own examples: the page's
 * encoder is the thing being checked, not a copy of it.
 */
export function request(decision, values) {
  const shape = shapeOf(decision, values);
  const out = new Uint8Array(shape.length);
  out[0] = decision.tag;
  for (const field of [...shape.fields, ...shape.items]) {
    const value = values.has(field.name) ? values.get(field.name) : field.value;
    if (field.kind === "label") {
      const digits = String(value).replace(/\s+/g, "").toLowerCase();
      if (digits.length !== field.width * 2 || /[^0-9a-f]/.test(digits)) {
        throw new Error(`${field.name}: ${field.width * 2} hexadecimal digits`);
      }
      for (let i = 0; i < field.width; i += 1) {
        out[field.at + i] = parseInt(digits.slice(i * 2, i * 2 + 2), 16);
      }
      continue;
    }
    let number;
    try {
      number = BigInt(String(value).trim() === "" ? "0" : String(value).trim());
    } catch {
      throw new Error(`${field.name}: a whole number`);
    }
    const limit = BYTE ** BigInt(field.width);
    if (number < 0n || number >= limit) {
      throw new Error(`${field.name}: from 0 to ${limit - 1n}`);
    }
    for (let i = field.width - 1; i >= 0; i -= 1) {
      out[field.at + i] = Number(number % BYTE);
      number /= BYTE;
    }
  }
  return out;
}

/** The input for one field, and how to read its value back. */
function input(field) {
  const label = document.createElement("label");
  const name = document.createElement("span");
  name.className = "k";
  name.textContent = field.label ?? field.name;
  label.append(name);

  let control;
  if (field.choice) {
    control = document.createElement("select");
    for (const [text, code] of field.choice) {
      const option = document.createElement("option");
      option.value = String(code);
      option.textContent = `${text} (${code})`;
      control.append(option);
    }
  } else {
    control = document.createElement("input");
    control.autocomplete = "off";
    control.spellcheck = false;
    if (field.kind === "label") {
      control.className = "label-in";
      label.classList.add("wide");
    } else {
      control.inputMode = "numeric";
    }
  }
  control.value = String(field.value ?? 0);
  label.append(control);
  return { label, control };
}

export default async function start({ app, reason, $, chip, show }) {
  // The table is in the page, not a request away: `cargo xtask page --write`
  // writes it there from the kernel, along with the chooser's options and the
  // first decision's fields. The page is whole before this script runs, which
  // is what the frame measures.
  const decisions = JSON.parse($("decisions").textContent).decisions;

  const chooser = $("decision");
  const asks = $("asks");
  const fields = $("fields");
  const answer = $("answer");
  const bytes = $("bytes");
  let controls = new Map();
  /// The fields the form was last drawn from, so a value typed into one can be
  /// found again when the form changes shape.
  let drawn = [];

  const chosen = () => decisions[Number(chooser.value)] ?? decisions[0];

  /// Bind to the controls the page was built with, without touching the DOM:
  /// a redraw here would be the layout shift the static markup exists to
  /// avoid. The markup holds the first decision's fields, so a first decision
  /// with a list or a shape per format is drawn rather than adopted.
  function adopt() {
    const decision = chosen();
    if (typeof decision.length !== "number") {
      draw();
      return;
    }
    controls = new Map();
    const found = fields.querySelectorAll("input, select");
    decision.fields.forEach((field, index) => {
      const control = found[index];
      if (control) controls.set(field.name, control);
    });
  }

  /// The form for the decision chosen, keeping any value already typed for a
  /// field of the same name.
  function draw(keep = new Map()) {
    const decision = chosen();
    asks.textContent = decision.asks;
    // Which form to draw is itself a value on the page: the count of a list,
    // or the format of a request whose format chooses its shape. A value that
    // asks for no form the kernel reads still draws the fields it was typed
    // into, so it can be corrected.
    const guess = new Map();
    for (const field of decision.fields) {
      guess.set(field.name, keep.get(field.name) ?? String(field.value));
    }
    let shape;
    try {
      shape = shapeOf(decision, guess);
    } catch {
      // A value no form of this decision has: the fields it was typed into
      // are still drawn, and the one that chooses the form still redraws it.
      const length = decision.length;
      const drives = typeof length === "number" ? undefined : (length.countAt ?? length.at);
      shape = { fields: decision.fields, items: [], drives };
    }

    fields.replaceChildren();
    controls = new Map();
    drawn = [...shape.fields, ...shape.items];
    for (const field of shape.fields) {
      // A byte two formats read differently has a name in each of them, so a
      // value already typed is looked for by its byte as well as its name.
      const had = keep.get(field.name) ?? keep.get(`@${field.at}`) ?? guess.get(field.name);
      const { label, control } = input({ ...field, value: had ?? field.value });
      fields.append(label);
      controls.set(field.name, control);
    }
    for (const group of decision.list?.groups ?? []) {
      const mine = shape.items.filter((item) => item.group === group.reader);
      if (mine.length === 0) continue;
      const box = document.createElement("div");
      box.className = "group";
      const head = document.createElement("span");
      head.className = "k";
      const many = mine.length === 1 ? "1 item" : `${mine.length} items`;
      head.textContent = `${group.name} — ${many} of ${group.width} bytes, from byte ${mine[0].at}`;
      box.append(head);
      const row = document.createElement("div");
      row.className = "items";
      for (const item of mine) {
        const { label, control } = input({ ...item, value: keep.get(item.name) ?? item.value });
        row.append(label);
        controls.set(item.name, control);
      }
      box.append(row);
      fields.append(box);
    }
    // The control that chooses the form redraws it as it changes.
    if (shape.drives !== undefined) {
      const control = controls.get(named(decision, shape.drives));
      if (control) {
        control.addEventListener(control.tagName === "SELECT" ? "change" : "input", again);
      }
    }

    answer.textContent = "—";
    answer.className = "";
    bytes.textContent = "—";
  }

  /// Draw again for a form that has changed shape, keeping what was typed and
  /// the caret where it was.
  function again() {
    const keep = new Map();
    for (const [name, control] of controls) keep.set(name, control.value);
    // A fixed field keeps its byte whatever the form; an item of a list does
    // not, because where a group begins follows the count.
    for (const field of drawn) {
      const control = controls.get(field.name);
      if (control && field.group === undefined) keep.set(`@${field.at}`, control.value);
    }
    const focused = [...controls].find(([, control]) => control === document.activeElement);
    draw(keep);
    const control = focused && controls.get(focused[0]);
    if (control) {
      control.focus();
      try {
        control.setSelectionRange(control.value.length, control.value.length);
      } catch {
        // A select has no caret to put back.
      }
    }
  }

  function written() {
    const values = new Map();
    for (const [name, control] of controls) values.set(name, control.value);
    return request(chosen(), values);
  }

  async function decide() {
    let asked;
    try {
      asked = written();
    } catch (error) {
      answer.textContent = String(error.message ?? error);
      answer.className = "bad";
      return;
    }
    bytes.textContent = Array.from(asked, (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (!app) {
      answer.textContent = reason || "the kernel is not here";
      answer.className = "bad";
      return;
    }
    answer.textContent = "…";
    answer.className = "";
    try {
      const given = await app.dispatch(asked);
      answer.textContent = show(given);
      answer.className = "";
    } catch (error) {
      answer.textContent = String(error.message ?? error);
      answer.className = "bad";
    }
  }

  chooser.addEventListener("change", () => draw());
  $("decide").addEventListener("click", decide);
  $("reset-fields").addEventListener("click", () => draw());
  fields.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      decide();
    }
  });

  adopt();
  const status = $("status");
  if (status && !app) chip(status, reason || "no kernel", "warn");
}
