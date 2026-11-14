// The app's own surface: pick a decision, fill in its numbers, and the
// verified guest answers.
//
// Every layout comes from the kernel itself: `cargo xtask page --write` reads
// the offsets, widths, lengths and tags out of `app/src/Kernel` and writes
// them into this page and into site/decisions.json, which `EN-14` checks
// against the kernel's own examples. A field at the wrong offset or of the
// wrong width fails the gate rather than quietly asking the kernel something
// else. Nothing in this file decides anything; it writes the request the
// kernel reads and shows the answer the kernel gives.

const BYTE = 256n;

/**
 * The request bytes for `decision`, from the values on the page.
 *
 * Exported so `EN-14` can run it over the kernel's own examples: the page's
 * encoder is the thing being checked, not a copy of it.
 */
export function request(decision, values) {
  const out = new Uint8Array(decision.length);
  out[0] = decision.tag;
  for (const field of decision.fields) {
    const value = values.get(field.name);
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
  name.textContent = field.name;
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

  const chosen = () => decisions[Number(chooser.value)] ?? decisions[0];

  /// Bind to the controls the page was built with, without touching the DOM:
  /// a redraw here would be the layout shift the static markup exists to
  /// avoid.
  function adopt() {
    const decision = chosen();
    controls = new Map();
    const found = fields.querySelectorAll("input, select");
    decision.fields.forEach((field, index) => {
      const control = found[index];
      if (control) controls.set(field.name, control);
    });
  }

  function draw() {
    const decision = chosen();
    asks.textContent = decision.asks;
    fields.replaceChildren();
    controls = new Map();
    for (const field of decision.fields) {
      const { label, control } = input(field);
      fields.append(label);
      controls.set(field.name, control);
    }
    answer.textContent = "—";
    answer.className = "";
    bytes.textContent = "—";
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

  chooser.addEventListener("change", draw);
  $("decide").addEventListener("click", decide);
  $("reset-fields").addEventListener("click", draw);
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
