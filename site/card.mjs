// The app's own surface: pick a decision, fill in its numbers, and the
// verified guest answers.
//
// Every layout here comes from site/decisions.json, which `EN-14` checks
// against the kernel's own examples: a field at the wrong offset or of the
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
  const decisions = await fetch("./decisions.json")
    .then((response) => response.json())
    .then((read) => read.decisions);

  const chooser = $("decision");
  const asks = $("asks");
  const fields = $("fields");
  const answer = $("answer");
  const bytes = $("bytes");
  let controls = new Map();

  for (const [index, decision] of decisions.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${decision.tag}. ${decision.name}`;
    chooser.append(option);
  }

  const chosen = () => decisions[Number(chooser.value)] ?? decisions[0];

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

  draw();
  const status = $("status");
  if (status && !app) chip(status, reason || "no kernel", "warn");
}
