// A model's chat template, rendered as Hugging Face's transformers renders it:
// Jinja with trim_blocks and lstrip_blocks, its own tojson (Python's json.dumps,
// keys in order, non-ASCII kept), raise_exception and strftime_now. Only what
// the templates of the models this page runs use is followed; any other tag,
// filter, test or method is refused by name, so a template this does not
// follow is never rendered approximately. Checked against Python's jinja2,
// configured as transformers configures it, on the three exports' templates.

const UNDEFINED = Object.freeze({ undefined: true });
class Namespace {
  constructor(values) {
    Object.assign(this, values);
  }
}
export class TemplateError extends Error {}
const refuse = (what) => {
  throw new TemplateError(`template: ${what} is not one this engine follows`);
};

// ─── Python's values as JavaScript holds them ─────────────────────────────
const isMapping = (v) => v !== null && typeof v === "object" && !Array.isArray(v) && v !== UNDEFINED && !(v instanceof Namespace);
const truthy = (v) => {
  if (v === UNDEFINED || v === null || v === false || v === 0 || v === "") return false;
  if (Array.isArray(v)) return v.length > 0;
  if (isMapping(v)) return Object.keys(v).length > 0;
  return true;
};
// Python's json.dumps with ensure_ascii False: ", " and ": " between items,
// or the indented form, keys in their order.
function dumps(value, indent = null, depth = 0) {
  if (value === UNDEFINED) refuse("tojson of an undefined value");
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") {
    if (!Number.isInteger(value)) refuse("tojson of a number with a fraction");
    return String(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  const items = Array.isArray(value) ? value.map((v) => dumps(v, indent, depth + 1)) : Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${dumps(v, indent, depth + 1)}`);
  const [open, close] = Array.isArray(value) ? ["[", "]"] : ["{", "}"];
  if (!items.length) return open + close;
  if (indent === null) return open + items.join(", ") + close;
  const pad = " ".repeat(indent * (depth + 1));
  return `${open}\n${pad}${items.join(`,\n${pad}`)}\n${" ".repeat(indent * depth)}${close}`;
}
// Python's str() of a value written into the output.
const str = (v) => {
  if (v === UNDEFINED) return "";
  if (v === null) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "number") {
    if (!Number.isInteger(v)) refuse("writing a number with a fraction");
    return String(v);
  }
  if (typeof v === "string") return v;
  refuse("writing a list, mapping or namespace as text");
};
// Python's str.strip and its kin: whitespace, or any of the characters given.
const stripper = (s, chars, left, right) => {
  const drop = chars === undefined || chars === null ? (c) => /\s/.test(c) : (c) => chars.includes(c);
  const cs = [...s];
  let a = 0;
  let b = cs.length;
  if (left) while (a < b && drop(cs[a])) a++;
  if (right) while (b > a && drop(cs[b - 1])) b--;
  return cs.slice(a, b).join("");
};
const STRING_METHODS = {
  startswith: (s, p) => s.startsWith(p),
  endswith: (s, p) => s.endsWith(p),
  strip: (s, c) => stripper(s, c, true, true),
  lstrip: (s, c) => stripper(s, c, true, false),
  rstrip: (s, c) => stripper(s, c, false, true),
  split: (s, sep) => (sep === undefined || sep === null ? s.split(/\s+/).filter(Boolean) : s.split(sep)),
};
const index = (seq, i) => {
  const n = typeof seq === "string" ? [...seq].length : seq.length;
  const at = i < 0 ? n + i : i;
  if (at < 0 || at >= n) return UNDEFINED;
  return typeof seq === "string" ? [...seq][at] : seq[at];
};
// Python's slice of a list or a string.
function slice(seq, start, stop, step) {
  const items = typeof seq === "string" ? [...seq] : seq;
  const n = items.length;
  const s = step ?? 1;
  if (s === 0) refuse("a slice step of zero");
  const clamp = (v, low, high) => Math.min(Math.max(v, low), high);
  const norm = (v, dflt) => (v === null || v === undefined ? dflt : v < 0 ? v + n : v);
  const out = [];
  if (s > 0) {
    for (let i = clamp(norm(start, 0), 0, n); i < clamp(norm(stop, n), 0, n); i += s) out.push(items[i]);
  } else {
    const a = start === null || start === undefined ? n - 1 : clamp(norm(start, 0), -1, n - 1);
    const b = stop === null || stop === undefined ? -1 : clamp(norm(stop, 0), -1, n - 1);
    for (let i = a; i > b; i += s) out.push(items[i]);
  }
  return typeof seq === "string" ? out.join("") : out;
}
const equal = (a, b) => {
  if (a === UNDEFINED || b === UNDEFINED) return a === b;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => equal(v, b[i]));
  if (isMapping(a) && isMapping(b)) return Object.keys(a).length === Object.keys(b).length && Object.keys(a).every((k) => k in b && equal(a[k], b[k]));
  return a === b;
};
const contains = (container, item) => {
  if (typeof container === "string") return typeof item === "string" && container.includes(item);
  if (Array.isArray(container)) return container.some((v) => equal(v, item));
  if (isMapping(container)) return Object.prototype.hasOwnProperty.call(container, item);
  refuse("`in` on a value that is not a string, list or mapping");
};

const FILTERS = {
  length: (v) => (typeof v === "string" ? [...v].length : Array.isArray(v) ? v.length : isMapping(v) ? Object.keys(v).length : refuse("length of this value")),
  trim: (v) => stripper(str(v), null, true, true),
  tojson: (v, kw) => dumps(v, kw.indent ?? null),
};
const TESTS = {
  defined: (v) => v !== UNDEFINED,
  none: (v) => v === null,
  string: (v) => typeof v === "string",
  false: (v) => v === false,
  true: (v) => v === true,
  mapping: (v) => isMapping(v),
  iterable: (v) => typeof v === "string" || Array.isArray(v) || isMapping(v),
  number: (v) => typeof v === "number",
  sequence: (v) => typeof v === "string" || Array.isArray(v),
};

// ─── Reading the template ─────────────────────────────────────────────────
// Text and tags, with whitespace control: a "-" beside the tag strips the
// whitespace on that side; trim_blocks drops the newline after a block or
// comment tag; lstrip_blocks drops the spaces and tabs before one that begins
// its line.
function segments(source) {
  const out = [];
  let at = 0;
  const open = /\{([{%#])(-|\+)?/g;
  while (at < source.length) {
    open.lastIndex = at;
    const m = open.exec(source);
    if (!m) {
      out.push({ text: source.slice(at) });
      break;
    }
    let text = source.slice(at, m.index);
    const kind = m[1];
    const close = { "{": "}}", "%": "%}", "#": "#}" }[kind];
    // The tag ends at its closing pair outside any quoted string; a comment
    // holds no strings.
    let end = -1;
    let quote = null;
    for (let k = m.index + 2 + (m[2] ? 1 : 0); k < source.length - 1; k++) {
      const ch = source[k];
      if (quote) {
        if (ch === "\\") k++;
        else if (ch === quote) quote = null;
      } else if (kind !== "#" && (ch === "'" || ch === '"')) quote = ch;
      else if (source.startsWith(close, k)) {
        end = k;
        break;
      }
    }
    if (end < 0) refuse(`an unclosed ${"{" + kind}`);
    const rightStrip = source[end - 1] === "-";
    const inner = source.slice(m.index + 2 + (m[2] ? 1 : 0), rightStrip ? end - 1 : end);
    if (m[2] === "-") text = text.replace(/\s+$/, "");
    else if (kind !== "{" && m[2] !== "+") {
      const lineStart = text.lastIndexOf("\n") + 1;
      if (/^[ \t]*$/.test(text.slice(lineStart))) text = text.slice(0, lineStart);
    }
    if (text) out.push({ text });
    if (kind === "{") out.push({ expr: inner });
    else if (kind === "%") out.push({ stmt: inner.trim() });
    at = end + 2;
    if (rightStrip) {
      while (at < source.length && /\s/.test(source[at])) at++;
    } else if (kind !== "{" && source[at] === "\n") at++;
    else if (kind !== "{" && source.startsWith("\r\n", at)) at += 2;
  }
  return out;
}

// Expression tokens.
function tokens(src) {
  const out = [];
  const re = /\s*(?:(\d+)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|([A-Za-z_]\w*)|(==|!=|<=|>=|\/\/|\*\*|[-+*/%<>()[\]{}.,:|=~]))/y;
  let at = 0;
  while (at < src.length) {
    if (/^\s*$/.test(src.slice(at))) break;
    re.lastIndex = at;
    const m = re.exec(src);
    if (!m) refuse(`the expression text "${src.slice(at, at + 20)}"`);
    if (m[1] !== undefined) out.push({ num: Number(m[1]) });
    else if (m[2] !== undefined) out.push({ str: unquote(m[2]) });
    else if (m[3] !== undefined) out.push({ name: m[3] });
    else out.push({ op: m[4] });
    at = re.lastIndex;
  }
  return out;
}
function unquote(literal) {
  const body = literal.slice(1, -1);
  const escapes = { n: "\n", t: "\t", r: "\r", "\\": "\\", "'": "'", '"': '"', "0": "\0" };
  return body.replace(/\\(u[0-9a-fA-F]{4}|.)/g, (_, e) => (e.length === 5 ? String.fromCharCode(parseInt(e.slice(1), 16)) : escapes[e] ?? `\\${e}`));
}

// An expression parser, in Jinja's precedence: or, and, not, comparisons and
// in, + and -, unary minus, then a primary with its attribute, item, slice and
// call suffixes and its filters and tests.
function parseExpression(src) {
  const t = tokens(src);
  let i = 0;
  const peek = (k = 0) => t[i + k];
  const isOp = (op, k = 0) => peek(k)?.op === op;
  const isName = (n, k = 0) => peek(k)?.name === n;
  const expect = (op) => {
    if (!isOp(op)) refuse(`expected "${op}" in "${src}"`);
    i++;
  };
  const or = () => {
    let left = and();
    while (isName("or")) {
      i++;
      const l = left;
      const r = and();
      left = (c) => (truthy(l(c)) ? l(c) : r(c));
    }
    return left;
  };
  const and = () => {
    let left = not();
    while (isName("and")) {
      i++;
      const l = left;
      const r = not();
      left = (c) => (truthy(l(c)) ? r(c) : l(c));
    }
    return left;
  };
  const not = () => {
    if (isName("not")) {
      i++;
      const v = not();
      return (c) => !truthy(v(c));
    }
    return compare();
  };
  const compare = () => {
    let left = sum();
    for (;;) {
      const l = left;
      if (["==", "!=", "<", ">", "<=", ">="].some((o) => isOp(o))) {
        const op = t[i++].op;
        const r = sum();
        left = (c) => {
          const a = l(c);
          const b = r(c);
          if (op === "==") return equal(a, b);
          if (op === "!=") return !equal(a, b);
          if (typeof a !== typeof b || (typeof a !== "number" && typeof a !== "string")) refuse(`comparing with ${op} other than numbers or strings`);
          return op === "<" ? a < b : op === ">" ? a > b : op === "<=" ? a <= b : a >= b;
        };
      } else if (isName("in") || (isName("not") && isName("in", 1))) {
        const negated = isName("not");
        i += negated ? 2 : 1;
        const r = sum();
        left = (c) => contains(r(c), l(c)) !== negated;
      } else return left;
    }
  };
  const sum = () => {
    let left = unary();
    while (isOp("+") || isOp("-")) {
      const op = t[i++].op;
      const l = left;
      const r = unary();
      left = (c) => {
        const a = l(c);
        const b = r(c);
        if (op === "-") {
          if (typeof a !== "number" || typeof b !== "number") refuse("- on values that are not numbers");
          return a - b;
        }
        if (typeof a === "string" && typeof b === "string") return a + b;
        if (typeof a === "number" && typeof b === "number") return a + b;
        if (Array.isArray(a) && Array.isArray(b)) return [...a, ...b];
        refuse("+ on values of different kinds");
      };
    }
    return left;
  };
  const unary = () => {
    if (isOp("-")) {
      i++;
      const v = unary();
      return (c) => -v(c);
    }
    return filtered(postfix(primary()));
  };
  // Arguments: positional, then name=value.
  const args = () => {
    const positional = [];
    const named = {};
    expect("(");
    while (!isOp(")")) {
      if (peek()?.name && isOp("=", 1)) {
        const key = t[i].name;
        i += 2;
        named[key] = or();
      } else positional.push(or());
      if (!isOp(")")) expect(",");
    }
    expect(")");
    return { positional, named };
  };
  const primary = () => {
    const tok = t[i++];
    if (!tok) refuse(`the end of "${src}"`);
    if (tok.num !== undefined) return () => tok.num;
    if (tok.str !== undefined) {
      let s = tok.str;
      while (peek()?.str !== undefined) s += t[i++].str;
      return () => s;
    }
    if (tok.op === "(") {
      const v = or();
      expect(")");
      return v;
    }
    if (tok.op === "[") {
      const items = [];
      while (!isOp("]")) {
        items.push(or());
        if (!isOp("]")) expect(",");
      }
      expect("]");
      return (c) => items.map((f) => f(c));
    }
    if (tok.op === "{") {
      const pairs = [];
      while (!isOp("}")) {
        const k = or();
        expect(":");
        pairs.push([k, or()]);
        if (!isOp("}")) expect(",");
      }
      expect("}");
      return (c) => Object.fromEntries(pairs.map(([k, v]) => [k(c), v(c)]));
    }
    if (tok.name !== undefined) {
      const n = tok.name;
      if (n === "true" || n === "True") return () => true;
      if (n === "false" || n === "False") return () => false;
      if (n === "none" || n === "None") return () => null;
      return (c) => c.lookup(n);
    }
    refuse(`"${tok.op}" at the start of an expression`);
  };
  const postfix = (base) => {
    let v = base;
    for (;;) {
      const target = v;
      if (isOp(".")) {
        i++;
        const name = t[i++]?.name;
        if (name === undefined) refuse(`an attribute name in "${src}"`);
        v = (c) => attribute(target(c), name);
      } else if (isOp("[")) {
        i++;
        const parts = [null, null, null];
        let isSlice = false;
        let part = 0;
        while (!isOp("]")) {
          if (isOp(":")) {
            isSlice = true;
            part++;
            i++;
            continue;
          }
          parts[part] = or();
        }
        expect("]");
        if (isSlice) {
          v = (c) => {
            const seq = target(c);
            if (typeof seq !== "string" && !Array.isArray(seq)) refuse("a slice of a value that is not a string or list");
            const [a, b, s] = parts.map((p) => (p ? p(c) : null));
            return slice(seq, a, b, s);
          };
        } else {
          const key = parts[0];
          v = (c) => item(target(c), key(c));
        }
      } else if (isOp("(")) {
        const a = args();
        v = (c) => {
          const f = target(c);
          if (typeof f !== "function") refuse("calling a value that is not a function");
          return f(a.positional.map((p) => p(c)), Object.fromEntries(Object.entries(a.named).map(([k, p]) => [k, p(c)])));
        };
      } else return v;
    }
  };
  const filtered = (base) => {
    let v = base;
    for (;;) {
      const target = v;
      if (isOp("|")) {
        i++;
        const name = t[i++]?.name;
        const filter = FILTERS[name];
        if (!filter) refuse(`the filter ${name}`);
        const a = isOp("(") ? args() : { positional: [], named: {} };
        v = (c) => filter(target(c), Object.fromEntries(Object.entries(a.named).map(([k, p]) => [k, p(c)])), a.positional.map((p) => p(c)));
      } else if (isName("is")) {
        i++;
        const negated = isName("not");
        if (negated) i++;
        const name = t[i++]?.name;
        const test = TESTS[name];
        if (!test) refuse(`the test ${name}`);
        v = (c) => test(target(c)) !== negated;
      } else return v;
    }
  };
  const value = or();
  if (i !== t.length) refuse(`"${src}" past where it ends`);
  return value;
}
function attribute(v, name) {
  if (typeof v === "string") {
    const method = STRING_METHODS[name];
    if (!method) refuse(`the string method ${name}`);
    return (positional) => method(v, ...positional);
  }
  if (v instanceof Namespace) return Object.prototype.hasOwnProperty.call(v, name) ? v[name] : UNDEFINED;
  if (isMapping(v)) return Object.prototype.hasOwnProperty.call(v, name) ? v[name] : UNDEFINED;
  return UNDEFINED;
}
function item(v, key) {
  if (typeof key === "number" && (typeof v === "string" || Array.isArray(v))) return index(v, key);
  if (isMapping(v) && typeof key === "string") return Object.prototype.hasOwnProperty.call(v, key) ? v[key] : UNDEFINED;
  return UNDEFINED;
}

// ─── Statements ───────────────────────────────────────────────────────────
function parseBody(parts, start, ends) {
  const body = [];
  let at = start;
  while (at < parts.length) {
    const p = parts[at];
    if (p.text !== undefined) {
      body.push({ text: p.text });
      at++;
      continue;
    }
    if (p.expr !== undefined) {
      body.push({ out: parseExpression(p.expr) });
      at++;
      continue;
    }
    const word = p.stmt.split(/\s+/)[0];
    if (ends.includes(word)) return { body, at, end: p.stmt };
    if (word === "if") {
      const branches = [{ test: parseExpression(p.stmt.slice(2)) }];
      let r = parseBody(parts, at + 1, ["elif", "else", "endif"]);
      branches[0].body = r.body;
      while (r.end.split(/\s+/)[0] !== "endif") {
        const w = r.end.split(/\s+/)[0];
        const branch = { test: w === "elif" ? parseExpression(r.end.slice(4)) : null };
        r = parseBody(parts, r.at + 1, w === "else" ? ["endif"] : ["elif", "else", "endif"]);
        branch.body = r.body;
        branches.push(branch);
      }
      body.push({ branches });
      at = r.at + 1;
    } else if (word === "for") {
      const m = /^for\s+(\w+)\s+in\s+(.+)$/s.exec(p.stmt);
      if (!m || /\s(if|recursive)\s/.test(` ${m[2]} `)) refuse(`{% ${p.stmt} %}`);
      const r = parseBody(parts, at + 1, ["endfor", "else"]);
      if (r.end !== "endfor") refuse("{% else %} inside a for");
      body.push({ loopVar: m[1], over: parseExpression(m[2]), body: r.body });
      at = r.at + 1;
    } else if (word === "set") {
      const m = /^set\s+(\w+)(?:\.(\w+))?\s*=\s*(.+)$/s.exec(p.stmt);
      if (!m) refuse(`{% ${p.stmt} %}`);
      body.push({ set: m[1], attr: m[2], value: parseExpression(m[3]) });
      at++;
    } else refuse(`{% ${word} %}`);
  }
  if (ends.length) refuse(`a template that ends before {% ${ends.join(" or ")} %}`);
  return { body, at, end: null };
}

class Scope {
  constructor(parent, values = {}) {
    this.parent = parent;
    this.values = values;
  }
  lookup(name) {
    for (let s = this; s; s = s.parent) if (Object.prototype.hasOwnProperty.call(s.values, name)) return s.values[name];
    return UNDEFINED;
  }
}
function run(nodes, scope, out) {
  for (const n of nodes) {
    if (n.text !== undefined) out.push(n.text);
    else if (n.out) out.push(str(n.out(scope)));
    else if (n.branches) {
      const taken = n.branches.find((b) => b.test === null || truthy(b.test(scope)));
      if (taken) run(taken.body, scope, out);
    } else if (n.loopVar) {
      const over = n.over(scope);
      const items = over === UNDEFINED || over === null ? [] : typeof over === "string" ? [...over] : Array.isArray(over) ? over : isMapping(over) ? Object.keys(over) : refuse("a for over a value that is not iterable");
      items.forEach((v, k) => {
        const loop = { index0: k, index: k + 1, first: k === 0, last: k === items.length - 1, length: items.length, revindex: items.length - k, revindex0: items.length - k - 1 };
        run(n.body, new Scope(scope, { [n.loopVar]: v, loop }), out);
      });
    } else if (n.set) {
      const value = n.value(scope);
      if (n.attr) {
        const target = scope.lookup(n.set);
        if (!(target instanceof Namespace)) refuse(`setting ${n.set}.${n.attr} on a value that is not a namespace`);
        target[n.attr] = value;
      } else scope.values[n.set] = value;
    }
  }
}

// Compile a chat template once; render(context) answers the text.
export function compileTemplate(source) {
  // Jinja drops one newline at the very end of a template
  // (keep_trailing_newline is off), as transformers leaves it.
  const parts = segments(source.replace(/\r?\n$/, ""));
  const { body } = parseBody(parts, 0, []);
  return (context) => {
    const globals = {
      namespace: (_, named) => new Namespace(named),
      raise_exception: ([message]) => {
        throw new TemplateError(String(message));
      },
      ...context,
    };
    if (typeof context.strftime_now === "function") globals.strftime_now = ([format]) => context.strftime_now(format);
    const out = [];
    run(body, new Scope(null, globals), out);
    return out.join("");
  };
}

// strftime for the formats chat templates ask for: %d %b %B %Y %m %y.
export function strftime(date, format) {
  const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const two = (n) => String(n).padStart(2, "0");
  return format.replace(/%([dbBYmy%])/g, (_, f) => ({ d: two(date.getDate()), b: months[date.getMonth()].slice(0, 3), B: months[date.getMonth()], Y: String(date.getFullYear()), m: two(date.getMonth() + 1), y: two(date.getFullYear() % 100), "%": "%" })[f]);
}
