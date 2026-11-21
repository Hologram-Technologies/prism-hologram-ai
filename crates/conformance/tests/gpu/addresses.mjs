// GP-61: the page reads every wasm32 address unsigned. A module whose memory
// is past 2 GiB hands the page addresses at or above 2^31, which reach
// JavaScript as negative numbers; read signed, a slice at one takes other
// bytes. This gives `moduleOf` a stand-in instance whose answer lies at
// 0xFFFFFFFC and whose module asks for a product at 0x80000020 and
// 0xFFFFFFF0, and prints what the page read. Prints one JSON line.
//
//   node crates/conformance/tests/gpu/addresses.mjs <repository root>

import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.argv[2];
const { moduleOf } = await import(pathToFileURL(join(root, "site/app/engine/loader.mjs")).href);

// The stand-in: a small memory whose first bytes are marked, an answer said
// to lie at 0xFFFFFFFC (as the module's i32 export returns it, -4),
// and a product the module asks for at addresses past 2 GiB.
const memory = { buffer: new Uint8Array([7, 7, 7, 7, 7, 7, 7, 7]).buffer };
let imports = null;
const Real = WebAssembly.Instance;
WebAssembly.Instance = function (compiled, given) {
  imports = given;
  return {
    exports: {
      memory,
      result_at: () => 0xfffffffc | 0,
      result_len: () => 12,
      allocate: () => 0x80000000 | 0,
      release: () => 0,
      answer: () => 0,
    },
  };
};
let asked = null;
try {
  const m = moduleOf({});
  // Read signed, the answer is a slice from -4 to 8, which counts from the
  // end and gives four marked bytes; read unsigned, it lies past this
  // memory's end and gives nothing.
  const answer = Array.from(m.call("answer"));
  m.setProducts((number, at, len, out, cap) => {
    asked = { number, at, len, out, cap };
    return 0;
  });
  imports.kernel.product(3, 0x80000020 | 0, 16, 0xfffffff0 | 0, 8);
  console.log(JSON.stringify({ answer, asked }));
} finally {
  WebAssembly.Instance = Real;
}
