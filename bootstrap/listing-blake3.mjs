#!/usr/bin/env node
// Adds each served file's BLAKE3 to site/precache.json, the listing the
// frame's precache.mjs writes. BLAKE3 is the address Hologram AI names its
// own files by; the frame is the template's and pinned by frame.lock, so its
// listing keeps its own SHA-256 and version, and the page compares by the
// blake3 written here. Run right after precache.mjs.
//   node bootstrap/listing-blake3.mjs site
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { blake3Hex } from "../try/vendor/prism-code/src/blake3.mjs";

const site = process.argv[2] ?? "site";
const path = join(site, "precache.json");
const listing = JSON.parse(await readFile(path, "utf8"));
for (const file of listing.files) file.blake3 = blake3Hex(await readFile(join(site, file.path)));
await writeFile(path, JSON.stringify(listing, null, 2) + "\n");
console.log(`listing-blake3: ${listing.files.length} files named by BLAKE3`);
