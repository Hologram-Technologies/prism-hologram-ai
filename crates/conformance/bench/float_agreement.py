"""Greedy agreement of the integer engine with the float model (rows 2, 3).

The float model is the same ONNX file the integer engine read, run by ONNX
Runtime on the CPU with graph optimisation off, so what runs is the graph as
the export wrote it, half-precision activations included. It is a comparison
only: nothing here ships.

Two readings per prompt, over the reference's 256 tokens:
  teacher-forced: the float model's argmax at each position, given the integer
    engine's tokens before it; how many of the 256 agree, and the float model's
    margin between its best two logits where they do not;
  free: the float model's own greedy run from the prompt; the first index
    where it parts from the integer engine's.
Both are also read up to the integer engine's end of turn.

    python3 -P float_agreement.py <reference.json> <model cache> <out.json>
"""

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def session_for(reference, cache):
    graphs = [f for f in reference["modelFiles"] if f.endswith(".onnx")]
    if len(graphs) != 1:
        raise SystemExit(f"expected one graph among {reference['modelFiles']}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = os.cpu_count()
    path = os.path.join(cache, graphs[0])
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"]), graphs[0]


class Float:
    def __init__(self, session):
        self.s = session
        self.inputs = {i.name: i for i in session.get_inputs()}
        self.past = [n for n in self.inputs if n.startswith("past_key_values.")]
        self.out_names = [o.name for o in session.get_outputs()]

    def empty_past(self):
        feeds = {}
        for name in self.past:
            i = self.inputs[name]
            dtype = np.float16 if "float16" in i.type else np.float32
            # [batch, kv heads, past length, head size]: one batch, no past.
            shape = [d if isinstance(d, int) else (1 if k == 0 else 0) for k, d in enumerate(i.shape)]
            feeds[name] = np.zeros(shape, dtype=dtype)
        return feeds

    def forward(self, ids, start, past):
        n = len(ids)
        feeds = {"input_ids": np.array([ids], dtype=np.int64)}
        if "attention_mask" in self.inputs:
            feeds["attention_mask"] = np.ones((1, start + n), dtype=np.int64)
        if "position_ids" in self.inputs:
            feeds["position_ids"] = np.arange(start, start + n, dtype=np.int64)[None, :]
        feeds.update(past)
        outs = dict(zip(self.out_names, self.s.run(None, feeds)))
        present = {}
        for name in self.past:
            present[name] = outs[name.replace("past_key_values", "present")]
        return outs["logits"][0].astype(np.float32), present


def agreement(model, run, tokens):
    prompt, ids = run["promptIds"], run["ids"][:tokens]
    logits, _ = model.forward(prompt + ids[:-1], 0, model.empty_past())
    picks = logits[len(prompt) - 1:].argmax(axis=1)
    forced = [int(p) == int(t) for p, t in zip(picks, ids)]
    misses = []
    for k, ok in enumerate(forced):
        if not ok:
            row = logits[len(prompt) - 1 + k]
            top = np.sort(row)[-2:]
            misses.append({"at": k, "float": int(picks[k]), "integer": ids[k], "margin": float(top[1] - top[0]),
                           "integer_gap": float(row[picks[k]] - row[ids[k]])})
    # The float model's own greedy run, one position at a time from the prompt.
    free = []
    out, past = model.forward(prompt, 0, model.empty_past())
    nxt = int(out[-1].argmax())
    at = len(prompt)
    while len(free) < len(ids):
        free.append(nxt)
        if len(free) == len(ids):
            break
        out, past = model.forward([nxt], at, past)
        at += 1
        nxt = int(out[-1].argmax())
    parted = next((k for k, (a, b) in enumerate(zip(free, ids)) if a != b), None)
    end = run.get("endOfTurnAt")
    upto = len(ids) if end is None else end + 1
    return {
        "prompt": run["prompt"],
        "tokens": len(ids),
        "forced_equal": sum(forced),
        "forced_equal_to_end_of_turn": sum(forced[:upto]),
        "end_of_turn_at": end,
        "free_parts_at": parted,
        "misses": misses,
    }


def main():
    reference_path, cache, out = sys.argv[1:4]
    with open(reference_path) as f:
        reference = json.load(f)
    session, graph = session_for(reference, cache)
    model = Float(session)
    started = time.time()
    rows = [agreement(model, run, reference["tokens"]) for run in reference["runs"]]
    result = {
        "key": reference["key"],
        "model": reference["model"],
        "graph": graph,
        "float": f"onnxruntime {ort.__version__}, CPUExecutionProvider, graph optimisation off",
        "seconds": round(time.time() - started, 1),
        "prompts": rows,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "prompts"} | {
        "forced": [f"{r['forced_equal']}/{r['tokens']}" for r in rows],
        "free_parts_at": [r["free_parts_at"] for r in rows]}))


if __name__ == "__main__":
    main()
