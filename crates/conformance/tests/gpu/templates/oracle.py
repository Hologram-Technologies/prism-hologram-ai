"""Render the three chat templates as transformers renders them, as fixtures.

transformers' chat template environment (src/transformers/utils/chat_template_utils.py):
ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
extensions=[loopcontrols]), with tojson as json.dumps(ensure_ascii=False,
indent, separators, sort_keys=False), raise_exception raising TemplateError,
and strftime_now(format) the current date. Here strftime_now gives a fixed date.
"""
import json
import sys
from datetime import datetime

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


FIXED = datetime(2026, 11, 17)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda fmt: FIXED.strftime(fmt)

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city, now.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name"}}, "required": ["city"]}}}
CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}

CONVERSATIONS = {
    "user": [{"role": "user", "content": "hi"}],
    "system": [{"role": "system", "content": "Be brief.  "}, {"role": "user", "content": "What is the capital of France?"}],
    "turns": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "And Spain?"}],
    "unicode": [{"role": "user", "content": "Café, 中国, 🙂 — \"quotes\" and\nnew lines\t tabs"}],
    "thinking": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "<think>\nadd them\n</think>\n\n4"}, {"role": "user", "content": "3+3?"}, {"role": "assistant", "content": "<think>\nagain\n</think>\n\n6"}],
    "tools": [{"role": "user", "content": "Weather in Paris?"}, {"role": "assistant", "content": "", "tool_calls": [CALL]}, {"role": "tool", "content": "{\"temp\": 18}"}, {"role": "tool", "content": "{\"sky\": \"clear\"}"}],
    "system_later": [{"role": "user", "content": "hi"}, {"role": "system", "content": "Speak French."}, {"role": "user", "content": "again"}],
}

CASES = []
for name, messages in CONVERSATIONS.items():
    for add in (True, False):
        CASES.append({"name": f"{name}{'' if add else ' (no generation prompt)'}", "context": {"messages": messages, "add_generation_prompt": add}})
CASES.append({"name": "tools defined", "context": {"messages": CONVERSATIONS["user"], "tools": [TOOL], "add_generation_prompt": True}})
CASES.append({"name": "tools defined with a system turn", "context": {"messages": CONVERSATIONS["system"], "tools": [TOOL], "add_generation_prompt": True}})
CASES.append({"name": "thinking off", "context": {"messages": CONVERSATIONS["user"], "add_generation_prompt": True, "enable_thinking": False}})
CASES.append({"name": "a date given", "context": {"messages": CONVERSATIONS["user"], "add_generation_prompt": True, "date_string": "01 Jan 2025"}})
CASES.append({"name": "tools with no user turn", "context": {"messages": [{"role": "system", "content": "Be brief."}], "tools": [TOOL], "add_generation_prompt": True}})
CASES.append({"name": "tools outside the first user turn", "context": {"messages": CONVERSATIONS["system"], "tools": [TOOL], "tools_in_user_message": False, "add_generation_prompt": True}})

for template_name in sys.argv[1:]:
    tc = json.load(open(f"{template_name}.tokenizer_config.json"))
    source = open(f"{template_name}.jinja").read()
    template = env.from_string(source)
    specials = {k: tc.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token") if isinstance(tc.get(k), str)}
    out = []
    for case in CASES:
        context = {**specials, **case["context"]}
        try:
            out.append({"name": case["name"], "context": context, "text": template.render(**context)})
        except jinja2.exceptions.TemplateError as e:
            out.append({"name": case["name"], "context": context, "error": str(e)})
    json.dump({"model": template_name, "date": "17 Nov 2026", "template": source, "cases": out}, open(f"{template_name}.fixture.json", "w"), ensure_ascii=False, indent=1)
    print(template_name, len(out), "cases,", sum(1 for c in out if "error" in c), "raise")
