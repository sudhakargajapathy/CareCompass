"""Recover complete top-level JSON objects from a truncated model response.

A response cut off mid-array is unparseable as a whole, but every entry
before the cut is intact — and losing all of them because the last was
clipped has bitten this system twice, in its two most expensive calls:

- The judge (round 9): a flat token ceiling truncated the ranking response,
  the parse failed, and every provider silently fell to the neutral score.
  The salvage walker shipped there first.
- Discovery extraction (the 2026-07-28 review runs): the same shape, one
  stage earlier and worse. A truncated extraction has no closing bracket,
  the regex repair cannot match, and `[]` comes back — so a home pool of
  ZERO looks exactly like "the pages named nobody", the ring expands on a
  bug, and the pool gets rebuilt from cities the user never asked about.
  Two runs of the same search, 11 minutes apart, overlapped on ~1
  physician; whether one response fit 8,000 tokens decided WHO got
  recommended.

One walker, shared: the scorer re-exports it under its original private
name (tests and round-9 history bind to that), and the gatherer calls it
as the last resort after full parse and regex repair both fail — salvage
loses the cut entry, so it must never outrank a successful whole-array
parse.

The walk is a character state machine rather than a regex because braces
appear inside string values ('"note": "a } b {"') — a brace-counting regex
mis-nests on exactly the prose fields review summaries are made of.
"""

import json
from typing import Any, Dict, List


def salvage_json_objects(text: str) -> List[Dict[str, Any]]:
    """Every complete top-level ``{...}`` object in ``text``, in order.

    Tolerant of surrounding prose, markdown fences, and a missing closing
    bracket: anything that is not a balanced top-level object is skipped,
    and an object that individually fails ``json.loads`` is dropped rather
    than aborting the walk.
    """
    objects: List[Dict[str, Any]] = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        parsed = json.loads(text[start:position + 1])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start = None
    return objects
