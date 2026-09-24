"""Build a searchable HTS index from the USITC raw JSON export.

Source: https://hts.usitc.gov/reststop/exportList?from=0100&to=9999&format=JSON&styles=false
The raw export is a flat list of rows with an `indent` level that encodes the
hierarchy. Rows with an empty `htsno` are continuation text belonging to the
previous code.

Critical detail discovered by inspection: statistical-leaf descriptions are often
generic ("Other", "Of a thickness of 0.4 mm or more") and only make sense with
their ancestors. So every entry stores `full_path` -- the chain of real ancestor
descriptions -- and `search_text` is built from path + own description. Without
this, lexical recall returns semantically empty candidates.

Output: data/hts_index.json
  [{code, description, chapter, heading, level, path_text, full_path, search_text}]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "hts_raw.json"
OUT = DATA / "hts_index.json"


def _indent(row: dict) -> int:
    try:
        return int(str(row.get("indent", "0")).strip())
    except ValueError:
        return 0


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.rstrip(":").strip()


def build() -> list[dict]:
    rows = json.loads(RAW.read_text())
    entries: list[dict] = []
    # Most recent description at each indent level -> the ancestor chain.
    ancestors: dict[int, str] = {}
    last: dict | None = None

    for row in rows:
        code = (row.get("htsno") or "").strip()
        desc = _clean(row.get("description", ""))
        indent = _indent(row)

        if not code:
            # Continuation text belongs to the previous code; fold it in so the
            # full legal text stays searchable.
            if last is not None and desc and desc not in last["description"]:
                last["description"] = (last["description"] + " " + desc).strip()
            continue

        digits = code.replace(".", "")
        level = {2: "chapter", 4: "heading", 6: "subheading", 8: "statistical",
                 10: "statistical"}.get(len(digits), "other")
        chapter = digits[:2]

        if desc:
            ancestors[indent] = desc
        # Drop any stale deeper entries when we move back up the tree.
        for k in [k for k in ancestors if k > indent]:
            del ancestors[k]

        parent_chain = [ancestors[i] for i in sorted(ancestors) if i < indent]
        path_text = " > ".join(parent_chain[-3:]) if parent_chain else ""
        # Full chain, used for search relevance -- this is what rescues generic
        # leaf descriptions like "Other" or "Of a thickness of 0.4 mm or more".
        full_path = " > ".join(parent_chain)

        entry = {
            "code": code,
            "description": desc,
            "chapter": chapter,
            "heading": digits[:4],
            "level": level,
            "path_text": path_text,
            "full_path": full_path,
            "search_text": " ".join(x for x in (desc, full_path) if x),
        }
        entries.append(entry)
        last = entry

    return entries


def main() -> None:
    entries = build()
    OUT.write_text(json.dumps(entries, indent=0))
    stats: dict[str, int] = {}
    for e in entries:
        stats[e["level"]] = stats.get(e["level"], 0) + 1
    print(f"wrote {OUT} : {len(entries)} rows")
    print("levels:", stats)
    for target in ("7318.12.00.00", "8471.30.01.00"):
        for e in entries:
            if e["code"] == target:
                print(f"\n{target}")
                print("  desc      :", e["description"][:70])
                print("  full_path :", e["full_path"][:150])


if __name__ == "__main__":
    main()
