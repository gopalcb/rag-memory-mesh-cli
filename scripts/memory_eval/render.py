"""Embed a recorded run into a standalone, offline case explorer."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
template = Path(__file__).with_name("report.template.html").read_text(encoding="utf-8")
report = json.loads(args.results.read_text(encoding="utf-8"))
data = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
args.output.write_text(template.replace("__RUN_JSON__", data, 1), encoding="utf-8")
