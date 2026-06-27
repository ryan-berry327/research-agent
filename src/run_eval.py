"""Run the eval harness and exit non-zero if the gate fails (used by CI)."""
from __future__ import annotations

import json
import sys

from .assembly import build_agent
from .eval.harness import run_eval, gate


def main() -> int:
    report = run_eval(build_agent())
    print(report.summary())
    print(json.dumps(report.cases, indent=2))
    passed = gate(report)
    print("GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
