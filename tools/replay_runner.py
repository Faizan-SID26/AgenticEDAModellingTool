"""Drive `lib.replay.replay_project` from the command line."""
from __future__ import annotations

import argparse
import json
import sys

from lib.replay import replay_project


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("project_name")
    p.add_argument("--workspace", default=None)
    p.add_argument("--up-to-iteration", type=int, default=None)
    args = p.parse_args(argv)
    out = replay_project(args.project_name, workspace=args.workspace, up_to_iteration=args.up_to_iteration)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
