"""Post-merge extractor — run by CI on merge to main.

Walks the just-merged project and appends extracted entries to
`knowledge/`. Intended to be invoked with the project name as a CLI arg.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from lib.extract_knowledge import extract_from_project

_log = logging.getLogger("eda.post_merge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-merge knowledge extractor.")
    parser.add_argument("project_name", help="Project that was just merged to main.")
    parser.add_argument("--workspace", default=None, help="Workspace root (default: cwd or marker).")
    parser.add_argument("--min-info-gain", type=float, default=0.1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    res = extract_from_project(
        args.project_name,
        workspace=args.workspace,
        min_info_gain=args.min_info_gain,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
