"""Load a training configuration and run the training entry point."""

import argparse
import json
from pathlib import Path
import runpy
import sys


def profile_arguments(path):
    with Path(path).open(encoding="utf-8") as handle:
        profile = json.load(handle)
    result = []
    for key, value in profile["args"].items():
        if value is None or value is False:
            continue
        result.append(f"--{key}")
        if value is not True:
            result.append(str(value))
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", type=Path)
    known, remaining = parser.parse_known_args()
    defaults = profile_arguments(known.profile) if known.profile else []
    sys.argv = [sys.argv[0], *defaults, *remaining]
    runpy.run_module("EvoRibo.train_direct_trunk_ddp", run_name="__main__")


if __name__ == "__main__":
    main()
