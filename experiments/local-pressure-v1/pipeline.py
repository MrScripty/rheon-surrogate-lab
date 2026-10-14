#!/usr/bin/env python3
"""Explicit local CPU preparation and execution; every output stage is exclusive."""
import argparse
import json
from pathlib import Path
import sys

from contract import ContractError, decode, experiment, read, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("capture-build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--prepared-source", type=Path, required=True)
    build.add_argument("--cargo", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("capture")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--binary", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--fixture", action="store_true")
    generate.add_argument("--fixture-capture", type=Path)
    verify = sub.add_parser("verify-capture")
    verify.add_argument("--capture", type=Path, required=True)
    verify.add_argument("--candidate", action="store_true")
    pack = sub.add_parser("pack")
    pack.add_argument("--capture", type=Path, required=True)
    pack.add_argument("--native-binary", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import capture
        if args.command == "capture-build":
            config = experiment(decode(read(args.config)))
            binary = capture.build(config, args.prepared_source.resolve(), args.cargo.absolute(), args.output)
            result = {"status": "qualified_build", "binary": str(binary)}
        elif args.command == "capture":
            config = experiment(decode(read(args.config)))
            if not args.fixture:
                require(args.fixture_capture is not None, "fixture_required", "qualify a one-frame train fixture before full capture")
                fixture = capture.verify(args.fixture_capture.absolute())
                require(fixture["status"] == "verified_fixture" and fixture["frames"] == 1
                        and fixture["experiment_sha256"] == __import__("contract").sha(__import__("contract").canonical(config)), "fixture_identity", "same configured experiment")
            result = capture.generate(config, args.binary.absolute(), args.output, args.fixture)
        elif args.command == "pack":
            import data
            result = data.pack(args.capture.absolute(), args.native_binary.absolute(), args.output)
        else:
            result = capture.verify(args.capture.absolute(), args.candidate)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
