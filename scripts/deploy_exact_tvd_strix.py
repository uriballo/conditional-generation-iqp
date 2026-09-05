#!/usr/bin/env python3
"""Deploy a clean, content-addressed campaign source bundle to both Strix hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("192.168.1.58", "192.168.1.67")


def source_digest() -> str:
    digest = hashlib.sha256()
    for base in (ROOT / "configs", ROOT / "scripts", ROOT / "src", ROOT / "tests"):
        for path in sorted(value for value in base.rglob("*") if value.is_file()):
            relative = path.relative_to(ROOT)
            digest.update(str(relative).encode())
            digest.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="uribagi")
    parser.add_argument("--host", action="append", choices=HOSTS)
    parser.add_argument(
        "--remote-root", default="/home/uribagi/exact-tvd-ancilla-v1-src"
    )
    args = parser.parse_args()
    digest = source_digest()
    hosts = tuple(args.host) if args.host else HOSTS
    destination = f"{args.remote_root}/{digest}"
    for host in hosts:
        remote = f"{args.user}@{host}"
        subprocess.run(["ssh", remote, "mkdir", "-p", destination], check=True)
        for name in ("configs", "scripts", "src", "tests", "pyproject.toml", "uv.lock"):
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--exclude=._*",
                    str(ROOT / name),
                    f"{remote}:{destination}/",
                ],
                check=True,
            )
    print(
        json.dumps(
            {"source_digest": digest, "remote_path": destination, "hosts": hosts}
        )
    )


if __name__ == "__main__":
    main()
