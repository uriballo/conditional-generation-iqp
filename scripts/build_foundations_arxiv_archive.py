"""Build a deterministic, self-contained internal arXiv source archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/arxiv/conditional_generation_classically_trainable_iqp.tar.gz"
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("results/arxiv-foundations-v1/release/source-archive.json"),
    )
    return parser


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_path() -> Path:
    result = subprocess.run(
        ["kpsewhich", "quantumarticle.cls"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = Path(result)
    if not path.is_file():
        raise RuntimeError("quantumarticle.cls is unavailable")
    return path


def _tar_bytes(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(root))
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, filename="") as stream:
        stream.write(buffer.getvalue())
    return compressed.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paper = args.paper.resolve()
    with tempfile.TemporaryDirectory(prefix="arxiv-foundations-") as directory:
        root = Path(directory)
        sources = (
            paper / "main.tex",
            paper / "supplement.tex",
            paper / "references.bib",
            paper / "foundations_figures" / "gaussian_evidence.pdf",
            paper / "foundations_figures" / "ising_representability.pdf",
            _class_path(),
        )
        for source in sources:
            destination = (
                root / "foundations_figures" / source.name
                if source.parent.name == "foundations_figures"
                else root / source.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        status = root / "STATUS.txt"
        status.write_text(
            "INTERNAL DRAFT — NOT UPLOAD-READY\n"
            "The preregistered Ising representability gate failed.\n"
            "Author, affiliation, contribution, competing-interest, category, "
            "license, and human-proof metadata remain unverified.\n"
        )
        checksums = root / "checksums.sha256"
        checksums.write_text(
            "".join(
                f"{_sha(path)}  {path.relative_to(root).as_posix()}\n"
                for path in sorted(item for item in root.rglob("*") if item.is_file())
                if path != checksums
            )
        )
        payload = _tar_bytes(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise RuntimeError("refusing to rewrite a different arXiv source archive")
    else:
        args.output.write_bytes(payload)
    archive_sha = _sha(args.output)
    evidence = {
        "schema_version": 1,
        "archive_path": str(args.output),
        "archive_sha256": archive_sha,
        "upload_ready": False,
        "blocking_empirical_gate": "representability",
        "metadata_complete": False,
        "deterministic_tar_metadata": True,
    }
    if args.evidence.exists():
        if read_immutable_json(args.evidence) != evidence:
            raise RuntimeError("source-archive evidence disagrees with the archive")
    else:
        write_immutable_json(args.evidence, evidence)
    print(f"{archive_sha}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
