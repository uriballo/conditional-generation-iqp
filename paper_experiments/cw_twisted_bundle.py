"""Bundle completed twisted-parity Curie--Weiss finals for paper reporting.

Each campaign root must hold a completed final stage from ``cw_encoder_hpo``
(``--campaign twisted`` or ``twisted_independent``). The bundle stores the
per-repeat curves of every family in the layout of the follow-up bundle,
together with the protocol and campaign receipts. ``--optional ROOT``
names a campaign that is skipped, and recorded as omitted, while its final
stage has not been written yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cw_followup_fit import write_json

KEYS = ("trial_id", "config", "beta", "curves", "marker_beta", "marker_curves")
FAMILIES = ("shared_twisted_parity", "independent_twisted_parity")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _final(root, families=FAMILIES):
    root = Path(root)
    status = json.loads((root / "final" / "review-ready.json").read_text())
    path = root / "final" / "figure_data.json"
    if status.get("status") != "final_complete" or status.get(
        "figure_data_sha256"
    ) != _sha(path):
        raise ValueError(f"incomplete or modified final stage: {root}")
    data = json.loads(path.read_text())
    if data.get("family") not in families or len(data["repeat_scores"]) != 10:
        raise ValueError(f"unexpected final data: {root}")
    row = {key: data[key] for key in KEYS}
    row["repeat_scores"] = data["repeat_scores"]
    receipt = dict(
        protocol_sha256=data["provenance"]["protocol"]["protocol_sha256"],
        campaign_id=data["provenance"]["protocol"]["campaign_id"],
        figure_data_sha256=status["figure_data_sha256"],
        campaign_sha256=_sha(root / "campaign.json"),
        final_campaign_sha256=_sha(root / "final" / "campaign.json"),
    )
    return data["family"], row, receipt


def build(roots, destination, optional=()):
    families, provenance, omitted = {}, {}, []
    present = list(roots)
    for root in optional:
        if (Path(root) / "final" / "review-ready.json").exists():
            present.append(root)
        else:
            omitted.append(str(root))
    if not present:
        raise ValueError("no completed campaign to bundle")
    for root in present:
        name, row, receipt = _final(root)
        if name in families:
            raise ValueError(f"duplicate family: {name}")
        families[name], provenance[name] = row, receipt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(
        kind="cw_twisted_final",
        stage="final",
        smoke=False,
        repeats=10,
        starts=3,
        families=families,
        provenance=provenance,
    )
    if omitted:
        payload["omitted_roots"] = omitted
    write_json(destination / "figure_data.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument(
        "--optional",
        action="append",
        default=[],
        type=Path,
        help="campaign root skipped while its final stage is not written",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build(args.roots, args.output, args.optional)
    print(json.dumps(sorted(payload["families"]), indent=1))
    for root in payload.get("omitted_roots", ()):
        print(f"omitted (final stage not written): {root}")


if __name__ == "__main__":
    main()
