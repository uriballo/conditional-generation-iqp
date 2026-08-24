from pathlib import Path

ROOT = Path(__file__).parents[1]


def _curie_section(text: str) -> str:
    start = text.index(r"\section{Curie--Weiss experiments}")
    end = text.index(r"\section{Discussion}", start)
    return text[start:end]


def test_active_manuscript_uses_only_the_fixed_r5_v3_campaign():
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplement.tex").read_text()
    section = _curie_section(main)
    assert r"\ifcurieweissancillaresults" in main
    assert "80 production fits" in section
    assert "eight visible sites" in section
    assert "13-qubit" in section
    assert "five ancillas" in section
    assert r"r\in\{0,1,3,5,7,9\}" not in section
    assert "profile and raw bitstring TVD as co-primary metrics" in section
    assert "symmetry-tied" not in main + supplement
    assert "symmetry-aligned" not in main + supplement
    assert "270" not in main + supplement
    assert "n=12" not in section and "n=16" not in section
    assert "curie_weiss_figures" not in main + supplement
    assert "curie_weiss_ancilla_figures" in main
    assert "spin-flip two-branch" in section
    assert "Curie--Weiss v3 campaign" in main
    assert "matched-capacity control" not in section.lower()


def test_curie_claims_are_gated_and_v2_capacity_result_is_supplementary():
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplement.tex").read_text()
    assert "generated/curie_weiss_ancilla_claims.tex" in main
    assert r"\ifcurieweissancillaresults" in main
    assert "80-fit\nCurie--Weiss campaign" in main
    assert "curieweissancillaeffects" in supplement
    assert "generated/capacity_control_claims.tex" in main
    assert r"\ifcapacitycontrolresults" in supplement
    assert "Historical v2 matched-capacity diagnostic" in supplement
    assert "not a matched control for the v3" in supplement
    assert "fig:capacity-control" not in main
    assert "fig:capacity-control" in supplement
    assert "did not test" in supplement and "optimal ancilla count" in supplement


def test_symmetry_screens_are_exploratory_and_supplement_only():
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplement.tex").read_text()
    assert r"\section{Symmetry-matched input states}" not in main
    assert r"\section{Exploratory symmetry screens}" in supplement
    assert "post-core screens" in supplement
    assert "does not isolate symmetry matching" in supplement
    assert "fig:two-block" not in main and "fig:radial" not in main


def test_superseded_manuscript_snapshot_is_content_addressed():
    roots = list(
        (ROOT / "archive" / "curie-weiss-foundations-v1-manuscript-superseded").glob(
            "*"
        )
    )
    assert len(roots) == 1
    assert (roots[0] / "SHA256SUMS").is_file()
    assert (roots[0] / "main.tex").is_file()
    assert (roots[0] / "supplement.tex").is_file()

    ladder = list(
        (ROOT / "archive" / "curie-weiss-ancilla-ladder-manuscript-superseded").glob(
            "*"
        )
    )
    assert len(ladder) == 1
    assert (ladder[0] / "SHA256SUMS").is_file()
    assert (ladder[0] / "main.pdf").is_file()
