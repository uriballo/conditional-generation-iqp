from pathlib import Path

ROOT = Path(__file__).parents[1]


def _curie_section(text: str) -> str:
    start = text.index(r"\section{Experimental setup}")
    end = text.index(r"\section{Results}", start)
    return text[start:end]


def test_experimental_setup_declares_curie_ancilla_mechanism_controls():
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplement.tex").read_text()
    section = _curie_section(main)
    normalized = " ".join(section.split())
    assert r"n\in\{8,12,16\}" in section
    assert r"r\in\{1,3,5\}" in section
    assert "complete graph on all $n+r$ qubits" in section
    assert "removes every visible--ancilla edge" in normalized
    assert "distinct three-qubit diagonal generators" in section
    assert "isolates the effect of the visible input rank" in section
    assert "No-condition controls are evaluated at $r=0$ and $r=5$" in section
    assert "symmetry-tied" not in main + supplement
    assert "symmetry-aligned" not in main + supplement
    assert "The ancilla analysis tests the three paired differences" in normalized


def test_signed_field_study_is_equivariant_and_outcome_neutral():
    main = (ROOT / "paper" / "main.tex").read_text()
    section = _curie_section(main)
    assert r"p_{\beta,h}(x)" in section
    assert r"\ket{\phi_{\mathrm{eq}}(h)}" in section
    assert r"q_\theta(\bar x\mid-h)=q_\theta(x\mid h)" in section
    assert r"\epsilon_{\mathrm{eq}}(h)" in section
    assert "trained on the same nonnegative fields" in section
    assert "trained on both signs" in section
    assert "incorrectly imposes" in section
    assert "We separated model development from confirmation" in section


def test_confirmatory_protocol_and_historical_results_are_separated():
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplement.tex").read_text()
    section = _curie_section(main)
    assert "ten fresh optimizer seeds" in section
    assert "including negative and mixed comparisons" in section
    assert "performance_thresholds_applied" not in section
    assert "curieweissancillaeffects" in supplement
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
