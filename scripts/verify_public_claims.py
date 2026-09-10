#!/usr/bin/env python3
"""Verify public ICTAI claims against the audited camera-ready manifest.

This checker is intentionally standard-library-only. In CI, ``--documents-only``
checks the code repository without downloading paper artifacts. After the DOI
archive is extracted, ``--artifact-dir`` additionally verifies the source manifest,
machine-readable claims, and rounded statements exposed in README.md and
docs/index.html.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        for name in ("href", "src"):
            if attributes.get(name):
                self.references.append((name, str(attributes[name])))


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: public={actual!r}, source={expected!r}")


def _find(rows: list[dict], **conditions: str) -> dict:
    matches = [row for row in rows if all(row.get(k) == v for k, v in conditions.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def verify_source(public: dict, source_path: Path) -> None:
    if public.get("status") != "frozen camera-ready record; publication records pending":
        raise AssertionError("Public claim-record status is stale or ambiguous")
    source_bytes = source_path.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_hash = public["source_manifest_sha256"]
    if actual_hash != expected_hash:
        raise AssertionError(
            f"Source manifest hash mismatch: public={expected_hash}, actual={actual_hash}"
        )
    source = json.loads(source_bytes)

    if public["scope"]["n_examples"] != source["scope"]["n_examples"]:
        raise AssertionError("Dataset-size mismatch")
    if public["scope"]["n_classes"] != source["scope"]["n_classes"]:
        raise AssertionError("Class-count mismatch")

    headline = public["headline_tfidf_logreg"]
    source_headline = source["classical_predeclared_grid"][0]
    selection = source_headline["selection"]
    test = source_headline["test"]
    exact_fields = {
        "threshold": selection["threshold"],
        "n_test": test["n_test"],
        "n_routed": test["n_routed"],
        "always_route_cost": test["always_route_cost"],
        "policy_cost": test["policy_cost"],
        "coverage_percent": 100 * test["coverage"],
        "routed_accuracy_percent": 100 * test["accuracy_on_routed"],
        "wrong_route_rate_percent": 100 * test["wrong_route_rate"],
        "always_route_wrong_rate_percent": 100 * test["always_route_wrong_rate"],
    }
    for field, expected in exact_fields.items():
        if isinstance(expected, float):
            _assert_close(float(headline[field]), expected, field)
        elif headline[field] != expected:
            raise AssertionError(f"{field}: public={headline[field]!r}, source={expected!r}")

    reduction = test["relative_reduction_vs_always_route"]
    _assert_close(
        headline["relative_reduction_percent"],
        100 * reduction["point_estimate"],
        "relative_reduction_percent",
    )
    for public_value, source_value, label in zip(
        headline["relative_reduction_ci95_percent"],
        (100 * reduction["ci_low"], 100 * reduction["ci_high"]),
        ("reduction_ci_low", "reduction_ci_high"),
    ):
        _assert_close(public_value, source_value, label)

    sensitivity = source["lr_cost_model_sensitivity"]
    sensitivity_public = public["cost_parameter_sensitivity"]
    summary = sensitivity["summary"]
    sensitivity_fields = {
        "n_settings": sensitivity["cost_grid"]["n_settings"],
        "threshold_policy_selected_on_calibration": summary[
            "threshold_policy_selected_on_calibration"
        ],
        "test_cost_lower_than_both_trivial_policies": summary[
            "test_cost_lower_than_both_trivial_policies"
        ],
        "test_cost_equal_to_lower_cost_trivial_policy": summary[
            "test_cost_equal_to_lower_cost_trivial_policy"
        ],
        "test_cost_higher_than_lower_cost_trivial_policy": summary[
            "test_cost_higher_than_lower_cost_trivial_policy"
        ],
        "positive_rows_with_ci_excluding_zero": summary[
            "positive_rows_with_ratio_bootstrap_ci_excluding_zero"
        ],
    }
    for field, expected in sensitivity_fields.items():
        if sensitivity_public[field] != expected:
            raise AssertionError(
                f"cost sensitivity {field}: "
                f"public={sensitivity_public[field]!r}, source={expected!r}"
            )

    raw = _find(
        source["llm_confidence_metrics"],
        model_condition="gpt4o_mini_few_shot_k3__few_shot",
        confidence_method="raw_model_reported",
    )
    iso = _find(
        source["llm_confidence_metrics"],
        model_condition="gpt4o_mini_few_shot_k3__few_shot",
        confidence_method="isotonic",
    )
    caveat = public["confidence_caveat"]
    raw_ece_values = [
        row["ece_10_equal_width_bins"]
        for row in source["llm_confidence_metrics"]
        if row["confidence_method"] == "raw_model_reported"
    ]
    isotonic_ece_values = [
        row["ece_10_equal_width_bins"]
        for row in source["llm_confidence_metrics"]
        if row["confidence_method"] == "isotonic"
    ]
    for public_value, expected, label in zip(
        caveat["llm_raw_ece_range"],
        (round(min(raw_ece_values), 3), round(max(raw_ece_values), 3)),
        ("raw ECE minimum", "raw ECE maximum"),
    ):
        _assert_close(public_value, expected, label)
    for public_value, expected, label in zip(
        caveat["llm_isotonic_ece_range"],
        (round(min(isotonic_ece_values), 3), round(max(isotonic_ece_values), 3)),
        ("isotonic ECE minimum", "isotonic ECE maximum"),
    ):
        _assert_close(public_value, expected, label)
    _assert_close(caveat["gpt4o_mini_few_shot_raw_correctness_auroc"], raw["correctness_auroc"], "raw AUROC")
    _assert_close(caveat["gpt4o_mini_few_shot_isotonic_correctness_auroc"], iso["correctness_auroc"], "isotonic AUROC")

    haiku = _find(
        source["llm_fixed_break_even_policy"],
        model_condition="haiku_few_shot_k3__few_shot",
    )
    llm_public = public["llm_break_even_policy"]
    for field in ("threshold", "n_routed", "n_test", "policy_cost"):
        public_field = {
            "n_routed": "haiku_few_shot_routed",
            "n_test": "haiku_few_shot_test_size",
            "policy_cost": "haiku_few_shot_policy_cost",
        }.get(field, field)
        value = llm_public[public_field]
        expected = haiku[field]
        if isinstance(expected, float):
            _assert_close(value, expected, f"LLM {field}")
        elif value != expected:
            raise AssertionError(f"LLM {field}: public={value!r}, source={expected!r}")

    site = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    distilbert = _find(
        source["classical_predeclared_grid"],
        model="fine-tuned DistilBERT",
    )
    prompted_rows = [
        row
        for row in source["llm_confidence_metrics"]
        if row["confidence_method"] == "raw_model_reported"
    ]
    best_prompted_accuracy = max(row["accuracy"] for row in prompted_rows)
    reductions = [
        row["test"]["relative_reduction_vs_always_route"]["point_estimate"]
        for row in source["classical_predeclared_grid"]
    ]
    source_bound_site_values = (
        f'data-count="{best_prompted_accuracy:.3f}"',
        f'data-count="{distilbert["test_accuracy"]:.3f}"',
        f"Cost reductions of {100 * min(reductions):.1f}–{100 * max(reductions):.1f}%",
        f"seven of eight prompted settings route 0 of {llm_public['haiku_few_shot_test_size']:,} tickets",
        f"{llm_public['haiku_few_shot_routed']} of {llm_public['haiku_few_shot_test_size']:,} (1.9%)",
    )
    normalized_site = " ".join(site.split())
    for value in source_bound_site_values:
        if value not in site and value not in normalized_site:
            raise AssertionError(f"Site value is not bound to the source manifest: {value}")


def verify_page_structure() -> None:
    page_path = ROOT / "docs" / "index.html"
    parser = _PageParser()
    parser.feed(page_path.read_text(encoding="utf-8"))
    if len(parser.ids) != len(set(parser.ids)):
        duplicates = sorted({item for item in parser.ids if parser.ids.count(item) > 1})
        raise AssertionError(f"Site contains duplicate HTML ids: {duplicates}")
    known_ids = set(parser.ids)
    for kind, reference in parser.references:
        if reference.startswith("#"):
            target = reference[1:]
            if target and target not in known_ids:
                raise AssertionError(f"Site fragment does not resolve: {reference}")
            continue
        parsed = urlparse(reference)
        if parsed.scheme or reference.startswith("//"):
            continue
        local_path = (page_path.parent / parsed.path).resolve()
        if not local_path.exists():
            raise AssertionError(f"Site {kind} does not resolve locally: {reference}")


def verify_near_duplicate_record(record_path: Path) -> None:
    record = record_path.read_text(encoding="utf-8")
    required_record_values = (
        "near-dup rate (>=.90): 8.33% of corpus",
        "random-split acc    : 0.8474",
        "group-split acc     : 0.8500",
        "residual group leakage (>=.90): 26/14458 (0.18% of group test)",
    )
    for value in required_record_values:
        if value not in record:
            raise AssertionError(f"Near-duplicate record is missing: {value}")


def verify_documents() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    site = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_site = " ".join(site.split())
    required_readme = (
        "accepted at IEEE ICTAI 2026",
        "The IEEE paper DOI and",
        "archival Xplore URL are not yet available",
        "10.5281/zenodo.22609128",
        "Paper-specific predictions and result records are intentionally not stored in Git",
        "0.477",
        "seven of eight",
        "post-hoc 16-setting cost-parameter analysis",
        "held-out test data in 14 settings",
        "IEEE Xplore will be the authoritative paper",
    )
    required_site = (
        "IEEE paper DOI and Xplore record pending",
        "38.1%",
        "95% CI 35.7–40.4%",
        "correctness AUROC 0.500",
        'data-count="0.477"',
        'data-count="0.877"',
        "Cost reductions of 35.5–38.1%",
        "no jointly trained learning-to-defer method",
        "LLM-only M=8 control",
        "26 of 14,458 group-test tickets (0.18%)",
        "No IEEE Version of Record is hosted on this site.",
        "versioned Zenodo record",
        "routeguard_camera_ready_supplementary.zip",
        "cd routeguard_camera_ready_supplement/code",
        "requirements-camera-ready-lock.txt",
        "--paper-dir ../paper_artifacts",
        "A post-hoc 16-setting cost-parameter analysis selects one policy per setting on calibration; the frozen choice beats both trivial policies on held-out test data in 14 settings.",
    )
    # The artifact/publication boundary now lives in README's
    # "Artifact and publication status" section rather than a separate guide.
    required_artifact_text = (
        "The archive is published separately from this Git repository.",
        "camera_ready_results/claim_manifest.json",
        "GitHub contains no paper-specific prediction vectors or frozen result package",
        "not substitutes for the IEEE record",
    )
    for value in required_readme:
        if value not in readme and value not in normalized_readme:
            raise AssertionError(f"README is missing required text: {value}")
    for value in required_site:
        if value not in site and value not in normalized_site:
            raise AssertionError(f"Site is missing required text: {value}")
    for value in required_artifact_text:
        if value not in readme and value not in normalized_readme:
            raise AssertionError(
                f"README artifact/publication section is missing required text: {value}"
            )
    required_citation = (
        "value: 10.5281/zenodo.22609128",
        "RouteGuard versioned artifact series",
    )
    for value in required_citation:
        if value not in citation:
            raise AssertionError(f"CITATION.cff is missing required text: {value}")
    obsolete_claims = (
        "The paper evaluates sensitivity across combinations of correct-route, triage, and wrong-route costs.",
        "swept over a grid of cost settings for robustness",
        "robust across plausible cost settings",
        "<strong>4 + 4</strong>",
        "The complete frozen bundle and archival DOI remain pending.",
        "The repository also contains the audited, public-safe camera-ready record",
        "record is mirrored under [`artifacts/ictai2026/`]",
        "github.com/sachinkg12/RouteGuard/tree/main/artifacts/ictai2026",
        "Artifact DOI: pending",
        "The DOI is currently pending.",
        "After the DOI is assigned",
        "# DOI artifact extracts to artifacts/ictai2026/",
        "--paper-dir artifacts/ictai2026",
        "10.5281/zenodo.22609129",
        "published as Zenodo version v1.0.0",
    )
    joined = "\n".join((readme, site, citation))
    for claim in obsolete_claims:
        if claim in joined:
            raise AssertionError(f"Public documentation still contains obsolete text: {claim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional explicit path to claim_manifest.json in the DOI artifact.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Extracted DOI artifact directory ending in artifacts/ictai2026.",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Check repository documentation and page structure without paper artifacts.",
    )
    args = parser.parse_args()
    verify_documents()
    verify_page_structure()
    if args.documents_only:
        if args.artifact_dir or args.source_manifest:
            parser.error("--documents-only cannot be combined with artifact paths")
        print("Public documentation and page-structure checks passed.")
        return

    artifact_dir = args.artifact_dir.resolve() if args.artifact_dir else None
    if artifact_dir is None and args.source_manifest:
        artifact_dir = args.source_manifest.resolve().parent.parent
    if artifact_dir is None:
        parser.error("provide --artifact-dir, or use --documents-only")

    public = json.loads((artifact_dir / "public_claims.json").read_text(encoding="utf-8"))
    verify_near_duplicate_record(artifact_dir / "near_duplicate_analysis_result.txt")
    source_manifest = (
        args.source_manifest.resolve()
        if args.source_manifest
        else artifact_dir / "camera_ready_results" / "claim_manifest.json"
    )
    verify_source(public, source_manifest)
    print("Public claim consistency checks passed.")


if __name__ == "__main__":
    main()
