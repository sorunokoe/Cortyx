#!/usr/bin/env python3
"""Manifest-backed benchmark registry for Cortyx."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "benchmarks" / "registry.json"
ALLOWED_COMPETITOR_EVIDENCE_STATES = {
    "same-surface-baseline",
    "published-baseline",
    "capability-note",
    "none",
}
COMPETITOR_EVIDENCE_STATE_REASONS = {
    "same-surface-baseline": "a same-surface baseline cited in the repo",
    "published-baseline": "a published baseline cited in the repo",
    "capability-note": "only capability-note evidence is cited in the repo",
    "none": "no repo evidence is cited for this dimension yet",
}


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def load_benchmarks(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return registry["benchmarks"]


def load_matrix(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return registry.get("proof_matrix", [])


def load_overall_scorecard(registry: dict[str, Any]) -> dict[str, Any]:
    return registry.get("overall_scorecard", {})


def load_guardrail_suites(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return registry.get("guardrail_suites", [])


def allowed_proof_statuses(registry: dict[str, Any]) -> set[str]:
    return set(registry.get("proof_status_legend", {}).keys())


def find_benchmark(benchmarks: list[dict[str, Any]], bench_id: str) -> dict[str, Any]:
    for bench in benchmarks:
        if bench["id"] == bench_id:
            return bench
    raise SystemExit(f"Unknown benchmark id: {bench_id}")


def find_matrix_row(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    for row in rows:
        if row["id"] == row_id:
            return row
    raise SystemExit(f"Unknown proof-matrix id: {row_id}")


def find_guardrail_suite(suites: list[dict[str, Any]], suite_id: str) -> dict[str, Any]:
    for suite in suites:
        if suite["id"] == suite_id:
            return suite
    raise SystemExit(f"Unknown guardrail suite id: {suite_id}")


def filter_benchmarks(
    benchmarks: list[dict[str, Any]],
    *,
    official: bool = False,
    kind: str | None = None,
    surface: str | None = None,
    dimension: str | None = None,
    proof_status: str | None = None,
) -> list[dict[str, Any]]:
    selected = benchmarks
    if official:
        selected = [b for b in selected if b["official"]]
    if kind:
        selected = [b for b in selected if b["kind"] == kind]
    if surface:
        selected = [b for b in selected if b.get("surface") == surface]
    if dimension:
        selected = [b for b in selected if b.get("dimension") == dimension]
    if proof_status:
        selected = [b for b in selected if b.get("proof_status") == proof_status]
    return selected


def filter_matrix(
    rows: list[dict[str, Any]],
    *,
    proof_status: str | None = None,
) -> list[dict[str, Any]]:
    if proof_status:
        return [row for row in rows if row.get("status") == proof_status]
    return rows


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def format_status_map(status_map: dict[str, str]) -> str:
    return ", ".join(f"{entry_id}={status}" for entry_id, status in status_map.items())


def format_state_counts(state_counts: dict[str, int]) -> str:
    return ", ".join(f"{state}={count}" for state, count in state_counts.items())


def format_recorded_outcomes(
    outcomes: list[dict[str, Any]], *, dimension_id: str | None = None
) -> str:
    formatted: list[str] = []
    for outcome in outcomes:
        competitor_id = outcome.get("competitor_id")
        outcome_value = outcome.get("outcome")
        if not isinstance(competitor_id, str) or not isinstance(outcome_value, str):
            continue
        if dimension_id:
            formatted.append(f"{dimension_id} vs {competitor_id}={outcome_value}")
        else:
            formatted.append(f"{competitor_id}={outcome_value}")
    return ", ".join(formatted)


def reverse_outcome(outcome: str) -> str:
    reverse_map = {"win": "loss", "tie": "tie", "loss": "win"}
    try:
        return reverse_map[outcome]
    except KeyError as exc:  # pragma: no cover - validation guards the schema
        raise SystemExit(f"Unknown outcome value: {outcome}") from exc


def competitor_dimension_evidence(
    competitor: dict[str, Any], dimension_id: str
) -> str:
    evidence_by_dimension = competitor.get("dimension_evidence", {})
    if not isinstance(evidence_by_dimension, dict):
        return "none"
    evidence_state = evidence_by_dimension.get(dimension_id, "none")
    if evidence_state not in ALLOWED_COMPETITOR_EVIDENCE_STATES:
        return "none"
    return evidence_state


def build_outcome_ledger(
    *,
    dimension: dict[str, Any],
    competitors: list[dict[str, Any]],
    outcomes_by_competitor: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for competitor in competitors:
        competitor_id = competitor["id"]
        competitor_label = competitor.get("label", competitor_id)
        evidence_state = competitor_dimension_evidence(competitor, dimension["id"])
        recorded_outcome = outcomes_by_competitor.get(competitor_id)
        if recorded_outcome:
            entry_state = "recorded"
            blocking_reason = ""
            summary = recorded_outcome.get("summary", "")
            source_refs = recorded_outcome.get("source_refs", [])
        elif not dimension["counts_today"]:
            entry_state = "blocked-by-proof"
            blocking_reason = dimension["reason"]
            summary = ""
            source_refs = []
        elif evidence_state in {"same-surface-baseline", "published-baseline"}:
            entry_state = "pending-outcome"
            blocking_reason = (
                f"{competitor_label} has "
                f"{COMPETITOR_EVIDENCE_STATE_REASONS[evidence_state]} for "
                f"{dimension['label'].lower()}, but no win/tie/loss record is stored yet."
            )
            summary = ""
            source_refs = []
        elif evidence_state == "capability-note":
            entry_state = "insufficient-evidence"
            blocking_reason = (
                f"{competitor_label} is only covered by capability notes for "
                f"{dimension['label'].lower()}; a same-surface public benchmark or "
                "rubric is still missing."
            )
            summary = ""
            source_refs = []
        else:
            entry_state = "no-repo-evidence"
            blocking_reason = (
                f"No repo-cited competitor evidence exists yet for {competitor_label} on "
                f"{dimension['label'].lower()}."
            )
            summary = ""
            source_refs = []
        entries.append(
            {
                "competitor_id": competitor_id,
                "competitor_label": competitor_label,
                "current_state": entry_state,
                "evidence_state": evidence_state,
                "roster_source_refs": competitor.get("source_refs", []),
                "outcome": recorded_outcome.get("outcome") if recorded_outcome else None,
                "summary": summary,
                "source_refs": source_refs,
                "blocking_reason": blocking_reason,
            }
        )

    state_counts = Counter(entry["current_state"] for entry in entries)
    return {
        "current_state": "complete"
        if set(state_counts) in (set(), {"recorded"})
        else "incomplete",
        "state_counts": dict(sorted(state_counts.items())),
        "entries": entries,
    }


def print_benchmark_table(benchmarks: list[dict[str, Any]]) -> None:
    print("| ID | Dimension | Surface | Kind | Proof | Official | Metric | Current |")
    print("|---|---|---|---|---|---|---|---|")
    for bench in benchmarks:
        print(
            f"| {bench['id']} | {bench.get('dimension', '')} | {bench.get('surface', '')} | "
            f"{bench['kind']} | {bench.get('proof_status', '')} | "
            f"{'yes' if bench['official'] else 'no'} | {bench['metric']} | "
            f"{bench.get('current_result', '')} |"
        )


def print_matrix_table(rows: list[dict[str, Any]]) -> None:
    print("| ID | Label | Status | Live claim | Missing | Evidence IDs |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        evidence = ", ".join(row.get("evidence_ids", []))
        print(
            f"| {row['id']} | {row.get('label', '')} | {row.get('status', '')} | "
            f"{row.get('live_claim', '')} | {row.get('missing', '')} | {evidence} |"
        )


def print_guardrail_table(suites: list[dict[str, Any]]) -> None:
    print("| ID | Proof rows | Benchmarks | CI ready | Summary |")
    print("|---|---|---|---|---|")
    for suite in suites:
        print(
            f"| {suite['id']} | {', '.join(suite.get('proof_matrix_ids', []))} | "
            f"{', '.join(suite.get('benchmark_ids', []))} | "
            f"{'yes' if suite.get('ci_ready') else 'no'} | "
            f"{suite.get('summary', '')} |"
        )


def evaluate_scorecard(registry: dict[str, Any]) -> dict[str, Any]:
    scorecard = load_overall_scorecard(registry)
    if not scorecard:
        raise SystemExit("Registry has no overall_scorecard definition.")

    matrix = load_matrix(registry)
    benchmarks = load_benchmarks(registry)
    matrix_by_id = index_by_id(matrix)
    benchmarks_by_id = index_by_id(benchmarks)
    scoring_model = scorecard.get("scoring_model", {})
    outcome_values = scoring_model.get("outcome_values", {})
    weighted_dimensions = scorecard.get("weighted_dimensions", [])
    total_weight = scoring_model.get("weights_total", 0)
    comparison_scaffold = scorecard.get("comparison_scaffold", {})
    comparison_minimum_competitors = comparison_scaffold.get("minimum_competitors", 0)
    comparison_competitors = comparison_scaffold.get("competitors", [])
    required_scope_fields = comparison_scaffold.get("required_scope_fields", [])
    outcome_ledger_fields = comparison_scaffold.get("outcome_ledger_fields", [])
    comparison_records = comparison_scaffold.get("dimension_records", [])
    comparison_records_by_id = {
        record["dimension_id"]: record
        for record in comparison_records
        if isinstance(record, dict) and isinstance(record.get("dimension_id"), str)
    }
    valid_comparison_competitors = [
        competitor
        for competitor in comparison_competitors
        if isinstance(competitor, dict) and isinstance(competitor.get("id"), str)
    ]
    competitor_labels = {
        competitor["id"]: competitor.get("label", competitor["id"])
        for competitor in valid_comparison_competitors
    }
    competitor_ids = [competitor["id"] for competitor in valid_comparison_competitors]
    if len(competitor_ids) < comparison_minimum_competitors:
        roster_state = "blocked"
        roster_reason = (
            "No shared competitor roster is recorded yet, so the final best-overall "
            "comparison still has no named targets."
        )
        roster_next_flip = (
            "Populate overall_scorecard.comparison_scaffold.competitors with the "
            "shared named competitor roster."
        )
    else:
        roster_state = "ready"
        roster_reason = (
            f"Shared competitor roster names {len(competitor_ids)} repo-cited systems: "
            + ", ".join(competitor_labels[competitor_id] for competitor_id in competitor_ids)
            + "."
        )
        roster_next_flip = (
            "Keep this roster stable across every weighted dimension and only record "
            "same-surface outcomes against these named systems."
        )

    evaluated_dimensions: list[dict[str, Any]] = []
    eligible_weight = 0
    blocked_dimension_ids: list[str] = []

    for dimension in weighted_dimensions:
        proof_matrix_ids = dimension.get("proof_matrix_ids", [])
        eligible_statuses = set(dimension.get("eligible_statuses", []))
        required_kinds = dimension.get("required_benchmark_kinds", [])
        current_statuses = {
            row_id: matrix_by_id[row_id]["status"]
            for row_id in proof_matrix_ids
            if row_id in matrix_by_id
        }
        evidence_ids = [
            evidence_id
            for row_id in proof_matrix_ids
            if row_id in matrix_by_id
            for evidence_id in matrix_by_id[row_id].get("evidence_ids", [])
        ]
        available_kinds = sorted(
            {
                benchmarks_by_id[evidence_id].get("kind", "")
                for evidence_id in evidence_ids
                if evidence_id in benchmarks_by_id
            }
        )
        blocked_rows = [
            row_id
            for row_id, status in current_statuses.items()
            if status not in eligible_statuses
        ]
        missing_kinds = [
            kind for kind in required_kinds if kind not in available_kinds
        ]
        counts_today = not blocked_rows and not missing_kinds
        if counts_today:
            eligible_weight += int(dimension["weight"])
            reason = (
                f"Counts today: {format_status_map(current_statuses)}; "
                f"evidence kinds include {', '.join(required_kinds)}."
            )
        elif blocked_rows:
            blocked_dimension_ids.append(dimension["id"])
            reasons = [
                f"{row_id} is {current_statuses[row_id]} (needs one of {', '.join(sorted(eligible_statuses))})"
                for row_id in blocked_rows
            ]
            reason = f"Blocked today: {'; '.join(reasons)}."
        else:
            blocked_dimension_ids.append(dimension["id"])
            reason = (
                "Blocked today: missing benchmark kinds "
                f"{', '.join(missing_kinds)}."
            )

        evaluated_dimensions.append(
            {
                **dimension,
                "current_statuses": current_statuses,
                "evidence_ids": evidence_ids,
                "available_benchmark_kinds": available_kinds,
                "counts_today": counts_today,
                "reason": reason,
            }
        )

    weighted_dimensions_by_id = index_by_id(evaluated_dimensions)
    competitor_scores: dict[str, dict[str, Any]] = {
        competitor_id: {
            "competitor_id": competitor_id,
            "competitor_label": competitor_labels[competitor_id],
            "cortyx_score": 0.0,
            "competitor_score": 0.0,
            "delta": 0.0,
            "current_state": "incomplete",
            "missing_dimension_ids": [],
            "dimension_outcomes": [],
        }
        for competitor_id in competitor_ids
    }
    evaluated_dimension_records: list[dict[str, Any]] = []
    scored_dimension_ids: list[str] = []

    for dimension in evaluated_dimensions:
        comparison_record = comparison_records_by_id.get(dimension["id"], {})
        scope = comparison_record.get("scope", {})
        if not isinstance(scope, dict):
            scope = {}
        recorded_outcomes = comparison_record.get("recorded_outcomes", [])
        if not isinstance(recorded_outcomes, list):
            recorded_outcomes = []
        outcomes_by_competitor = {
            record["competitor_id"]: record
            for record in recorded_outcomes
            if isinstance(record, dict) and isinstance(record.get("competitor_id"), str)
        }
        recorded_outcome_summary = format_recorded_outcomes(recorded_outcomes)
        missing_scope_fields = [
            field
            for field in required_scope_fields
            if not isinstance(scope.get(field), str) or not scope.get(field, "").strip()
        ]
        missing_competitor_ids = [
            competitor_id
            for competitor_id in competitor_ids
            if competitor_id not in outcomes_by_competitor
        ]
        outcome_ledger = build_outcome_ledger(
            dimension=dimension,
            competitors=valid_comparison_competitors,
            outcomes_by_competitor=outcomes_by_competitor,
        )
        pending_outcome_ids = [
            entry["competitor_id"]
            for entry in outcome_ledger["entries"]
            if entry["current_state"] == "pending-outcome"
        ]
        insufficient_evidence_ids = [
            entry["competitor_id"]
            for entry in outcome_ledger["entries"]
            if entry["current_state"] == "insufficient-evidence"
        ]
        no_repo_evidence_ids = [
            entry["competitor_id"]
            for entry in outcome_ledger["entries"]
            if entry["current_state"] == "no-repo-evidence"
        ]

        if not dimension["counts_today"]:
            current_state = "blocked-by-proof"
            reason = dimension["reason"]
            next_flip = (
                "Promote "
                f"{', '.join(dimension.get('proof_matrix_ids', []))} to one of "
                f"{', '.join(dimension.get('eligible_statuses', []))} with "
                f"{', '.join(dimension.get('required_benchmark_kinds', []))} evidence."
            )
        elif len(competitor_ids) < comparison_minimum_competitors:
            current_state = "awaiting-comparator-set"
            reason = (
                f"{dimension['label']} is claim-eligible, but "
                "overall_scorecard.comparison_scaffold.competitors is still empty."
            )
            next_flip = (
                "Populate overall_scorecard.comparison_scaffold.competitors with the "
                f"shared named competitor roster (minimum {comparison_minimum_competitors})."
            )
        elif missing_scope_fields:
            current_state = "awaiting-scope"
            reason = (
                f"{dimension['label']} is claim-eligible, but scope fields are blank: "
                f"{', '.join(missing_scope_fields)}."
            )
            next_flip = (
                f"Fill the comparison_scaffold scope for {dimension['id']}: "
                f"{', '.join(missing_scope_fields)}."
            )
        elif insufficient_evidence_ids or no_repo_evidence_ids:
            current_state = "awaiting-evidence"
            ledger_reasons: list[str] = []
            if recorded_outcome_summary:
                ledger_reasons.append("recorded outcomes " + recorded_outcome_summary)
            if pending_outcome_ids:
                ledger_reasons.append(
                    "pending scorecard outcomes for "
                    + ", ".join(pending_outcome_ids)
                )
            if insufficient_evidence_ids:
                ledger_reasons.append(
                    "only capability-note evidence for "
                    + ", ".join(insufficient_evidence_ids)
                )
            if no_repo_evidence_ids:
                ledger_reasons.append(
                    "no repo evidence for " + ", ".join(no_repo_evidence_ids)
                )
            reason = (
                f"{dimension['label']} scope is defined, but the outcome ledger is still incomplete: "
                + "; ".join(ledger_reasons)
                + "."
            )
            next_flip = (
                "Add same-surface public comparator evidence (or revise the shared "
                f"roster) before recording {dimension['id']} outcomes."
            )
        elif pending_outcome_ids:
            current_state = "awaiting-outcomes"
            reason = (
                f"{dimension['label']} scope is defined, but scorecard outcomes are "
                f"still missing for: {', '.join(pending_outcome_ids)}."
            )
            if recorded_outcome_summary:
                reason += f" Recorded outcomes so far: {recorded_outcome_summary}."
            next_flip = (
                f"Record win/tie/loss plus source refs for {dimension['id']} against "
                "every named competitor with scoreable evidence."
            )
        else:
            current_state = "scored"
            reason = (
                f"{dimension['label']} has recorded outcomes for every named competitor."
            )
            next_flip = "None."
            scored_dimension_ids.append(dimension["id"])
            for competitor_id in competitor_ids:
                recorded_outcome = outcomes_by_competitor[competitor_id]
                cortyx_value = float(outcome_values[recorded_outcome["outcome"]])
                competitor_value = float(
                    outcome_values[reverse_outcome(recorded_outcome["outcome"])]
                )
                weighted_points = float(dimension["weight"]) * cortyx_value
                competitor_points = float(dimension["weight"]) * competitor_value
                competitor_scores[competitor_id]["cortyx_score"] += weighted_points
                competitor_scores[competitor_id]["competitor_score"] += competitor_points
                competitor_scores[competitor_id]["dimension_outcomes"].append(
                    {
                        "dimension_id": dimension["id"],
                        "label": dimension["label"],
                        "weight": dimension["weight"],
                        "outcome": recorded_outcome["outcome"],
                        "weighted_points": weighted_points,
                        "competitor_weighted_points": competitor_points,
                        "summary": recorded_outcome.get("summary", ""),
                        "source_refs": recorded_outcome.get("source_refs", []),
                    }
                )

        evaluated_dimension_records.append(
            {
                "dimension_id": dimension["id"],
                "label": dimension["label"],
                "proof_ready": dimension["counts_today"],
                "proof_reason": dimension["reason"],
                "current_state": current_state,
                "expected_competitor_ids": competitor_ids,
                "recorded_competitor_ids": sorted(outcomes_by_competitor.keys()),
                "missing_competitor_ids": missing_competitor_ids,
                "pending_outcome_ids": pending_outcome_ids,
                "insufficient_evidence_ids": insufficient_evidence_ids,
                "no_repo_evidence_ids": no_repo_evidence_ids,
                "recorded_outcome_summary": recorded_outcome_summary,
                "scope": scope,
                "missing_scope_fields": missing_scope_fields,
                "recorded_outcomes": recorded_outcomes,
                "outcome_ledger": {
                    **outcome_ledger,
                    "fields": outcome_ledger_fields,
                },
                "reason": reason,
                "next_flip": next_flip,
            }
        )

    incomplete_dimension_ids = [
        record["dimension_id"]
        for record in evaluated_dimension_records
        if record["current_state"] != "scored"
    ]
    for score in competitor_scores.values():
        score["missing_dimension_ids"] = incomplete_dimension_ids
        if incomplete_dimension_ids:
            score["current_state"] = "incomplete"
        elif score["cortyx_score"] > score["competitor_score"]:
            score["current_state"] = "ahead"
        elif score["cortyx_score"] < score["competitor_score"]:
            score["current_state"] = "behind"
        else:
            score["current_state"] = "tied"
        score["delta"] = score["cortyx_score"] - score["competitor_score"]

    evaluated_comparison_records_by_id = {
        record["dimension_id"]: record for record in evaluated_dimension_records
    }

    if len(competitor_ids) < comparison_minimum_competitors:
        comparison_scaffold_state = "blocked"
        comparison_scaffold_reason = (
            "No named competitor set is recorded yet, so the final best-overall "
            "comparison cannot be scored."
        )
        comparison_scaffold_next_flip = (
            "Populate overall_scorecard.comparison_scaffold.competitors with the "
            "shared named competitor roster."
        )
    else:
        scope_blockers = [
            f"{record['dimension_id']} ({', '.join(record['missing_scope_fields'])})"
            for record in evaluated_dimension_records
            if record["proof_ready"] and record["missing_scope_fields"]
        ]
        if scope_blockers:
            comparison_scaffold_state = "blocked"
            comparison_scaffold_reason = (
                "Named competitors exist, but claim-eligible dimensions still have "
                "blank comparison scope fields: "
                + "; ".join(scope_blockers)
                + "."
            )
            comparison_scaffold_next_flip = (
                "Fill the comparison_scaffold scope metadata for every "
                "claim-eligible weighted dimension."
            )
        else:
            comparison_scaffold_state = "ready"
            comparison_scaffold_reason = (
                f"Named competitor roster has {len(competitor_ids)} entries and every "
                "claim-eligible weighted dimension has scope metadata."
            )
            comparison_scaffold_next_flip = "None."

    evaluated_must_win: list[dict[str, Any]] = []
    unsatisfied_must_win_gate_ids: list[str] = []
    for gate in scorecard.get("must_win_gates", []):
        blocked_dimensions = [
            dimension_id
            for dimension_id in gate.get("dimension_ids", [])
            if not weighted_dimensions_by_id[dimension_id]["counts_today"]
        ]
        if blocked_dimensions:
            reasons = [
                f"{dimension_id}: {weighted_dimensions_by_id[dimension_id]['reason']}"
                for dimension_id in blocked_dimensions
            ]
            current_state = "blocked-by-proof"
            reason = "Blocked today: " + " ".join(reasons)
            next_flip = "Promote the blocked weighted dimensions to claim-eligible proof."
            unsatisfied_must_win_gate_ids.append(gate["id"])
        elif len(competitor_ids) < comparison_minimum_competitors:
            current_state = "awaiting-comparator-set"
            reason = (
                "Claim-eligible today, but the named competitor set is still empty, so "
                "there is no must-win ledger to evaluate."
            )
            next_flip = (
                "Populate the comparison_scaffold competitor roster, then record "
                "must-win outcomes."
            )
            unsatisfied_must_win_gate_ids.append(gate["id"])
        else:
            scope_blockers = [
                dimension_id
                for dimension_id in gate.get("dimension_ids", [])
                if evaluated_comparison_records_by_id[dimension_id]["missing_scope_fields"]
            ]
            evidence_blockers = [
                dimension_id
                for dimension_id in gate.get("dimension_ids", [])
                if evaluated_comparison_records_by_id[dimension_id]["current_state"]
                == "awaiting-evidence"
            ]
            outcome_blockers = [
                dimension_id
                for dimension_id in gate.get("dimension_ids", [])
                if evaluated_comparison_records_by_id[dimension_id]["current_state"]
                == "awaiting-outcomes"
            ]
            other_incomplete_dimensions = [
                {
                    "dimension_id": dimension_id,
                    "state": evaluated_comparison_records_by_id[dimension_id][
                        "current_state"
                    ],
                }
                for dimension_id in gate.get("dimension_ids", [])
                if evaluated_comparison_records_by_id[dimension_id]["current_state"]
                not in {"scored", "awaiting-scope", "awaiting-evidence", "awaiting-outcomes"}
            ]
            non_win_results = [
                {
                    "dimension_id": dimension_id,
                    "competitor_id": recorded_outcome["competitor_id"],
                    "outcome": recorded_outcome["outcome"],
                }
                for dimension_id in gate.get("dimension_ids", [])
                for recorded_outcome in evaluated_comparison_records_by_id[dimension_id][
                    "recorded_outcomes"
                ]
                if recorded_outcome.get("outcome") != "win"
            ]
            recorded_gate_outcomes = [
                {
                    "dimension_id": dimension_id,
                    "competitor_id": recorded_outcome["competitor_id"],
                    "outcome": recorded_outcome["outcome"],
                }
                for dimension_id in gate.get("dimension_ids", [])
                for recorded_outcome in evaluated_comparison_records_by_id[dimension_id][
                    "recorded_outcomes"
                ]
            ]
            recorded_gate_summary = "; ".join(
                f"{result['dimension_id']} vs {result['competitor_id']}={result['outcome']}"
                for result in recorded_gate_outcomes
            )
            if scope_blockers:
                current_state = "awaiting-scope"
                reason = (
                    "Claim-eligible today, but comparison scope fields are still blank for "
                    f"{', '.join(scope_blockers)}."
                )
                next_flip = "Fill the missing comparison scope fields before scoring the gate."
                unsatisfied_must_win_gate_ids.append(gate["id"])
            elif non_win_results:
                current_state = "blocked"
                reason = (
                    "Recorded outcomes are not all wins: "
                    + "; ".join(
                        f"{result['dimension_id']} vs {result['competitor_id']}="
                        f"{result['outcome']}"
                        for result in non_win_results
                    )
                    + "."
                )
                if evidence_blockers:
                    reason += (
                        " Same-surface competitor evidence is still missing for "
                        + ", ".join(evidence_blockers)
                        + "."
                    )
                elif outcome_blockers:
                    reason += (
                        " Additional scorecard outcomes are still missing for "
                        + ", ".join(outcome_blockers)
                        + "."
                    )
                next_flip = (
                    "The public claim remains disallowed until every must-win dimension "
                    "records only wins and any remaining same-surface evidence gaps are filled."
                )
                unsatisfied_must_win_gate_ids.append(gate["id"])
            elif evidence_blockers:
                current_state = "awaiting-evidence"
                reason = (
                    "Claim-eligible today, but apples-to-apples competitor evidence is "
                    f"still missing for {', '.join(evidence_blockers)}."
                )
                if recorded_gate_summary:
                    reason += f" Recorded outcomes so far: {recorded_gate_summary}."
                next_flip = (
                    "Add same-surface public benchmark evidence (or revise the shared "
                    "roster) before scoring the must-win gate."
                )
                unsatisfied_must_win_gate_ids.append(gate["id"])
            elif outcome_blockers:
                current_state = "awaiting-outcomes"
                reason = (
                    "Claim-eligible today, but must-win outcomes are still missing for "
                    f"{', '.join(outcome_blockers)}."
                )
                if recorded_gate_summary:
                    reason += f" Recorded outcomes so far: {recorded_gate_summary}."
                next_flip = "Record win/tie/loss outcomes for the must-win dimensions."
                unsatisfied_must_win_gate_ids.append(gate["id"])
            elif other_incomplete_dimensions:
                current_state = other_incomplete_dimensions[0]["state"]
                reason = (
                    "Must-win dimensions are not fully scored yet: "
                    + "; ".join(
                        f"{entry['dimension_id']}={entry['state']}"
                        for entry in other_incomplete_dimensions
                    )
                    + "."
                )
                next_flip = "Finish the missing comparator-readiness work before scoring."
                unsatisfied_must_win_gate_ids.append(gate["id"])
            else:
                current_state = "satisfied"
                reason = (
                    "Every named competitor is recorded as a win across the gate's "
                    "required dimensions."
                )
                next_flip = "None."
        evaluated_must_win.append(
            {
                **gate,
                "current_state": current_state,
                "reason": reason,
                "next_flip": next_flip,
            }
        )

    evaluated_regressions: list[dict[str, Any]] = []
    regression_blockers: list[str] = []
    for gate in scorecard.get("must_not_regress_gates", []):
        required_statuses = set(gate.get("required_statuses", []))
        current_statuses = {
            row_id: matrix_by_id[row_id]["status"]
            for row_id in gate.get("proof_matrix_ids", [])
            if row_id in matrix_by_id
        }
        blocked_rows = [
            row_id
            for row_id, status in current_statuses.items()
            if status not in required_statuses
        ]
        if blocked_rows:
            current_state = "blocked"
            regression_blockers.append(gate["id"])
            reasons = [
                f"{row_id} is {current_statuses[row_id]} (needs one of {', '.join(sorted(required_statuses))})"
                for row_id in blocked_rows
            ]
            reason = f"Gate failed: {'; '.join(reasons)}."
        else:
            current_state = "green"
            reason = f"Green today: {format_status_map(current_statuses)}."
        evaluated_regressions.append(
            {
                **gate,
                "current_statuses": current_statuses,
                "current_state": current_state,
                "reason": reason,
            }
        )

    evaluated_non_weighted: list[dict[str, Any]] = []
    for surface in scorecard.get("non_weighted_surfaces", []):
        current_statuses = {
            row_id: matrix_by_id[row_id]["status"]
            for row_id in surface.get("proof_matrix_ids", [])
            if row_id in matrix_by_id
        }
        evaluated_non_weighted.append(
            {
                **surface,
                "current_statuses": current_statuses,
                "reason": f"{surface['role']}: {format_status_map(current_statuses)}.",
            }
        )

    ready_to_score = (
        not blocked_dimension_ids
        and not regression_blockers
        and comparison_scaffold_state == "ready"
    )
    all_weighted_dimensions_scored = all(
        record["current_state"] == "scored" for record in evaluated_dimension_records
    )
    ready_to_claim = (
        ready_to_score
        and all_weighted_dimensions_scored
        and bool(competitor_scores)
        and not unsatisfied_must_win_gate_ids
        and all(score["current_state"] == "ahead" for score in competitor_scores.values())
    )

    if ready_to_claim:
        claim_state = "claimable"
    elif ready_to_score and all_weighted_dimensions_scored:
        claim_state = "not-best-overall"
    elif ready_to_score:
        claim_state = "ready-to-score"
    else:
        claim_state = "blocked"

    readiness_phases: list[dict[str, Any]] = []
    if blocked_dimension_ids:
        readiness_phases.append(
            {
                "id": "proof-eligibility",
                "label": "Weighted proof eligibility",
                "current_state": "blocked",
                "blocking_ids": blocked_dimension_ids,
                "reason": (
                    "Weighted dimensions still blocked by proof status: "
                    + ", ".join(blocked_dimension_ids)
                    + "."
                ),
                "next_flip": (
                    "Promote the blocked weighted dimensions to proven claim surfaces."
                ),
            }
        )
    else:
        readiness_phases.append(
            {
                "id": "proof-eligibility",
                "label": "Weighted proof eligibility",
                "current_state": "ready",
                "blocking_ids": [],
                "reason": "Every weighted dimension is claim-eligible today.",
                "next_flip": "None.",
            }
        )

    readiness_phases.append(
        {
            "id": "comparator-roster",
            "label": "Comparator roster",
            "current_state": "ready" if roster_state == "ready" else "blocked",
            "blocking_ids": []
            if roster_state == "ready"
            else ["named-competitors"],
            "reason": roster_reason,
            "next_flip": roster_next_flip,
        }
    )

    scope_blocking_dimensions = [
        record["dimension_id"]
        for record in evaluated_dimension_records
        if record["proof_ready"] and record["missing_scope_fields"]
    ]
    comparison_scope_blocking_ids = ["named-competitors"] if comparison_scaffold_state == "blocked" and len(competitor_ids) < comparison_minimum_competitors else scope_blocking_dimensions
    readiness_phases.append(
        {
            "id": "comparator-scope",
            "label": "Comparator scope",
            "current_state": "ready" if comparison_scaffold_state == "ready" else "blocked",
            "blocking_ids": comparison_scope_blocking_ids,
            "reason": comparison_scaffold_reason,
            "next_flip": comparison_scaffold_next_flip,
        }
    )

    weighted_outcome_blockers = [
        record["dimension_id"]
        for record in evaluated_dimension_records
        if record["current_state"] != "scored"
    ]
    if weighted_outcome_blockers:
        weighted_outcome_reason = (
            "Weighted outcome ledger is still incomplete: "
            + "; ".join(
                f"{record['dimension_id']}={record['current_state']}"
                + (
                    f" ({format_state_counts(record['outcome_ledger']['state_counts'])})"
                    if record.get("outcome_ledger", {}).get("state_counts")
                    else ""
                )
                for record in evaluated_dimension_records
                if record["current_state"] != "scored"
            )
            + "."
        )
        weighted_outcome_next_flip = (
            "Once proof and comparator scope are ready, record win/tie/loss outcomes "
            "for every weighted dimension."
        )
        weighted_outcome_state = "blocked"
    else:
        weighted_outcome_reason = (
            "Every weighted dimension has a complete win/tie/loss ledger."
        )
        weighted_outcome_next_flip = "None."
        weighted_outcome_state = "ready"
    readiness_phases.append(
        {
            "id": "weighted-outcomes",
            "label": "Weighted outcome ledger",
            "current_state": weighted_outcome_state,
            "blocking_ids": weighted_outcome_blockers,
            "reason": weighted_outcome_reason,
            "next_flip": weighted_outcome_next_flip,
        }
    )

    unsatisfied_must_win_gates = [
        gate for gate in evaluated_must_win if gate["current_state"] != "satisfied"
    ]
    if unsatisfied_must_win_gates:
        readiness_phases.append(
            {
                "id": "must-win-outcomes",
                "label": "Must-win outcomes",
                "current_state": "blocked",
                "blocking_ids": [gate["id"] for gate in unsatisfied_must_win_gates],
                "reason": (
                    "Must-win gates are not satisfied: "
                    + "; ".join(
                        f"{gate['id']}={gate['current_state']}"
                        for gate in unsatisfied_must_win_gates
                    )
                    + "."
                ),
                "next_flip": (
                    "Retrieval, answer quality, and collaboration/shared memory must "
                    "all record wins before the claim can unlock."
                ),
            }
        )
    else:
        readiness_phases.append(
            {
                "id": "must-win-outcomes",
                "label": "Must-win outcomes",
                "current_state": "satisfied",
                "blocking_ids": [],
                "reason": "Every must-win gate is recorded as a win.",
                "next_flip": "None.",
            }
        )

    if regression_blockers:
        readiness_phases.append(
            {
                "id": "must-not-regress",
                "label": "Must-not-regress gates",
                "current_state": "blocked",
                "blocking_ids": regression_blockers,
                "reason": (
                    "Must-not-regress gates failed: " + ", ".join(regression_blockers) + "."
                ),
                "next_flip": "Restore the blocked guardrails before using any overall claim.",
            }
        )
    else:
        readiness_phases.append(
            {
                "id": "must-not-regress",
                "label": "Must-not-regress gates",
                "current_state": "green",
                "blocking_ids": [],
                "reason": "Retrieval, speed, token economy, and footprint remain green.",
                "next_flip": "None.",
            }
        )

    remaining_requirements: list[str] = []
    for phase in readiness_phases:
        if phase["current_state"] in {"ready", "green", "satisfied"}:
            continue
        remaining_requirements.append(phase["reason"])
        if phase["next_flip"] != "None.":
            remaining_requirements.append(phase["next_flip"])
    deduped_requirements: list[str] = []
    seen_requirements: set[str] = set()
    for requirement in remaining_requirements:
        if requirement in seen_requirements:
            continue
        seen_requirements.add(requirement)
        deduped_requirements.append(requirement)

    return {
        "id": scorecard["id"],
        "label": scorecard["label"],
        "summary": scorecard.get("summary", ""),
        "claim_state": claim_state,
        "eligible_weight": eligible_weight,
        "total_weight": total_weight,
        "eligible_dimension_ids": [
            dimension["id"]
            for dimension in evaluated_dimensions
            if dimension["counts_today"]
        ],
        "blocked_dimension_ids": blocked_dimension_ids,
        "scoring_model": scoring_model,
        "weighted_dimensions": evaluated_dimensions,
        "counting_rules": scorecard.get("counting_rules", []),
        "comparison_rules": scorecard.get("comparison_rules", []),
        "comparison_scaffold": {
            "minimum_competitors": comparison_minimum_competitors,
            "require_same_competitor_set": comparison_scaffold.get(
                "require_same_competitor_set", False
            ),
            "roster": {
                "current_state": roster_state,
                "reason": roster_reason,
                "next_flip": roster_next_flip,
                "competitor_count": len(competitor_ids),
                "competitor_ids": competitor_ids,
            },
            "competitors": comparison_competitors,
            "required_scope_fields": required_scope_fields,
            "outcome_ledger_fields": outcome_ledger_fields,
            "current_state": comparison_scaffold_state,
            "reason": comparison_scaffold_reason,
            "next_flip": comparison_scaffold_next_flip,
            "dimension_records": evaluated_dimension_records,
            "scored_dimension_ids": scored_dimension_ids,
        },
        "competitor_scores": list(competitor_scores.values()),
        "must_win_gates": evaluated_must_win,
        "must_not_regress_gates": evaluated_regressions,
        "non_weighted_surfaces": evaluated_non_weighted,
        "claim_readiness": {
            "current_state": claim_state,
            "ready_to_score": ready_to_score,
            "ready_to_claim": ready_to_claim,
            "phases": readiness_phases,
        },
        "remaining_requirements": deduped_requirements,
    }


def print_scorecard_table(evaluation: dict[str, Any]) -> None:
    outcomes = evaluation.get("scoring_model", {}).get("outcome_values", {})
    print(f"Claim state: {evaluation['claim_state']}")
    claim_readiness = evaluation.get("claim_readiness", {})
    print(
        "Readiness: "
        f"ready_to_score={'yes' if claim_readiness.get('ready_to_score') else 'no'}, "
        f"ready_to_claim={'yes' if claim_readiness.get('ready_to_claim') else 'no'}"
    )
    print(f"Eligible weight today: {evaluation['eligible_weight']}/{evaluation['total_weight']}")
    print(
        "Scoring model: "
        f"{evaluation.get('scoring_model', {}).get('id', '')} "
        f"(win={outcomes.get('win')}, tie={outcomes.get('tie')}, loss={outcomes.get('loss')})"
    )
    print("| Dimension | Weight | Proof rows | Current statuses | Counts today | Reason |")
    print("|---|---|---|---|---|---|")
    for dimension in evaluation.get("weighted_dimensions", []):
        print(
            f"| {dimension['label']} | {dimension['weight']} | "
            f"{', '.join(dimension.get('proof_matrix_ids', []))} | "
            f"{format_status_map(dimension.get('current_statuses', {}))} | "
            f"{'yes' if dimension['counts_today'] else 'no'} | {dimension['reason']} |"
        )

    print("\nMust-win gates:")
    for gate in evaluation.get("must_win_gates", []):
        print(
            f"- {gate['label']} [{gate['current_state']}]: {gate['reason']} "
            f"Next: {gate['next_flip']}"
        )

    print("\nMust-not-regress gates:")
    for gate in evaluation.get("must_not_regress_gates", []):
        print(f"- {gate['label']} [{gate['current_state']}]: {gate['reason']}")

    print("\nNon-weighted surfaces:")
    for surface in evaluation.get("non_weighted_surfaces", []):
        print(f"- {surface['label']} [{surface['role']}]: {surface['reason']}")

    comparison_scaffold = evaluation.get("comparison_scaffold", {})
    roster = comparison_scaffold.get("roster", {})
    print("\nComparator roster:")
    print(
        f"- roster [{roster.get('current_state', '')}]: {roster.get('reason', '')} "
        f"Next: {roster.get('next_flip', '')}"
    )

    print("\nComparison scaffold:")
    print(
        f"- scope [{comparison_scaffold.get('current_state', '')}]: "
        f"{comparison_scaffold.get('reason', '')} "
        f"Next: {comparison_scaffold.get('next_flip', '')}"
    )
    for record in comparison_scaffold.get("dimension_records", []):
        ledger = record.get("outcome_ledger", {})
        ledger_counts = format_state_counts(ledger.get("state_counts", {}))
        print(
            f"- {record['label']} [{record['current_state']}]: {record['reason']} "
            f"Ledger: {ledger_counts or 'none'}. Next: {record['next_flip']}"
        )
        recorded_summary = record.get("recorded_outcome_summary", "")
        if recorded_summary:
            print(f"  Recorded outcomes: {recorded_summary}")

    print("\nCompetitor totals:")
    competitor_scores = evaluation.get("competitor_scores", [])
    if not competitor_scores:
        print("- none recorded yet")
    for competitor in competitor_scores:
        if competitor["current_state"] == "incomplete":
            print(
                f"- {competitor['competitor_label']} [incomplete]: "
                f"missing {', '.join(competitor['missing_dimension_ids'])}"
            )
            continue
        print(
            f"- {competitor['competitor_label']} [{competitor['current_state']}]: "
            f"Cortyx {competitor['cortyx_score']} vs competitor "
            f"{competitor['competitor_score']} (delta {competitor['delta']})"
        )

    print("\nClaim readiness:")
    for phase in claim_readiness.get("phases", []):
        print(
            f"- {phase['label']} [{phase['current_state']}]: {phase['reason']} "
            f"Next: {phase['next_flip']}"
        )

    print("\nRemaining requirements:")
    for requirement in evaluation.get("remaining_requirements", []):
        print(f"- {requirement}")


def run_benchmark(bench: dict[str, Any]) -> int:
    command = bench.get("command")
    if not command:
        raise SystemExit(f"Benchmark {bench['id']} has no executable command.")
    print(f"▶ {bench['id']} — {bench['name']}")
    print(f"  dimension: {bench.get('dimension', '')}")
    print(f"  proof: {bench.get('proof_status', '')}")
    print(f"  metric: {bench['metric']}")
    print(f"  command: {command}")
    print(f"  notes: {bench['notes']}")
    return subprocess.run(command, cwd=REPO_ROOT, shell=True).returncode


def run_guardrail_suite(suite: dict[str, Any], benchmarks: list[dict[str, Any]]) -> int:
    benchmark_ids = suite.get("benchmark_ids", [])
    if not benchmark_ids:
        raise SystemExit(f"Guardrail suite {suite['id']} has no benchmark_ids.")
    print(f"== Guardrail suite: {suite['id']} — {suite['label']} ==")
    print(f"proof rows: {', '.join(suite.get('proof_matrix_ids', []))}")
    print(f"summary: {suite.get('summary', '')}")
    print(f"notes: {suite.get('notes', '')}")
    seen: set[str] = set()
    for bench_id in benchmark_ids:
        if bench_id in seen:
            continue
        seen.add(bench_id)
        code = run_benchmark(find_benchmark(benchmarks, bench_id))
        if code != 0:
            return code
    return 0


def validate_registry(registry: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    benchmarks = load_benchmarks(registry)
    matrix = load_matrix(registry)
    scorecard = load_overall_scorecard(registry)
    guardrail_suites = load_guardrail_suites(registry)
    allowed_statuses = allowed_proof_statuses(registry)
    benchmarks_by_id = index_by_id(benchmarks)

    if not allowed_statuses:
        problems.append("proof_status_legend must define at least one status")

    benchmark_ids: set[str] = set()
    matrix_ids: set[str] = set()
    referenced_ids: set[str] = set()

    required_benchmark_fields = {
        "id",
        "name",
        "official",
        "surface",
        "dimension",
        "kind",
        "proof_status",
        "metric",
        "current_result",
        "notes",
    }
    required_matrix_fields = {
        "id",
        "label",
        "status",
        "live_claim",
        "honest_read",
        "missing",
        "evidence_ids",
    }

    for bench in benchmarks:
        missing_fields = sorted(required_benchmark_fields - bench.keys())
        if missing_fields:
            problems.append(
                f"benchmark {bench.get('id', '<missing-id>')} missing fields: {', '.join(missing_fields)}"
            )
            continue
        bench_id = bench["id"]
        if bench_id in benchmark_ids:
            problems.append(f"duplicate benchmark id: {bench_id}")
        benchmark_ids.add(bench_id)

        proof_status = bench.get("proof_status")
        if proof_status not in allowed_statuses:
            problems.append(f"benchmark {bench_id} has unknown proof_status: {proof_status}")
        if bench.get("official") and proof_status != "proven":
            problems.append(f"official benchmark {bench_id} must use proof_status=proven")
        if proof_status != "pending" and not bench.get("command"):
            problems.append(f"benchmark {bench_id} must provide a command unless proof_status=pending")

    for row in matrix:
        missing_fields = sorted(required_matrix_fields - row.keys())
        if missing_fields:
            problems.append(
                f"proof matrix row {row.get('id', '<missing-id>')} missing fields: {', '.join(missing_fields)}"
            )
            continue
        row_id = row["id"]
        if row_id in matrix_ids:
            problems.append(f"duplicate proof-matrix id: {row_id}")
        matrix_ids.add(row_id)

        status = row.get("status")
        if status not in allowed_statuses:
            problems.append(f"proof matrix row {row_id} has unknown status: {status}")
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
            problems.append(f"proof matrix row {row_id} must use a string list for evidence_ids")
            continue
        if status != "pending" and not evidence_ids:
            problems.append(f"proof matrix row {row_id} must reference evidence_ids unless status=pending")
        referenced_ids.update(evidence_ids)

    if matrix_ids:
        for bench in benchmarks:
            if bench.get("dimension") not in matrix_ids:
                problems.append(
                    f"benchmark {bench.get('id', '<missing-id>')} references unknown dimension {bench.get('dimension')!r}"
                )

    unknown_refs = sorted(referenced_ids - benchmark_ids)
    if unknown_refs:
        problems.append(f"proof matrix references unknown benchmark ids: {', '.join(unknown_refs)}")

    orphaned_benchmarks = sorted(benchmark_ids - referenced_ids)
    if orphaned_benchmarks:
        problems.append(f"benchmarks not referenced by proof_matrix: {', '.join(orphaned_benchmarks)}")

    for row in matrix:
        row_id = row.get("id")
        status = row.get("status")
        evidence_ids = row.get("evidence_ids")
        if not isinstance(row_id, str) or not isinstance(status, str):
            continue
        if status == "pending" or not isinstance(evidence_ids, list):
            continue
        evidence_statuses = {
            benchmarks_by_id[evidence_id].get("proof_status")
            for evidence_id in evidence_ids
            if evidence_id in benchmarks_by_id
        }
        if status not in evidence_statuses:
            problems.append(
                f"proof matrix row {row_id} must reference at least one benchmark with proof_status={status}"
            )

    required_scorecard_fields = {
        "id",
        "label",
        "summary",
        "scoring_model",
        "weighted_dimensions",
        "counting_rules",
        "comparison_rules",
        "comparison_scaffold",
        "must_win_gates",
        "must_not_regress_gates",
        "non_weighted_surfaces",
    }
    required_scoring_model_fields = {
        "id",
        "weights_total",
        "outcome_values",
        "formula",
        "claim_allowed_when",
    }
    if not scorecard:
        problems.append("overall_scorecard is required")
        return problems

    missing_scorecard_fields = sorted(required_scorecard_fields - scorecard.keys())
    if missing_scorecard_fields:
        problems.append(
            "overall_scorecard missing fields: " + ", ".join(missing_scorecard_fields)
        )
        return problems

    scoring_model = scorecard.get("scoring_model", {})
    missing_scoring_model_fields = sorted(required_scoring_model_fields - scoring_model.keys())
    if missing_scoring_model_fields:
        problems.append(
            "overall_scorecard.scoring_model missing fields: "
            + ", ".join(missing_scoring_model_fields)
        )

    weighted_dimensions = scorecard.get("weighted_dimensions", [])
    weighted_dimension_ids: set[str] = set()
    if not isinstance(weighted_dimensions, list) or not weighted_dimensions:
        problems.append("overall_scorecard.weighted_dimensions must be a non-empty list")
    else:
        weight_total = 0
        for dimension in weighted_dimensions:
            required_dimension_fields = {
                "id",
                "label",
                "weight",
                "proof_matrix_ids",
                "eligible_statuses",
                "required_benchmark_kinds",
                "must_win",
            }
            missing_dimension_fields = sorted(required_dimension_fields - dimension.keys())
            if missing_dimension_fields:
                problems.append(
                    f"weighted dimension {dimension.get('id', '<missing-id>')} missing fields: "
                    + ", ".join(missing_dimension_fields)
                )
                continue

            dimension_id = dimension["id"]
            if dimension_id in weighted_dimension_ids:
                problems.append(f"duplicate weighted dimension id: {dimension_id}")
            weighted_dimension_ids.add(dimension_id)

            weight = dimension.get("weight")
            if not isinstance(weight, int) or weight <= 0:
                problems.append(f"weighted dimension {dimension_id} must use a positive integer weight")
            else:
                weight_total += weight

            proof_matrix_ids = dimension.get("proof_matrix_ids")
            if not isinstance(proof_matrix_ids, list) or not all(
                isinstance(item, str) for item in proof_matrix_ids
            ):
                problems.append(
                    f"weighted dimension {dimension_id} must use a string list for proof_matrix_ids"
                )
            else:
                for row_id in proof_matrix_ids:
                    if row_id not in matrix_ids:
                        problems.append(
                            f"weighted dimension {dimension_id} references unknown proof-matrix id {row_id}"
                        )

            eligible_statuses = dimension.get("eligible_statuses")
            if not isinstance(eligible_statuses, list) or not all(
                isinstance(item, str) for item in eligible_statuses
            ):
                problems.append(
                    f"weighted dimension {dimension_id} must use a string list for eligible_statuses"
                )
            else:
                for status in eligible_statuses:
                    if status not in allowed_statuses:
                        problems.append(
                            f"weighted dimension {dimension_id} uses unknown eligible status {status}"
                        )

            required_kinds = dimension.get("required_benchmark_kinds")
            if not isinstance(required_kinds, list) or not all(
                isinstance(item, str) for item in required_kinds
            ):
                problems.append(
                    f"weighted dimension {dimension_id} must use a string list for required_benchmark_kinds"
                )

            if not isinstance(dimension.get("must_win"), bool):
                problems.append(f"weighted dimension {dimension_id} must use a boolean must_win")

        if isinstance(scoring_model.get("weights_total"), int) and weight_total != scoring_model["weights_total"]:
            problems.append(
                "overall_scorecard weighted_dimensions sum to "
                f"{weight_total}, expected {scoring_model['weights_total']}"
            )

    for collection_name in ("counting_rules", "comparison_rules"):
        collection = scorecard.get(collection_name, [])
        if not isinstance(collection, list) or not collection:
            problems.append(f"overall_scorecard.{collection_name} must be a non-empty list")
            continue
        seen_ids: set[str] = set()
        for entry in collection:
            if not isinstance(entry, dict):
                problems.append(f"overall_scorecard.{collection_name} entries must be objects")
                continue
            if not isinstance(entry.get("id"), str) or not isinstance(entry.get("rule"), str):
                problems.append(
                    f"overall_scorecard.{collection_name} entries must include string id and rule"
                )
                continue
            if entry["id"] in seen_ids:
                problems.append(f"duplicate {collection_name} id: {entry['id']}")
            seen_ids.add(entry["id"])

    comparison_scaffold = scorecard.get("comparison_scaffold", {})
    required_comparison_scaffold_fields = {
        "minimum_competitors",
        "require_same_competitor_set",
        "competitors",
        "required_scope_fields",
        "outcome_ledger_fields",
        "dimension_records",
    }
    missing_comparison_scaffold_fields = sorted(
        required_comparison_scaffold_fields - comparison_scaffold.keys()
    )
    if missing_comparison_scaffold_fields:
        problems.append(
            "overall_scorecard.comparison_scaffold missing fields: "
            + ", ".join(missing_comparison_scaffold_fields)
        )
    else:
        minimum_competitors = comparison_scaffold.get("minimum_competitors")
        if not isinstance(minimum_competitors, int) or minimum_competitors <= 0:
            problems.append(
                "overall_scorecard.comparison_scaffold.minimum_competitors must be a positive integer"
            )

        if comparison_scaffold.get("require_same_competitor_set") is not True:
            problems.append(
                "overall_scorecard.comparison_scaffold.require_same_competitor_set must be true"
            )

        competitors = comparison_scaffold.get("competitors")
        competitor_ids: set[str] = set()
        if not isinstance(competitors, list):
            problems.append(
                "overall_scorecard.comparison_scaffold.competitors must be a list"
            )
        else:
            for competitor in competitors:
                required_competitor_fields = {
                    "id",
                    "label",
                    "dimension_evidence",
                    "source_refs",
                    "notes",
                }
                if not isinstance(competitor, dict):
                    problems.append(
                        "overall_scorecard.comparison_scaffold.competitors entries must be objects"
                    )
                    continue
                missing_competitor_fields = sorted(
                    required_competitor_fields - competitor.keys()
                )
                if missing_competitor_fields:
                    problems.append(
                        "overall_scorecard.comparison_scaffold.competitors entries missing fields: "
                        + ", ".join(missing_competitor_fields)
                    )
                    continue
                if not isinstance(competitor.get("id"), str) or not isinstance(
                    competitor.get("label"), str
                ):
                    problems.append(
                        "overall_scorecard.comparison_scaffold.competitors entries must include string id and label"
                    )
                    continue
                competitor_id = competitor["id"]
                if competitor_id in competitor_ids:
                    problems.append(
                        f"duplicate comparison_scaffold competitor id: {competitor_id}"
                    )
                competitor_ids.add(competitor_id)
                if not isinstance(competitor.get("notes"), str):
                    problems.append(
                        f"comparison_scaffold competitor {competitor_id} must use a string notes field"
                    )
                source_refs = competitor.get("source_refs")
                if not isinstance(source_refs, list) or not source_refs or not all(
                    isinstance(item, str) for item in source_refs
                ):
                    problems.append(
                        f"comparison_scaffold competitor {competitor_id} must use a non-empty string list for source_refs"
                    )
                dimension_evidence = competitor.get("dimension_evidence")
                if not isinstance(dimension_evidence, dict):
                    problems.append(
                        f"comparison_scaffold competitor {competitor_id} must use an object for dimension_evidence"
                    )
                else:
                    missing_dimension_evidence = sorted(
                        weighted_dimension_ids - set(dimension_evidence.keys())
                    )
                    if missing_dimension_evidence:
                        problems.append(
                            f"comparison_scaffold competitor {competitor_id} missing dimension_evidence for: "
                            + ", ".join(missing_dimension_evidence)
                        )
                    for dimension_id, evidence_state in dimension_evidence.items():
                        if (
                            weighted_dimension_ids
                            and dimension_id not in weighted_dimension_ids
                        ):
                            problems.append(
                                f"comparison_scaffold competitor {competitor_id} references unknown weighted dimension {dimension_id} in dimension_evidence"
                            )
                            continue
                        if evidence_state not in ALLOWED_COMPETITOR_EVIDENCE_STATES:
                            problems.append(
                                f"comparison_scaffold competitor {competitor_id} uses unknown evidence state {evidence_state!r} for {dimension_id}"
                            )

        required_scope_fields = comparison_scaffold.get("required_scope_fields")
        if not isinstance(required_scope_fields, list) or not required_scope_fields:
            problems.append(
                "overall_scorecard.comparison_scaffold.required_scope_fields must be a non-empty list"
            )
            required_scope_fields = []
        elif not all(isinstance(field, str) for field in required_scope_fields):
            problems.append(
                "overall_scorecard.comparison_scaffold.required_scope_fields must contain only strings"
            )
            required_scope_fields = [
                field for field in required_scope_fields if isinstance(field, str)
            ]

        outcome_ledger_fields = comparison_scaffold.get("outcome_ledger_fields")
        if not isinstance(outcome_ledger_fields, list) or not outcome_ledger_fields:
            problems.append(
                "overall_scorecard.comparison_scaffold.outcome_ledger_fields must be a non-empty list"
            )
        else:
            seen_outcome_field_ids: set[str] = set()
            for field in outcome_ledger_fields:
                required_outcome_field_keys = {"id", "required", "meaning"}
                if not isinstance(field, dict):
                    problems.append(
                        "overall_scorecard.comparison_scaffold.outcome_ledger_fields entries must be objects"
                    )
                    continue
                missing_outcome_field_keys = sorted(
                    required_outcome_field_keys - field.keys()
                )
                if missing_outcome_field_keys:
                    problems.append(
                        "comparison_scaffold outcome_ledger_fields entry missing fields: "
                        + ", ".join(missing_outcome_field_keys)
                    )
                    continue
                field_id = field.get("id")
                if not isinstance(field_id, str):
                    problems.append(
                        "comparison_scaffold outcome_ledger_fields entries must use a string id"
                    )
                    continue
                if field_id in seen_outcome_field_ids:
                    problems.append(
                        f"duplicate comparison_scaffold outcome_ledger_fields id: {field_id}"
                    )
                seen_outcome_field_ids.add(field_id)
                if not isinstance(field.get("required"), bool):
                    problems.append(
                        f"comparison_scaffold outcome_ledger_fields {field_id} must use a boolean required"
                    )
                if not isinstance(field.get("meaning"), str):
                    problems.append(
                        f"comparison_scaffold outcome_ledger_fields {field_id} must use a string meaning"
                    )

        dimension_records = comparison_scaffold.get("dimension_records")
        if not isinstance(dimension_records, list) or not dimension_records:
            problems.append(
                "overall_scorecard.comparison_scaffold.dimension_records must be a non-empty list"
            )
        else:
            seen_dimension_record_ids: set[str] = set()
            allowed_outcomes = set(scoring_model.get("outcome_values", {}).keys())
            for record in dimension_records:
                required_record_fields = {"dimension_id", "scope", "recorded_outcomes"}
                if not isinstance(record, dict):
                    problems.append(
                        "overall_scorecard.comparison_scaffold.dimension_records entries must be objects"
                    )
                    continue
                missing_record_fields = sorted(required_record_fields - record.keys())
                if missing_record_fields:
                    problems.append(
                        "comparison_scaffold dimension record "
                        f"{record.get('dimension_id', '<missing-dimension-id>')} missing fields: "
                        + ", ".join(missing_record_fields)
                    )
                    continue

                dimension_id = record["dimension_id"]
                if not isinstance(dimension_id, str):
                    problems.append(
                        "comparison_scaffold dimension records must use a string dimension_id"
                    )
                    continue
                if dimension_id in seen_dimension_record_ids:
                    problems.append(
                        f"duplicate comparison_scaffold dimension record id: {dimension_id}"
                    )
                seen_dimension_record_ids.add(dimension_id)
                if weighted_dimension_ids and dimension_id not in weighted_dimension_ids:
                    problems.append(
                        "comparison_scaffold dimension record references unknown weighted dimension "
                        f"{dimension_id}"
                    )

                scope = record.get("scope")
                if not isinstance(scope, dict):
                    problems.append(
                        f"comparison_scaffold dimension record {dimension_id} must use an object for scope"
                    )
                else:
                    for field in required_scope_fields:
                        if field not in scope:
                            problems.append(
                                f"comparison_scaffold dimension record {dimension_id} missing scope field {field}"
                            )
                            continue
                        if scope[field] is not None and not isinstance(scope[field], str):
                            problems.append(
                                f"comparison_scaffold dimension record {dimension_id} scope field {field} must be a string or null"
                            )

                recorded_outcomes = record.get("recorded_outcomes")
                if not isinstance(recorded_outcomes, list):
                    problems.append(
                        f"comparison_scaffold dimension record {dimension_id} must use a list for recorded_outcomes"
                    )
                    continue
                if not competitor_ids and recorded_outcomes:
                    problems.append(
                        f"comparison_scaffold dimension record {dimension_id} cannot record outcomes before competitors are named"
                    )

                seen_recorded_competitors: set[str] = set()
                for outcome in recorded_outcomes:
                    required_outcome_fields = {
                        "competitor_id",
                        "outcome",
                        "summary",
                        "source_refs",
                    }
                    if not isinstance(outcome, dict):
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} recorded_outcomes entries must be objects"
                        )
                        continue
                    missing_outcome_fields = sorted(required_outcome_fields - outcome.keys())
                    if missing_outcome_fields:
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} recorded outcome missing fields: "
                            + ", ".join(missing_outcome_fields)
                        )
                        continue

                    competitor_id = outcome.get("competitor_id")
                    if not isinstance(competitor_id, str):
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} recorded outcomes must use string competitor_id"
                        )
                        continue
                    if competitor_id in seen_recorded_competitors:
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} has duplicate outcome for competitor {competitor_id}"
                        )
                    seen_recorded_competitors.add(competitor_id)
                    if competitor_ids and competitor_id not in competitor_ids:
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} references unknown competitor {competitor_id}"
                        )

                    if outcome.get("outcome") not in allowed_outcomes:
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} uses unknown outcome {outcome.get('outcome')}"
                        )
                    if not isinstance(outcome.get("summary"), str):
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} recorded outcomes must use a string summary"
                        )
                    source_refs = outcome.get("source_refs")
                    if not isinstance(source_refs, list) or not source_refs or not all(
                        isinstance(item, str) for item in source_refs
                    ):
                        problems.append(
                            f"comparison_scaffold dimension record {dimension_id} recorded outcomes must use a non-empty string list for source_refs"
                        )

            if weighted_dimension_ids:
                missing_dimension_records = sorted(
                    weighted_dimension_ids - seen_dimension_record_ids
                )
                if missing_dimension_records:
                    problems.append(
                        "comparison_scaffold is missing dimension records for: "
                        + ", ".join(missing_dimension_records)
                    )

    for gate in scorecard.get("must_win_gates", []):
        required_gate_fields = {"id", "label", "dimension_ids", "rule"}
        missing_gate_fields = sorted(required_gate_fields - gate.keys())
        if missing_gate_fields:
            problems.append(
                f"must_win gate {gate.get('id', '<missing-id>')} missing fields: "
                + ", ".join(missing_gate_fields)
            )
            continue
        dimension_ids = gate.get("dimension_ids")
        if not isinstance(dimension_ids, list) or not all(
            isinstance(item, str) for item in dimension_ids
        ):
            problems.append(
                f"must_win gate {gate['id']} must use a string list for dimension_ids"
            )
            continue
        for dimension_id in dimension_ids:
            if dimension_id not in weighted_dimension_ids:
                problems.append(
                    f"must_win gate {gate['id']} references unknown weighted dimension {dimension_id}"
                )

    for gate in scorecard.get("must_not_regress_gates", []):
        required_gate_fields = {"id", "label", "proof_matrix_ids", "required_statuses", "rule"}
        missing_gate_fields = sorted(required_gate_fields - gate.keys())
        if missing_gate_fields:
            problems.append(
                f"must_not_regress gate {gate.get('id', '<missing-id>')} missing fields: "
                + ", ".join(missing_gate_fields)
            )
            continue
        proof_matrix_ids = gate.get("proof_matrix_ids")
        if not isinstance(proof_matrix_ids, list) or not all(
            isinstance(item, str) for item in proof_matrix_ids
        ):
            problems.append(
                f"must_not_regress gate {gate['id']} must use a string list for proof_matrix_ids"
            )
        else:
            for row_id in proof_matrix_ids:
                if row_id not in matrix_ids:
                    problems.append(
                        f"must_not_regress gate {gate['id']} references unknown proof-matrix id {row_id}"
                    )

        required_statuses = gate.get("required_statuses")
        if not isinstance(required_statuses, list) or not all(
            isinstance(item, str) for item in required_statuses
        ):
            problems.append(
                f"must_not_regress gate {gate['id']} must use a string list for required_statuses"
            )
        else:
            for status in required_statuses:
                if status not in allowed_statuses:
                    problems.append(
                        f"must_not_regress gate {gate['id']} uses unknown required status {status}"
                    )

    non_weighted_surfaces = scorecard.get("non_weighted_surfaces", [])
    if not isinstance(non_weighted_surfaces, list) or not non_weighted_surfaces:
        problems.append("overall_scorecard.non_weighted_surfaces must be a non-empty list")
    else:
        allowed_roles = {"must-not-regress", "support-only"}
        seen_ids: set[str] = set()
        for surface in non_weighted_surfaces:
            required_surface_fields = {"id", "label", "proof_matrix_ids", "role", "rule"}
            missing_surface_fields = sorted(required_surface_fields - surface.keys())
            if missing_surface_fields:
                problems.append(
                    f"non_weighted surface {surface.get('id', '<missing-id>')} missing fields: "
                    + ", ".join(missing_surface_fields)
                )
                continue
            if surface["id"] in seen_ids:
                problems.append(f"duplicate non_weighted surface id: {surface['id']}")
            seen_ids.add(surface["id"])
            proof_matrix_ids = surface.get("proof_matrix_ids")
            if not isinstance(proof_matrix_ids, list) or not all(
                isinstance(item, str) for item in proof_matrix_ids
            ):
                problems.append(
                    f"non_weighted surface {surface['id']} must use a string list for proof_matrix_ids"
                )
            else:
                for row_id in proof_matrix_ids:
                    if row_id not in matrix_ids:
                        problems.append(
                            f"non_weighted surface {surface['id']} references unknown proof-matrix id {row_id}"
                        )
            if surface.get("role") not in allowed_roles:
                problems.append(
                    f"non_weighted surface {surface['id']} uses unknown role {surface.get('role')}"
                )

    if not isinstance(guardrail_suites, list) or not guardrail_suites:
        problems.append("guardrail_suites must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        benchmarks_by_id = index_by_id(benchmarks)
        for suite in guardrail_suites:
            required_suite_fields = {
                "id",
                "label",
                "summary",
                "proof_matrix_ids",
                "benchmark_ids",
                "ci_ready",
                "notes",
            }
            missing_suite_fields = sorted(required_suite_fields - suite.keys())
            if missing_suite_fields:
                problems.append(
                    f"guardrail suite {suite.get('id', '<missing-id>')} missing fields: "
                    + ", ".join(missing_suite_fields)
                )
                continue
            if suite["id"] in seen_ids:
                problems.append(f"duplicate guardrail suite id: {suite['id']}")
            seen_ids.add(suite["id"])
            if not isinstance(suite.get("label"), str) or not isinstance(suite.get("summary"), str):
                problems.append(
                    f"guardrail suite {suite['id']} must include string label and summary"
                )
            if not isinstance(suite.get("notes"), str):
                problems.append(f"guardrail suite {suite['id']} must include string notes")
            if not isinstance(suite.get("ci_ready"), bool):
                problems.append(f"guardrail suite {suite['id']} must use a boolean ci_ready")

            proof_matrix_ids = suite.get("proof_matrix_ids")
            if not isinstance(proof_matrix_ids, list) or not all(
                isinstance(item, str) for item in proof_matrix_ids
            ):
                problems.append(
                    f"guardrail suite {suite['id']} must use a string list for proof_matrix_ids"
                )
            else:
                for row_id in proof_matrix_ids:
                    if row_id not in matrix_ids:
                        problems.append(
                            f"guardrail suite {suite['id']} references unknown proof-matrix id {row_id}"
                        )

            benchmark_ids = suite.get("benchmark_ids")
            if not isinstance(benchmark_ids, list) or not all(
                isinstance(item, str) for item in benchmark_ids
            ):
                problems.append(
                    f"guardrail suite {suite['id']} must use a string list for benchmark_ids"
                )
                continue

            covered_dimensions = {
                benchmarks_by_id[bench_id]["dimension"]
                for bench_id in benchmark_ids
                if bench_id in benchmarks_by_id
            }
            for bench_id in benchmark_ids:
                if bench_id not in benchmarks_by_id:
                    problems.append(
                        f"guardrail suite {suite['id']} references unknown benchmark id {bench_id}"
                    )
            if isinstance(proof_matrix_ids, list):
                for row_id in proof_matrix_ids:
                    if row_id in matrix_ids and row_id not in covered_dimensions:
                        problems.append(
                            f"guardrail suite {suite['id']} has no benchmark coverage for proof-matrix id {row_id}"
                        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List benchmark ids")
    list_parser.add_argument("--official", action="store_true", help="Show only official entries")
    list_parser.add_argument("--kind", help="Filter to one benchmark kind")
    list_parser.add_argument("--surface", help="Filter to one benchmark surface")
    list_parser.add_argument("--dimension", help="Filter to one proof-matrix dimension")
    list_parser.add_argument("--proof-status", help="Filter to one proof status")
    list_parser.add_argument("--json", action="store_true", help="Emit matching entries as JSON")

    matrix_parser = sub.add_parser("matrix", help="Show proof-matrix summary rows")
    matrix_parser.add_argument("--proof-status", help="Filter to one proof status")
    matrix_parser.add_argument("--json", action="store_true", help="Emit matching rows as JSON")

    scorecard_parser = sub.add_parser("scorecard", help="Show the overall weighted claim gate")
    scorecard_parser.add_argument("--json", action="store_true", help="Emit scorecard evaluation as JSON")

    guardrails_parser = sub.add_parser("guardrails", help="List or run named guardrail suites")
    guardrails_parser.add_argument("suite_ids", nargs="*", help="Guardrail suite ids")
    guardrails_parser.add_argument("--json", action="store_true", help="Emit matching suites as JSON")
    guardrails_parser.add_argument("--run", action="store_true", help="Run the selected guardrail suite(s)")

    show_parser = sub.add_parser("show", help="Show one benchmark, proof-matrix, or scorecard entry")
    show_parser.add_argument("entry_id")

    run_parser = sub.add_parser("run", help="Run one or more benchmarks")
    run_parser.add_argument("benchmark_ids", nargs="*", help="Benchmark ids to run")
    run_parser.add_argument("--official", action="store_true", help="Run all official entries")
    run_parser.add_argument("--kind", help="Run all entries of one benchmark kind")
    run_parser.add_argument("--surface", help="Run all entries on one benchmark surface")
    run_parser.add_argument("--dimension", help="Run all entries in one proof-matrix dimension")
    run_parser.add_argument("--proof-status", help="Run all entries with one proof status")

    sub.add_parser("validate", help="Validate registry structure and proof-matrix links")

    args = parser.parse_args()
    registry = load_registry()
    benchmarks = load_benchmarks(registry)
    matrix = load_matrix(registry)
    guardrail_suites = load_guardrail_suites(registry)

    if args.command == "list":
        selected = filter_benchmarks(
            benchmarks,
            official=args.official,
            kind=args.kind,
            surface=args.surface,
            dimension=args.dimension,
            proof_status=args.proof_status,
        )
        if args.json:
            print(json.dumps(selected, indent=2))
            return 0
        if not selected:
            print("No benchmarks matched.")
            return 0
        print_benchmark_table(selected)
        return 0

    if args.command == "matrix":
        selected = filter_matrix(matrix, proof_status=args.proof_status)
        if args.json:
            print(json.dumps(selected, indent=2))
            return 0
        if not selected:
            print("No proof-matrix rows matched.")
            return 0
        print_matrix_table(selected)
        return 0

    if args.command == "scorecard":
        evaluation = evaluate_scorecard(registry)
        if args.json:
            print(json.dumps(evaluation, indent=2))
            return 0
        print_scorecard_table(evaluation)
        return 0

    if args.command == "guardrails":
        selected = (
            [find_guardrail_suite(guardrail_suites, suite_id) for suite_id in args.suite_ids]
            if args.suite_ids
            else guardrail_suites
        )
        if args.run:
            if args.json:
                raise SystemExit("--json cannot be combined with --run for guardrails")
            if not selected:
                raise SystemExit("No guardrail suites matched the requested ids.")
            for suite in selected:
                code = run_guardrail_suite(suite, benchmarks)
                if code != 0:
                    return code
            return 0
        if args.json:
            print(json.dumps(selected, indent=2))
            return 0
        if not selected:
            print("No guardrail suites matched.")
            return 0
        print_guardrail_table(selected)
        return 0

    if args.command == "show":
        scorecard = load_overall_scorecard(registry)
        if scorecard.get("id") == args.entry_id:
            print(json.dumps(evaluate_scorecard(registry), indent=2))
            return 0
        for suite in guardrail_suites:
            if suite["id"] == args.entry_id:
                print(json.dumps(suite, indent=2))
                return 0
        for row in matrix:
            if row["id"] == args.entry_id:
                print(json.dumps(row, indent=2))
                return 0
        bench = find_benchmark(benchmarks, args.entry_id)
        print(json.dumps(bench, indent=2))
        return 0

    if args.command == "validate":
        problems = validate_registry(registry)
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(
            f"Validated {len(benchmarks)} benchmark entries, {len(matrix)} proof-matrix rows, "
            f"and {len(guardrail_suites)} guardrail suites."
        )
        return 0

    selected: list[dict[str, Any]]
    if args.benchmark_ids:
        selected = [find_benchmark(benchmarks, bench_id) for bench_id in args.benchmark_ids]
        selected = filter_benchmarks(
            selected,
            official=args.official,
            kind=args.kind,
            surface=args.surface,
            dimension=args.dimension,
            proof_status=args.proof_status,
        )
    else:
        if not any((args.official, args.kind, args.surface, args.dimension, args.proof_status)):
            raise SystemExit(
                "Provide benchmark ids or pass --official/--kind/--surface/--dimension/--proof-status."
            )
        selected = filter_benchmarks(
            benchmarks,
            official=args.official,
            kind=args.kind,
            surface=args.surface,
            dimension=args.dimension,
            proof_status=args.proof_status,
        )

    if not selected:
        raise SystemExit("No benchmarks matched the requested filters.")

    for bench in selected:
        code = run_benchmark(bench)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
