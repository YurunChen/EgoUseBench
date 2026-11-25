"""Canonical construct-validity and selective-review contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .synthesis_reports import REPORTS

CONFIDENCE_POLICY_VERSION = "coding-agent-selective-review-v3"

ADMISSION_GATES = (
    "utility",
    "effectiveness",
    "continuous_change",
    "verifiability",
)
EVIDENCE_DIMENSIONS = (
    "visual_evidence",
    "causal_contract",
    "information_partition",
    "site_mapping",
    "answer_boundary",
    "user_instruction",
)
EVIDENCE_BY_ADMISSION_GATE = {
    "utility": ("user_instruction",),
    "effectiveness": ("visual_evidence", "site_mapping", "answer_boundary"),
    "continuous_change": ("causal_contract",),
    "verifiability": ("information_partition", "answer_boundary"),
}
COUNTEREXAMPLE_CHECKS = (
    "no_scene",
    "site_only",
    "static_substitute",
    "early_commit",
    "tempting_choice",
    "wrong_state",
    "undeclared_equivalent",
    "continuous_interval",
)
COUNTEREXAMPLE_GUIDANCE = {
    "no_scene": (
        "Remove only scene evidence; keep the complete instruction, user updates, clarification "
        "answer, and public website. Fail if this remaining evidence guarantees completion "
        "of at least one legal complete final state, including all required selections, "
        "quantities, and fields. Recovering every accepted state is not necessary for a "
        "visual shortcut. One legal entity alone is not enough when the complete state "
        "requires more. A guessed accepted answer or a unique title-keyword hit is not "
        "proof of that guarantee. Otherwise identify the concrete unresolved requirements "
        "and plausible alternatives; inspect answer-set completeness separately."
    ),
    "site_only": "Use only public site data; the final target must remain unresolved.",
    "static_substitute": (
        "Test the track's decision-changing source, not whether any visual anchor is static. "
        "For Clarify, remove the clarification answer; for Graft, remove the later user "
        "update and compare the accepted state sets. For Graft-relax, removing B must remove "
        "newly legal states, while every originally legal state remains accepted after B. "
        "Keeping an originally legal answer is valid and does not disprove the update's "
        "effect; check the expanded legal set rather than demand a changed model choice. "
        "A stable visual anchor is allowed when the update changes the accepted state set. "
        "For Refresh-replace, test whether removing later evidence eliminates the "
        "reason to revise. For Refresh-retain, V0 may already identify a valid target and "
        "removing V1 need not change the answer: test whether a real later distractor tempts "
        "an unjustified switch while the fixed request still binds the original target. "
        "Do not require retain to change its answer, or count a redundant image as a real "
        "retention challenge."
    ),
    "early_commit": (
        "Judge the specific action against information sufficient at that time, not merely "
        "whether an update has arrived. Graft allows valid A-side commitments before B, "
        "even if B later retracts them; distinguish those from submitting an incomplete "
        "final result. Refresh-retain allows a V0-grounded valid target before V1 and must "
        "not label it premature solely for lacking V1. For Clarify and Refresh-replace, "
        "test the unresolved referent and registered sufficiency boundary. Separate "
        "candidate commitment, final submission, stale state, and false update; report "
        "what the actual Timeline, runtime, and checker enforce instead of demanding "
        "blanket rejection of every early action."
    ),
    "tempting_choice": (
        "Test an actual plausible distractor or tempting wrong state; the checker must "
        "reject it. Do not invent neighbors to meet a candidate count. If no such candidate "
        "or state exists, record task-specific applicability evidence in answer_boundary, "
        "including the inspected candidate scope and why no tempting choice applies. "
        "A bare default pass is not evidence; use uncertain when applicability cannot be "
        "established. A state accepted after Relax is not a distractor merely because it "
        "was rejected before the update."
    ),
    "wrong_state": "Try missing, extra, stale, duplicate, or otherwise wrong final state.",
    "undeclared_equivalent": (
        "Inspect candidates reachable through allowed search and detail operations in the "
        "frozen website, not only the model's observed results, for an entity satisfying every "
        "released scene, update, and website predicate. Every such entity must already be "
        "accepted; search rank and constructor preference are not rejection evidence."
    ),
    "continuous_interval": (
        "Inspect the source evidence needed for the deciding claim. Readable released frames "
        "can establish static objects and attributes. For action, method, order or change, "
        "inspect the relevant interval with enough temporal and spatial detail to distinguish "
        "the material alternatives; account for the whole interval when the claim requires "
        "excluding competing events. Use contact sheets to locate deciding views, then inspect "
        "those views or native playback. Reuse adequate recorded inspection and state its actual "
        "method, times and limits; do not require or claim every decoded frame by default."
    ),
}
PASS_NAMES = ("blind_reconstruction", "adversarial_falsification")
STATUS_VALUES = frozenset({"pass", "fail", "uncertain"})
DECISIONS = ("admit", "repair")
CONFIDENCE_VALUES = ("high", "low")
FOLLOW_UP_ACTIONS = ("repair", "collect_evidence", "human_review")

POLICY: dict[str, Any] = {
    "schema_version": 1,
    "policy_version": CONFIDENCE_POLICY_VERSION,
    "assessment": {
        "report": REPORTS["assessment"].path.as_posix(),
        "prompt_version": "task-confidence-blind-reconstruction-v4",
        "assessor": "skill_executing_coding_agent",
        "pass_names": list(PASS_NAMES),
        "required_assessments_per_task": len(PASS_NAMES),
        "default_reviewers_per_task": 1,
        "verdict_values": list(DECISIONS),
        "confidence_values": list(CONFIDENCE_VALUES),
    },
    "routing": {
        "high_admit": "automatic_construction_admission",
        "high_repair": "repair",
        "low_confidence": "human_review_pending",
        "assessment_error": "coding_agent_retry_pending",
    },
}


class ConstructValidityError(ValueError):
    """Raised when a construct-validity judgement violates the contract."""


def _merge_statuses(values: list[str]) -> str:
    if "fail" in values:
        return "fail"
    if "uncertain" in values:
        return "uncertain"
    return "pass"


def admission_gate_statuses(
    evidence_statuses: Mapping[str, Any],
    *,
    counterexample_statuses: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Derive the four admission gates from recorded semantic evidence."""

    if set(evidence_statuses) != set(EVIDENCE_DIMENSIONS) or any(
        value not in STATUS_VALUES for value in evidence_statuses.values()
    ):
        raise ConstructValidityError("semantic evidence statuses are malformed")
    result = {
        gate: _merge_statuses(
            [
                str(evidence_statuses[dimension])
                for dimension in EVIDENCE_BY_ADMISSION_GATE[gate]
            ]
        )
        for gate in ADMISSION_GATES
    }
    if counterexample_statuses is not None:
        if set(counterexample_statuses) != set(COUNTEREXAMPLE_CHECKS) or any(
            value not in STATUS_VALUES for value in counterexample_statuses.values()
        ):
            raise ConstructValidityError("counterexample statuses are malformed")
        result["verifiability"] = _merge_statuses(
            [
                result["verifiability"],
                *(str(value) for value in counterexample_statuses.values()),
            ]
        )
    return result


def validate_assessment_verdict(
    verdict: Mapping[str, Any],
    *,
    pass_name: str,
    blind_verdict: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate one Coding-Agent verdict and return its four admission gates."""

    if pass_name not in PASS_NAMES:
        raise ConstructValidityError(f"unsupported assessment pass {pass_name!r}")
    if verdict.get("decision") not in DECISIONS:
        raise ConstructValidityError("assessment decision is malformed")
    if verdict.get("confidence") not in CONFIDENCE_VALUES:
        raise ConstructValidityError("assessment confidence is malformed")

    evidence_statuses = verdict.get("gates")
    if not isinstance(evidence_statuses, Mapping):
        raise ConstructValidityError(
            "assessment semantic evidence statuses are missing"
        )
    _validate_evidence_references(
        verdict,
        blind_verdict if pass_name == PASS_NAMES[1] else None,
    )
    reviewer = verdict.get("reviewer")
    if reviewer is not None and (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"agent_id", "context_id"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in reviewer.values()
        )
    ):
        raise ConstructValidityError("assessment reviewer identity is malformed")

    reasons = verdict.get("uncertainty_reasons")
    issues = verdict.get("issues", [])
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        raise ConstructValidityError("assessment uncertainty reasons are malformed")
    if not isinstance(issues, list) or any(
        not isinstance(issue, str) or not issue.strip() for issue in issues
    ):
        raise ConstructValidityError("assessment issues are malformed")

    checks = verdict.get("counterexample_checks")
    if pass_name == PASS_NAMES[1]:
        if "follow_up" in verdict:
            _validate_follow_up(verdict)
        if "human_review_reuse" in verdict:
            _validate_human_review_reuse(verdict)
        if "blind_resolutions" in verdict:
            validate_blind_resolutions(
                verdict["blind_resolutions"], blind_verdict, verdict
            )
        if not isinstance(checks, Mapping):
            raise ConstructValidityError(
                "adversarial counterexample statuses are missing"
            )
        return admission_gate_statuses(
            evidence_statuses,
            counterexample_statuses=checks,
        )
    if checks is not None:
        raise ConstructValidityError(
            "blind reconstruction cannot contain counterexample results"
        )
    if "blind_resolutions" in verdict:
        raise ConstructValidityError(
            "blind reconstruction cannot resolve its own concerns"
        )
    if "follow_up" in verdict:
        raise ConstructValidityError("follow-up decisions belong to the second pass")
    if "human_review_reuse" in verdict:
        raise ConstructValidityError("human review reuse belongs to the second pass")
    if verdict.get("decision") == "admit" and not isinstance(
        verdict.get("derived_final_state"), Mapping
    ):
        raise ConstructValidityError("blind admission needs a derived final state")
    return admission_gate_statuses(evidence_statuses)


def blind_concerns(verdict: Mapping[str, Any]) -> dict[str, str]:
    """Address the sealed concerns without changing their original wording."""
    return {
        **{
            f"gates.{dimension}": status
            for dimension, status in verdict["gates"].items()
            if status != "pass"
        },
        **{
            f"{field}.{index}": text
            for field in ("uncertainty_reasons", "issues")
            for index, text in enumerate(verdict.get(field, []))
        },
    }


def _validate_current_evidence(references: Any, verdict: Mapping[str, Any]) -> None:
    if not isinstance(references, list) or not references:
        raise ConstructValidityError(
            "review decision needs current evidence references"
        )
    for reference in references:
        parts = reference.split(".") if isinstance(reference, str) else []
        if (
            len(parts) != 3
            or parts[0] != "evidence"
            or parts[1] not in EVIDENCE_DIMENSIONS
            or not parts[2].isdecimal()
        ):
            raise ConstructValidityError(
                "review decision evidence reference is malformed"
            )
        entries = verdict["evidence"][parts[1]]
        index = int(parts[2])
        if (
            index >= len(entries)
            or not isinstance(entries[index], str)
            or not entries[index].strip()
        ):
            raise ConstructValidityError(
                "review decision must cite concrete current evidence"
            )


def _validate_follow_up(verdict: Mapping[str, Any]) -> None:
    follow_up = verdict["follow_up"]
    if not isinstance(verdict.get("reviewer"), Mapping):
        raise ConstructValidityError("follow-up decisions require the actual reviewer")
    if not isinstance(follow_up, list) or not follow_up:
        raise ConstructValidityError("follow-up decisions must be a nonempty list")
    for item in follow_up:
        if not isinstance(item, Mapping) or set(item) != {
            "action",
            "evidence",
            "reason",
        }:
            raise ConstructValidityError("follow-up decision fields are malformed")
        if item["action"] not in FOLLOW_UP_ACTIONS:
            raise ConstructValidityError("follow-up action is invalid")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ConstructValidityError("follow-up decision needs a concrete reason")
        _validate_current_evidence(item["evidence"], verdict)


def _validate_human_review_reuse(verdict: Mapping[str, Any]) -> None:
    reuse = verdict["human_review_reuse"]
    if not isinstance(verdict.get("reviewer"), Mapping):
        raise ConstructValidityError("human review reuse requires the actual reviewer")
    if not isinstance(reuse, list) or not reuse:
        raise ConstructValidityError("human review reuse must be a nonempty list")
    submissions: set[tuple[str, str]] = set()
    for item in reuse:
        if not isinstance(item, Mapping) or set(item) != {
            "submission",
            "outcome",
            "retained_conclusions",
            "affected_conclusions",
            "evidence",
        }:
            raise ConstructValidityError("human review reuse fields are malformed")
        submission = item["submission"]
        if (
            not isinstance(submission, Mapping)
            or set(submission) != {"path", "submitted_at_utc"}
            or any(
                not isinstance(value, str) or not value.strip()
                for value in submission.values()
            )
        ):
            raise ConstructValidityError("human review source is missing or duplicated")
        identity = (submission["path"], submission["submitted_at_utc"])
        if identity in submissions:
            raise ConstructValidityError("human review source is missing or duplicated")
        submissions.add(identity)
        if not isinstance(item["outcome"], str) or item["outcome"] not in {
            "retained",
            "partial",
            "repair_verified",
        }:
            raise ConstructValidityError("human review reuse outcome is invalid")
        for field in ("retained_conclusions", "affected_conclusions"):
            values = item[field]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ConstructValidityError(
                    "human review reuse needs concrete conclusions"
                )
        if not item["retained_conclusions"]:
            raise ConstructValidityError(
                "human review reuse must identify retained conclusions"
            )
        _validate_current_evidence(item["evidence"], verdict)
        if item["outcome"] == "partial":
            if not item["affected_conclusions"]:
                raise ConstructValidityError(
                    "partial human review reuse needs the affected scope"
                )
            continue
        if (
            item["affected_conclusions"]
            or verdict.get("issues")
            or verdict["uncertainty_reasons"]
            or verdict.get("follow_up")
            or not any(
                reference.startswith("evidence.answer_boundary.")
                for reference in item["evidence"]
            )
            or any(status != "pass" for status in verdict["gates"].values())
            or not isinstance(verdict.get("counterexample_checks"), Mapping)
            or any(
                status != "pass" for status in verdict["counterexample_checks"].values()
            )
        ):
            raise ConstructValidityError(
                "unresolved current scope cannot inherit a whole human decision"
            )


def validate_blind_resolutions(
    resolutions: Any,
    blind: Mapping[str, Any] | None,
    adversarial: Mapping[str, Any],
) -> set[str]:
    """Check traceable review records; their semantic support remains reviewer judgement."""
    if blind is None or not isinstance(resolutions, list) or not resolutions:
        raise ConstructValidityError(
            "blind resolutions require sealed concerns and a nonempty list"
        )
    if not isinstance(adversarial.get("reviewer"), Mapping):
        raise ConstructValidityError(
            "blind resolutions require the actual reviewer identity"
        )
    concerns = blind_concerns(blind)
    resolved: set[str] = set()
    for resolution in resolutions:
        if not isinstance(resolution, Mapping) or set(resolution) != {
            "concern",
            "original",
            "evidence",
            "reason",
        }:
            raise ConstructValidityError("blind resolution fields are malformed")
        concern = resolution["concern"]
        if (
            not isinstance(concern, str)
            or concern not in concerns
            or concern in resolved
        ):
            raise ConstructValidityError(
                "blind resolution concern is missing or duplicated"
            )
        if resolution["original"] != concerns[concern]:
            raise ConstructValidityError(
                "blind resolution does not quote the sealed concern"
            )
        if (
            not isinstance(resolution["reason"], str)
            or not resolution["reason"].strip()
        ):
            raise ConstructValidityError(
                "blind resolution needs a concrete explanation"
            )
        references = resolution["evidence"]
        if not isinstance(references, list) or not references:
            raise ConstructValidityError(
                "blind resolution needs current evidence references"
            )
        for reference in references:
            parts = reference.split(".") if isinstance(reference, str) else []
            if len(parts) != 3 or parts[0] != "evidence" or not parts[2].isdecimal():
                raise ConstructValidityError(
                    "blind resolution evidence reference is malformed"
                )
            entries = adversarial.get("evidence", {}).get(parts[1], [])
            index = int(parts[2])
            if (
                index >= len(entries)
                or not isinstance(entries[index], str)
                or not entries[index].strip()
            ):
                raise ConstructValidityError(
                    "blind resolution must cite concrete second-pass evidence"
                )
        resolved.add(concern)
    return resolved


def _validate_evidence_references(
    verdict: Mapping[str, Any], blind_verdict: Mapping[str, Any] | None
) -> None:
    """Resolve short references without copying or rewriting archived evidence."""

    evidence = verdict.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(EVIDENCE_DIMENSIONS):
        raise ConstructValidityError("assessment evidence is not task-specific")
    scopes = {"": evidence}
    if blind_verdict is not None:
        blind_evidence = blind_verdict.get("evidence")
        if not isinstance(blind_evidence, Mapping):
            raise ConstructValidityError("sealed blind evidence is missing")
        scopes["blind_reconstruction."] = blind_evidence

    def visit(scope: str, dimension: str, active: frozenset[str]) -> None:
        key = f"{scope}evidence.{dimension}"
        if key in active:
            raise ConstructValidityError(f"assessment evidence reference cycle: {key}")
        entries = scopes.get(scope, {}).get(dimension)
        if not isinstance(entries, list) or not entries:
            raise ConstructValidityError(
                f"assessment evidence reference is missing: {key}"
            )
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                continue
            if not isinstance(entry, Mapping) or set(entry) != {"ref"}:
                raise ConstructValidityError("assessment evidence is not task-specific")
            reference = entry["ref"]
            if not isinstance(reference, str):
                raise ConstructValidityError(
                    "assessment evidence reference is malformed"
                )
            target_scope = scope
            if reference.startswith("blind_reconstruction."):
                target_scope = "blind_reconstruction."
                reference = reference.removeprefix(target_scope)
            if not reference.startswith("evidence."):
                raise ConstructValidityError(
                    "assessment evidence reference is malformed"
                )
            target = reference.removeprefix("evidence.")
            if target not in EVIDENCE_DIMENSIONS or target_scope not in scopes:
                raise ConstructValidityError(
                    "assessment evidence reference is unavailable"
                )
            visit(target_scope, target, active | {key})

    for dimension in EVIDENCE_DIMENSIONS:
        visit("", dimension, frozenset())


def combine_pass_gates(passes: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Combine the two recorded passes without averaging away a failed gate."""

    if len(passes) != len(PASS_NAMES) or any(
        set(result) != set(ADMISSION_GATES) for result in passes
    ):
        raise ConstructValidityError("both admission-gate passes are required")
    return {
        gate: _merge_statuses([str(result[gate]) for result in passes])
        for gate in ADMISSION_GATES
    }
