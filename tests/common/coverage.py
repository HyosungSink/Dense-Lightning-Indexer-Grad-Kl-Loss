"""Executable audit against every documented operator requirement."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .case_matrix import (
    ABI_PENDING_CASES,
    INVALID_CASES,
    NUMERIC_CASES,
    RUNTIME_ONLY_CASES,
    SHAPE_ONLY_CASES,
    SUITES,
    all_specs,
)
from .reference import MAX_SEQUENCE, validate_case_spec


@dataclass(frozen=True)
class AuditResult:
    checks: tuple[str, ...]
    errors: tuple[str, ...]
    summary: dict[str, int]


def audit() -> AuditResult:
    checks: list[str] = []
    errors: list[str] = []

    def require(condition: bool, description: str) -> None:
        (checks if condition else errors).append(description)

    all_cases = tuple(all_specs())
    ids = [case["id"] for _, case in all_cases]
    tags = {tag for _, case in all_cases for tag in case.get("tags", [])}
    require(bool(ids) and len(ids) == len(set(ids)), "all case IDs are unique and non-empty")
    require(set(SUITES) == {"coverage", "correctness", "mock"} and all(SUITES.values()),
            "Coverage, Correctness and Mock Cases are non-empty and independently selectable")

    for case in (*NUMERIC_CASES, *SHAPE_ONLY_CASES):
        try:
            validate_case_spec(case)
        except ValueError as error:
            errors.append(f"valid case {case['id']} rejected: {error}")
    require(
        all(_invalid_case_matches(case) for case in INVALID_CASES),
        "all rejection cases fail for their declared reason",
    )

    head_pairs = {
        (case["shape"]["N1"], case["shape"]["Nidx1"])
        for case in NUMERIC_CASES
        if "head_cross_product" in case.get("tags", [])
    }
    require(
        head_pairs == set(itertools.product((32, 64, 128), (8, 16, 32, 64))),
        "all N1 x Nidx1 combinations have numeric coverage",
    )

    dtype_pairs = {
        (
            case.get("input_dtype", "float16"),
            case.get("weights_dtype", case.get("input_dtype", "float16")),
        )
        for case in NUMERIC_CASES
    }
    require(
        {
            ("float16", "float16"),
            ("float16", "float32"),
            ("bfloat16", "bfloat16"),
            ("bfloat16", "float32"),
        }
        <= dtype_pairs,
        "all documented input/weights dtype paths have numeric coverage",
    )

    require(any(case["shape"]["B"] == 1 for case in NUMERIC_CASES), "B=1 is numeric")
    require(any(case["shape"]["B"] == 256 for case in NUMERIC_CASES), "B=256 is numeric")
    require(any(case["shape"]["S1"] == 1 for case in NUMERIC_CASES), "S1=1 is numeric")
    require(any(case["shape"]["S2"] == 1 for case in NUMERIC_CASES), "S2=1 is numeric")
    require(
        any(case["shape"]["S1"] == MAX_SEQUENCE for case in SHAPE_ONLY_CASES),
        "S1=128K is represented without dense materialization",
    )
    require(
        any(case["shape"]["S2"] == MAX_SEQUENCE for case in SHAPE_ONLY_CASES),
        "S2=128K is represented without dense materialization",
    )
    max_s1 = next((case for case in SHAPE_ONLY_CASES if case["id"] == "max_s1_shape_only"), None)
    require(
        max_s1 is not None
        and max_s1["shape"]["B"] * max_s1["shape"]["S1"] * max_s1["shape"]["N1"] * 128
        >= 2**32,
        "shape-only matrix exposes uint32 element-count overflow",
    )

    required_tags = {
        "actual_lengths_explicit",
        "compact_schema_bridge",
        "dkey_accumulation",
        "documented_example",
        "empty_causal_row",
        "finite_difference",
        "finite_kl",
        "full_abi_only",
        "loss_sum",
        "mask_boundary",
        "non_aligned_tail",
        "online_softmax",
        "output_dtypes",
        "output_init",
        "padding_ignored",
        "probability_tail",
        "relu_negative",
        "relu_positive",
        "relu_zero",
        "right_down_causal",
        "rope_nonzero",
        "s1_gt_s2",
        "s1_lt_s2",
        "stable_softmax",
        "variable_lengths",
        "weights_zero",
        "zero_probability_boundary",
    }
    require(required_tags <= tags, "all documented semantic coverage tags are present")
    require(
        {"atlas_a2", "atlas_a3", "ascend_950_unsupported", "deterministic", "nondeterministic"}
        <= tags,
        "platform and determinism requirements have runtime specifications",
    )
    require(
        {"compact_float32_pending", "compact_fp16_loss_pending", "full_abi_pending"} <= tags,
        "Host/formal ABI conflicts are quarantined as pending cases",
    )

    summary = {
        "numeric_cases": len(NUMERIC_CASES),
        "shape_only_cases": len(SHAPE_ONLY_CASES),
        "invalid_cases": len(INVALID_CASES),
        "runtime_only_cases": len(RUNTIME_ONLY_CASES),
        "abi_pending_cases": len(ABI_PENDING_CASES),
        "compact_cases": sum(case["kind"] == "compact" for _, case in all_cases),
        **{f"{name}_suite": len(cases) for name, cases in SUITES.items()},
        "checks": len(checks),
    }
    return AuditResult(tuple(checks), tuple(errors), summary)


def _invalid_case_matches(case: dict) -> bool:
    try:
        validate_case_spec(case)
    except ValueError as error:
        return case["expected_error"] in str(error)
    return False


def main() -> int:
    result = audit()
    for check in result.checks:
        print(f"PASS: {check}")
    for error in result.errors:
        print(f"FAIL: {error}")
    print("summary:", ", ".join(f"{key}={value}" for key, value in result.summary.items()))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
