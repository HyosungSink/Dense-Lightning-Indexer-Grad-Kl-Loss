from __future__ import annotations

import pytest

from tests.common.case_matrix import (
    INVALID_CASES,
    REFERENCE_CONVENTIONS,
    NUMERIC_CASES,
    SHAPE_ONLY_CASES,
)
from tests.common.coverage import audit
from tests.common.reference import validate_case_spec


def test_documented_coverage_audit_passes() -> None:
    result = audit()
    assert not result.errors, "\n".join(result.errors)
    assert result.summary == {
        "numeric_cases": 29,
        "shape_only_cases": 3,
        "invalid_cases": 22,
        "runtime_only_cases": 6,
        "abi_pending_cases": 3,
        "compact_cases": 29,
        "coverage_suite": 50,
        "correctness_suite": 35,
        "mock_suite": 7,
        "checks": len(result.checks),
    }


@pytest.mark.parametrize("case", NUMERIC_CASES, ids=lambda case: case["id"])
def test_numeric_specs_satisfy_documented_constraints(case) -> None:
    validate_case_spec(case)


@pytest.mark.parametrize("case", SHAPE_ONLY_CASES, ids=lambda case: case["id"])
def test_shape_only_specs_satisfy_documented_constraints(case) -> None:
    validate_case_spec(case)


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda case: case["id"])
def test_invalid_specs_are_rejected_for_the_expected_reason(case) -> None:
    with pytest.raises(ValueError, match=case["expected_error"]):
        validate_case_spec(case)


def test_reference_conventions_make_document_ambiguities_explicit() -> None:
    conventions = REFERENCE_CONVENTIONS
    assert conventions["relu_zero_gradient"] == 0
    assert conventions["empty_causal_row"] == "inactive_zero_output"
    assert "queryRope" in conventions["rope_score"]
    assert conventions["intermediate_precision"] == "float32"
