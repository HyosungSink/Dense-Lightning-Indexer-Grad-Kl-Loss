"""Shared registry for Coverage, Correctness and Mock Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


CASES_DIR = Path(__file__).resolve().parents[1] / "st" / "cases"
SUITE_NAMES = ("coverage", "correctness", "mock")
CASE_KINDS = ("numeric", "shape_only", "invalid", "runtime_only", "abi_pending", "compact")


def _load_suite(name: str) -> dict[str, Any]:
    document = json.loads((CASES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if document["category"] != name or document["schema_version"] != 1:
        raise ValueError(f"invalid suite metadata: {name}")
    if not document["cases"]:
        raise ValueError(f"empty suite: {name}")
    for case in document["cases"]:
        if case.get("kind") not in CASE_KINDS or not case.get("id"):
            raise ValueError(f"invalid case descriptor in {name}: {case}")
        if name == "mock" and case["kind"] != "compact":
            raise ValueError("Mock Cases must declare the compact fixture contract")
    return document


_DOCUMENTS = {name: _load_suite(name) for name in SUITE_NAMES}
SUITES = {name: tuple(document["cases"]) for name, document in _DOCUMENTS.items()}
REFERENCE_CONVENTIONS = _DOCUMENTS["correctness"]["reference_conventions"]
_CASE_SUITES: dict[str, str] = {}
for _suite, _cases in SUITES.items():
    for _case in _cases:
        if _case["id"] in _CASE_SUITES:
            raise ValueError(f"duplicate case ID across suites: {_case['id']}")
        _CASE_SUITES[_case["id"]] = _suite


def all_specs(suite: str = "all") -> Iterable[tuple[str, dict[str, Any]]]:
    names = SUITE_NAMES if suite == "all" else (suite,)
    for name in names:
        for case in SUITES[name]:
            yield f"{case['kind']}_cases", case


def case_by_id(case_id: str, suite: str = "all") -> tuple[str, dict[str, Any]]:
    for group, case in all_specs(suite):
        if case["id"] == case_id:
            return group, case
    raise KeyError(case_id)


def suite_for_case(case_id: str) -> str:
    return _CASE_SUITES[case_id]


def numeric_case_by_id(case_id: str) -> dict[str, Any]:
    group, case = case_by_id(case_id)
    if group != "numeric_cases":
        raise KeyError(f"{case_id} is {group}, not a numeric case")
    return case


def _of_kind(kind: str) -> tuple[dict[str, Any], ...]:
    return tuple(case for group, case in all_specs() if group == f"{kind}_cases")


NUMERIC_CASES = _of_kind("numeric")
SHAPE_ONLY_CASES = _of_kind("shape_only")
INVALID_CASES = _of_kind("invalid")
RUNTIME_ONLY_CASES = _of_kind("runtime_only")
ABI_PENDING_CASES = _of_kind("abi_pending")
COMPACT_CASES = _of_kind("compact")
MOCK_CASES = SUITES["mock"]
