#!/usr/bin/env python3
"""Verify NPU output binaries against one generated ST case."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_ORDER = ("dQueryIndex", "dKeyIndex", "dWeights", "loss")
TIMING_STATISTIC = "mean"
SESSION_STATISTIC = "median"


@dataclass(frozen=True)
class VerificationResult:
    name: str
    passed: bool
    message: str
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatch_count: int = 0
    element_count: int = 0


def error_ratios(results) -> dict[str, Any]:
    """Derive local mismatch fractions from counts, in declared output order.

    Both live comparison results and their JSON dictionaries are accepted.
    Fractions describe numerical comparisons only; an absent comparison has
    no measured rate. No cached summary or external verdict is consulted.
    """

    items = [item if isinstance(item, dict) else asdict(item) for item in results]
    positions = {name: index for index, name in enumerate(OUTPUT_ORDER)}
    counted = sorted(
        (item for item in items if item["element_count"] > 0),
        key=lambda item: positions.get(item["name"], len(positions)),
    )
    mismatches = sum(item["mismatch_count"] for item in counted)
    elements = sum(item["element_count"] for item in counted)
    per_tensor = {item["name"]: item["mismatch_count"] / item["element_count"] for item in counted}
    first = next((item for item in counted if item["mismatch_count"] > 0), None)
    return {
        "total": mismatches / elements if elements else None,
        "max_tensor": max(per_tensor.values()) if per_tensor else None,
        "first_failing": per_tensor[first["name"]] if first else (0.0 if counted else None),
        "first_failing_output": first["name"] if first else None,
        "per_tensor": per_tensor,
        "mismatch_total": mismatches,
        "element_total": elements,
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing_statistics(samples) -> dict[str, float]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("kernel durations must be finite and positive")
    even, odd = values[::2], values[1::2]
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
        "phase_balanced_mean": ((statistics.fmean(even) + statistics.fmean(odd)) / 2
                               if odd else statistics.fmean(even)),
    }


def session_statistics(means):
    values = [float(value) for value in means if value is not None]
    if not values:
        return None
    stats = timing_statistics(values)
    return {key: stats[key] for key in ("mean", "median", "min", "max")} | {
        "sessions": len(values), "spread": (stats["max"] - stats["min"]) / stats["mean"],
    }


def read_profile_records(profile: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(profile.rglob("OpBasicInfo*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if "DenseLightningIndexerGradKlLoss" in row.get("Op Name", ""):
                    records.append({**row, "csv_path": str(path.resolve()),
                                    "csv_sha256": file_digest(path), "csv_row": index})
    return records


def summarize_run_report(report: dict, verify_files: bool = True) -> dict:
    """Recompute local run results from session records and comparison counts."""

    result = copy.deepcopy(report)
    if result.get("kind") == "mock_suite":
        cases = result.get("cases") or []
        if len({case["case_id"] for case in cases}) != len(cases):
            raise ValueError("duplicate case results")
        result["cases"] = [summarize_run_report(case, verify_files) for case in cases]
        result["passed"] = bool(cases) and all(case["status"] == "Pass" for case in result["cases"])
        result["complete"] = {case["case_id"] for case in cases} == set(result["case_ids"])
        return result
    repeat, count = result["repeat"], result["session_count"]
    if type(repeat) is not int or type(count) is not int or min(repeat, count) < 1:
        raise ValueError("repeat and session_count must be positive integers")
    sessions = result.get("sessions") or []
    errors = []
    if verify_files:
        from tests.st.scripts.run_device import build_identity, fixture_identity

        try:
            actual_identity = {"build": build_identity(Path(result["build_dir"])),
                               "fixture": fixture_identity(Path(result["case_dir"]))}
            if result["identity"] != actual_identity:
                raise ValueError("build or fixture identity differs from the recorded run")
        except (OSError, KeyError, ValueError) as error:
            errors.append(str(error))
    if len(sessions) != count:
        errors.append(f"expected {count} sessions, received {len(sessions)}")
    means, records, iterations = [], [], []
    seen_records = set()
    worst = {}
    for index, session in enumerate(sessions):
        issues = []
        if verify_files:
            try:
                path = Path(session["runner_report"])
                if file_digest(path) != session["runner_report_sha256"]:
                    raise ValueError("runner report hash mismatch")
                execution = json.loads(path.read_text())
                if execution["case_id"] != result["case_id"]:
                    raise ValueError("runner report belongs to another case")
                for key in ("status", "stage", "acl_status", "error", "comparisons", "iterations", "guards"):
                    session.pop(key, None)
                    if key in execution:
                        session[key] = execution[key]
                session["execution_status"] = execution["status"]
            except (OSError, KeyError, ValueError) as error:
                issues.append(str(error))
        status = session.get("execution_status", session.get("status", "Harness Error"))
        observed = session.get("iterations") or []
        for entry in observed:
            entry["error_ratio"] = error_ratios(entry.get("comparisons") or [])
            iterations.append({**entry, "session": index + 1})
            for comparison in entry.get("comparisons") or []:
                name = comparison["name"]
                if name not in worst or comparison["mismatch_count"] > worst[name]["mismatch_count"]:
                    worst[name] = comparison
        if status == "Pass":
            complete = len(observed) == repeat and all(
                [item["name"] for item in entry.get("comparisons", [])] == list(OUTPUT_ORDER)
                and bool(entry.get("guards")) for entry in observed)
            if not complete or not session.get("guards"):
                issues.append("incomplete numerical comparisons or guard checks")
            elif (not all(session["guards"].values()) or any(
                    not all(entry["guards"].values())
                    or not all(item["passed"] and item["mismatch_count"] == 0
                               for item in entry["comparisons"]) for entry in observed)):
                status = "Wrong Answer"
        if session.get("timed_out"):
            status = "Time Limit Exceeded"
        elif session.get("process_error") or (session.get("process_returncode") != 0 and status == "Pass"):
            issues.append(session.get("process_error") or "profiler/runner process failed")
        session_records = session.get("profile_records") or []
        session["elapsed_us"] = None
        session["samples_us"] = []
        session.pop("timing_statistics", None)
        if result.get("profiled"):
            try:
                if len(session_records) != repeat:
                    raise ValueError(f"expected {repeat} kernel records, received {len(session_records)}")
                if verify_files:
                    for record in session_records:
                        if file_digest(Path(record["csv_path"])) != record["csv_sha256"]:
                            raise ValueError("profiler CSV hash mismatch")
                    raw = read_profile_records(Path(session["profile_dir"]))
                    if raw != session_records:
                        raise ValueError("stored profile records differ from raw CSV records")
                for record in session_records:
                    identity = (record["csv_path"], record["csv_row"])
                    if identity in seen_records:
                        raise ValueError("duplicate profiler record")
                    seen_records.add(identity)
                samples = [float(record["Task Duration(us)"]) for record in session_records]
                stats = timing_statistics(samples)
                session.update(elapsed_us=stats[TIMING_STATISTIC], samples_us=samples,
                               timing_statistics=stats)
            except (OSError, KeyError, TypeError, ValueError) as error:
                issues.append(str(error))
        means.append(session["elapsed_us"])
        records.extend(session_records)
        session["errors"] = issues
        session["status"] = "Harness Error" if issues and status == "Pass" else status
        if issues:
            errors.extend(f"session {index + 1}: {issue}" for issue in issues)
    combined = session_statistics(means)
    timing_complete = (len(sessions) == count and all(value is not None for value in means))
    failed = next((session for session in sessions if session["status"] != "Pass"), None)
    result.update(
        status=failed["status"] if failed else ("Harness Error" if errors else "Pass"),
        stage=failed.get("stage", "session") if failed else ("verification" if errors else "complete"),
        acl_status=failed.get("acl_status") if failed else None,
        sessions=sessions, session_statuses=[session["status"] for session in sessions],
        session_means_us=means, session_timings=combined, session_statistic=SESSION_STATISTIC,
        timing_statistic=TIMING_STATISTIC, timing_complete=bool(result.get("profiled") and timing_complete),
        elapsed_us=combined[SESSION_STATISTIC] if combined and timing_complete else None,
        samples_us=[value for session in sessions for value in session["samples_us"]],
        profile_records=records, iterations=iterations, comparisons=list(worst.values()), errors=errors,
        error_ratio=error_ratios(worst.values()),
        error_ratio_per_iteration=[entry["error_ratio"] for entry in iterations],
    )
    return result


def _load_numeric(path: Path, logical_dtype: str) -> np.ndarray:
    if logical_dtype == "float16":
        return np.fromfile(path, dtype=np.float16).astype(np.float32)
    if logical_dtype == "float32":
        return np.fromfile(path, dtype=np.float32)
    if logical_dtype == "bfloat16":
        words = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
        return (words << np.uint32(16)).view(np.float32)
    raise ValueError(f"unsupported output dtype: {logical_dtype}")


def verify_tensor(
    name: str,
    actual_path: Path,
    golden_path: Path,
    tensor: dict[str, Any],
    comparison: dict[str, Any],
) -> VerificationResult:
    expected_bytes = int(tensor["bytes"])
    elements = int(tensor["numel"])
    if not actual_path.is_file():
        return VerificationResult(name, False, f"actual output is missing: {actual_path}",
                                  element_count=elements, mismatch_count=elements)
    if not golden_path.is_file():
        return VerificationResult(name, False, f"golden output is missing: {golden_path}",
                                  element_count=elements, mismatch_count=elements)
    if actual_path.stat().st_size != expected_bytes:
        return VerificationResult(
            name,
            False,
            f"actual byte size mismatch: expected={expected_bytes}, actual={actual_path.stat().st_size}",
            element_count=elements,
            mismatch_count=elements,
        )
    if golden_path.stat().st_size != expected_bytes:
        return VerificationResult(
            name,
            False,
            f"golden byte size mismatch: expected={expected_bytes}, actual={golden_path.stat().st_size}",
            element_count=elements,
            mismatch_count=elements,
        )

    actual = _load_numeric(actual_path, tensor["logical_dtype"])
    golden = _load_numeric(golden_path, tensor["logical_dtype"])
    if actual.size != golden.size:
        return VerificationResult(
            name, False, f"element count mismatch: expected={golden.size}, actual={actual.size}",
            element_count=elements, mismatch_count=elements
        )
    nonfinite = comparison.get("require_finite", True) and not np.isfinite(actual).all()

    rtol = float(comparison["rtol"])
    atol = float(comparison["atol"])
    absolute_error = np.abs(actual - golden)
    tolerance = atol + rtol * np.abs(golden)
    matches = np.isfinite(actual) & np.isfinite(golden) & (absolute_error <= tolerance)
    mismatch_count = int(np.count_nonzero(~matches))
    finite_absolute = absolute_error[np.isfinite(absolute_error)]
    max_abs_error = float(finite_absolute.max(initial=0.0))
    denominator = np.maximum(np.abs(golden), np.float32(1.0e-12))
    relative_error = absolute_error / denominator
    finite_relative = relative_error[np.isfinite(relative_error)]
    max_rel_error = float(finite_relative.max(initial=0.0))
    passed = mismatch_count == 0
    message = "comparison passed" if passed else f"comparison failed: {mismatch_count}/{golden.size} mismatches"
    if nonfinite:
        message = "actual output contains NaN or Inf"
    return VerificationResult(
        name,
        passed,
        message,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        mismatch_count=mismatch_count,
        element_count=elements,
    )


def verify_case(case_dir: Path, actual_dir: Path) -> tuple[VerificationResult, ...]:
    metadata_path = case_dir / "case.json"
    if not metadata_path.is_file():
        return (VerificationResult("case", False, f"case metadata is missing: {metadata_path}"),)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("expected") != "success":
        return (VerificationResult("case", False, "only success cases have output binaries"),)

    results = []
    for name in metadata["output_order"]:
        tensor = metadata["tensors"][name]
        results.append(
            verify_tensor(
                name=name,
                actual_path=actual_dir / f"{name}.bin",
                golden_path=case_dir / tensor["file"],
                tensor=tensor,
                comparison=metadata["comparisons"][name],
            )
        )
    return tuple(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case-dir", type=Path)
    source.add_argument("--run-report", type=Path, help="recompute a local session/suite report from raw records")
    parser.add_argument(
        "--actual-dir",
        type=Path,
        help="directory containing dQueryIndex.bin, dKeyIndex.bin, dWeights.bin, and loss.bin",
    )
    parser.add_argument("--aggregation", choices=("first_failing", "total", "max_tensor"),
                        default="first_failing")
    parser.add_argument("--output", type=Path, help="write a recomputed run report")
    args = parser.parse_args()
    if args.case_dir and not args.actual_dir:
        parser.error("--case-dir requires --actual-dir")
    return args


def main() -> int:
    args = parse_args()
    if args.run_report:
        result = summarize_run_report(json.loads(args.run_report.read_text()))
        for case in result.get("cases", [result]):
            case["selected_aggregation"] = args.aggregation
            case["selected_error_ratio"] = case["error_ratio"][args.aggregation]
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        else:
            print(rendered, end="")
        return 0 if result.get("passed", result.get("status") == "Pass") else 1
    results = verify_case(args.case_dir, args.actual_dir)
    for result in results:
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.message}; "
            f"max_abs={result.max_abs_error:.8g}, max_rel={result.max_rel_error:.8g}, "
            f"mismatches={result.mismatch_count}/{result.element_count}"
        )
    print(json.dumps({"error_ratio": error_ratios(results)}, ensure_ascii=False))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
