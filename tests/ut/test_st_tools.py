from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.common.case_matrix import case_by_id, numeric_case_by_id
from tests.common.reference import decode_bfloat16_bits, make_inputs
from tests.st.scripts.gen_data import materialize_numeric_case, materialize_spec_case
from tests.st.scripts import verify_result
from tests.st.scripts.verify_result import VerificationResult, error_ratios, verify_case, verify_tensor
from tests.st.scripts.verify_result import (
    file_digest, read_profile_records, session_statistics, summarize_run_report, timing_statistics,
)


def _copy_golden_to_actual(case_dir: Path, actual_dir: Path) -> None:
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    actual_dir.mkdir(parents=True, exist_ok=True)
    for name in metadata["output_order"]:
        golden = case_dir / metadata["tensors"][name]["file"]
        (actual_dir / f"{name}.bin").write_bytes(golden.read_bytes())


def test_raw_bfloat16_fixture_round_trips_and_records_full_runner_contract(tmp_path) -> None:
    spec = numeric_case_by_id("dtype_bfloat16_weights_float32")
    case_dir = materialize_numeric_case(spec, tmp_path)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))

    assert metadata["abi"] == "formal_full_abi"
    assert metadata["expected"] == "success"
    assert metadata["input_order"] == [
        "query",
        "key",
        "queryIndex",
        "keyIndex",
        "weights",
        "softmaxMax",
        "softmaxSum",
        "softmaxMaxIndex",
        "softmaxSumIndex",
        "queryRope",
        "keyRope",
    ]
    assert metadata["output_order"] == ["dQueryIndex", "dKeyIndex", "dWeights", "loss"]
    assert metadata["attributes"]["scaleValue"] == np.float64(1 / np.sqrt(128))
    assert metadata["attributes"]["layout"] == "BSND"
    assert metadata["attributes"]["sparseMode"] == 3
    assert metadata["runner_checks"]["all_outputs_must_be_written"] is True
    assert metadata["runner_checks"]["guard_regions"]["bytes_before"] == 64

    query_meta = metadata["tensors"]["query"]
    query_words = np.fromfile(case_dir / query_meta["file"], dtype=np.uint16).reshape(
        query_meta["shape"]
    )
    np.testing.assert_array_equal(
        decode_bfloat16_bits(query_words), make_inputs(spec)["query"]
    )
    assert query_meta["logical_dtype"] == "bfloat16"
    assert query_meta["storage"] == "bfloat16_raw_uint16"
    assert metadata["tensors"]["dWeights"]["logical_dtype"] == "float32"
    assert metadata["tensors"]["loss"]["logical_dtype"] == "float32"


def test_documented_example_exposes_safe_compact_schema_bridge(tmp_path) -> None:
    spec = numeric_case_by_id("documented_example")
    case_dir = materialize_numeric_case(spec, tmp_path)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    compact = metadata["abi_variants"]["current_compact_schema"]
    assert compact["applicable"] is True
    assert compact["input_order"] == ["query", "key", "queryIndex", "keyIndex", "weights"]
    assert compact["attribute_name"] == "scale_value"
    assert compact["recompute_forward_statistics"] is True


def test_explicit_actual_lengths_follow_rope_in_formal_input_order(tmp_path) -> None:
    spec = numeric_case_by_id("variable_lengths_and_padding_poison")
    case_dir = materialize_numeric_case(spec, tmp_path)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert metadata["input_order"][-4:] == [
        "queryRope",
        "keyRope",
        "actual_seq_lengths_query",
        "actual_seq_lengths_key",
    ]
    for name in ("actual_seq_lengths_query", "actual_seq_lengths_key"):
        tensor = metadata["tensors"][name]
        assert tensor["logical_dtype"] == "int64"
        assert tensor["storage_dtype"] == "int64"


def test_st_verifier_accepts_matching_four_outputs(tmp_path) -> None:
    spec = numeric_case_by_id("documented_example")
    case_dir = materialize_numeric_case(spec, tmp_path / "cases")
    actual_dir = tmp_path / "actual"
    _copy_golden_to_actual(case_dir, actual_dir)

    results = verify_case(case_dir, actual_dir)
    assert len(results) == 4
    assert all(result.passed for result in results), results


def test_st_verifier_rejects_nonfinite_and_truncated_outputs(tmp_path) -> None:
    spec = numeric_case_by_id("documented_example")
    case_dir = materialize_numeric_case(spec, tmp_path / "cases")
    actual_dir = tmp_path / "actual"
    _copy_golden_to_actual(case_dir, actual_dir)

    np.asarray([np.nan], dtype=np.float32).tofile(actual_dir / "loss.bin")
    d_weights_path = actual_dir / "dWeights.bin"
    d_weights_path.write_bytes(d_weights_path.read_bytes()[:-2])
    results = {result.name: result for result in verify_case(case_dir, actual_dir)}
    assert not results["loss"].passed
    assert "NaN or Inf" in results["loss"].message
    assert not results["dWeights"].passed
    assert "byte size mismatch" in results["dWeights"].message


def test_spec_only_case_records_deferred_execution_without_allocating_tensors(tmp_path) -> None:
    group, spec = case_by_id("max_s2_shape_only")
    case_dir = materialize_spec_case(group, spec, tmp_path)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert metadata["expected"] == "shape_or_performance_run"
    assert metadata["materialized"] is False
    assert list(case_dir.iterdir()) == [case_dir / "case.json"]


def test_first_failure_uses_declared_output_order_and_per_output_counts() -> None:
    ordered = [
        {"name": "dQueryIndex", "mismatch_count": 176, "element_count": 4096},
        {"name": "dKeyIndex", "mismatch_count": 140, "element_count": 512},
        {"name": "dWeights", "mismatch_count": 23, "element_count": 32},
        {"name": "loss", "mismatch_count": 1, "element_count": 1},
    ]
    rates = error_ratios(reversed(ordered))
    assert rates["first_failing"] == pytest.approx(176 / 4096)
    assert rates["first_failing_output"] == "dQueryIndex"
    assert rates["max_tensor"] == 1.0
    assert rates["total"] == pytest.approx(340 / 4641)
    assert rates["mismatch_total"] == 340
    assert rates["element_total"] == 4641
    for item in ordered[:3]:
        item["mismatch_count"] = 0
    assert error_ratios(ordered)["first_failing_output"] == "loss"
    assert error_ratios(ordered)["first_failing"] == 1.0
    ordered[3]["mismatch_count"] = 0
    assert error_ratios(ordered)["first_failing_output"] is None
    assert error_ratios(ordered)["first_failing"] == 0.0


def test_local_ratios_are_recomputed_from_serialized_comparison_counts() -> None:
    result = VerificationResult("dWeights", False, "mismatch", mismatch_count=1, element_count=16)
    stored = asdict(result)
    stored["error_ratio"] = {"first_failing": 0.99}
    assert error_ratios([result]) == error_ratios([stored])
    assert error_ratios([stored])["first_failing"] == 0.0625


@pytest.mark.parametrize("results", [
    [], [VerificationResult("case", False, "missing metadata")],
])
def test_unavailable_comparisons_have_no_numerical_rate(results) -> None:
    rates = error_ratios(results)
    assert rates["first_failing_output"] is None
    assert rates["first_failing"] is None
    assert rates["total"] is None and rates["max_tensor"] is None
    assert rates["per_tensor"] == {}
    assert rates["element_total"] == 0


def test_nonfinite_outputs_also_count_finite_mismatches(tmp_path) -> None:
    actual = tmp_path / "actual.bin"
    golden = tmp_path / "golden.bin"
    np.asarray([np.nan, 7.0, 2.0, np.inf], dtype=np.float32).tofile(actual)
    np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32).tofile(golden)
    result = verify_tensor(
        "dWeights", actual, golden,
        {"bytes": 16, "numel": 4, "logical_dtype": "float32"},
        {"rtol": 1e-4, "atol": 1e-4, "require_finite": True},
    )
    assert not result.passed and "NaN or Inf" in result.message
    assert result.mismatch_count == 3
    assert error_ratios([result])["first_failing"] == 0.75


def test_verifier_cli_reports_first_failure_without_ignoring_later_outputs(
    tmp_path, monkeypatch, capsys,
) -> None:
    spec = numeric_case_by_id("documented_example")
    case_dir = materialize_numeric_case(spec, tmp_path / "cases")
    actual_dir = tmp_path / "actual"
    _copy_golden_to_actual(case_dir, actual_dir)
    weights = actual_dir / "dWeights.bin"
    values = np.fromfile(weights, dtype=np.float16)
    values[0] += 100
    values.tofile(weights)
    np.asarray([100.0], dtype=np.float32).tofile(actual_dir / "loss.bin")
    monkeypatch.setattr(sys, "argv", [
        "verify_result", "--case-dir", str(case_dir), "--actual-dir", str(actual_dir),
    ])
    assert verify_result.main() == 1
    output = capsys.readouterr().out
    rates = json.loads(output.splitlines()[-1])["error_ratio"]
    assert rates["first_failing_output"] == "dWeights"
    assert rates["first_failing"] == pytest.approx(1 / values.size)
    assert rates["per_tensor"]["loss"] == 1.0
    assert "FAIL loss:" in output


def _session_report(tmp_path, samples_by_session):
    from tests.st.scripts.run_device import build_identity, fixture_identity

    build, fixture = tmp_path / "build", tmp_path / "fixture"
    build.mkdir()
    fixture.mkdir()
    (build / "libcust_opapi.so").write_bytes(b"unit-test library identity")
    (fixture / "case.json").write_text("{}")
    sessions = []
    for index, samples in enumerate(samples_by_session, 1):
        root = tmp_path / f"session-{index}"
        root.mkdir()
        comparisons = [{"name": name, "passed": True, "mismatch_count": 0, "element_count": 8}
                       for name in verify_result.OUTPUT_ORDER]
        execution = {
            "case_id": "mock_case_3", "status": "Pass", "stage": "complete", "acl_status": None,
            "guards": {name: True for name in verify_result.OUTPUT_ORDER},
            "iterations": [{"iteration": row, "guards": {"outputs": True},
                            "comparisons": comparisons} for row in range(len(samples))],
        }
        runner = root / "runner.json"
        runner.write_text(json.dumps(execution))
        with (root / "OpBasicInfo.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("Op Name", "Task Duration(us)"))
            writer.writeheader()
            for value in samples:
                writer.writerow({"Op Name": "DenseLightningIndexerGradKlLoss_kernel",
                                 "Task Duration(us)": value})
        sessions.append({
            **execution, "runner_report": str(runner), "runner_report_sha256": file_digest(runner),
            "execution_status": "Pass", "process_returncode": 0, "profile_dir": str(root),
            "profile_records": read_profile_records(root),
        })
    return {
        "schema_version": 1, "kind": "device_sessions", "case_id": "mock_case_3",
        "build_dir": str(build), "case_dir": str(fixture),
        "identity": {"build": build_identity(build), "fixture": fixture_identity(fixture)},
        "profiled": True, "repeat": len(samples_by_session[0]),
        "session_count": len(sessions), "sessions": sessions,
    }


def test_session_timing_is_recomputed_as_median_of_raw_session_means(tmp_path):
    report = _session_report(tmp_path, [[96, 104], [118, 122], [99, 101]])
    report["elapsed_us"] = 999
    report["session_means_us"] = [999, 999, 999]
    for session in report["sessions"]:
        session["elapsed_us"] = 999
        session["samples_us"] = [999, 999]
    result = summarize_run_report(report)
    assert result["status"] == "Pass"
    assert result["session_means_us"] == [100, 120, 100]
    assert result["elapsed_us"] == 100
    assert result["session_statistic"] == "median"
    assert result["session_timings"]["mean"] == pytest.approx(320 / 3)
    assert result["session_timings"]["spread"] == pytest.approx(20 / (320 / 3))
    assert result["samples_us"] == [96, 104, 118, 122, 99, 101]
    assert len(result["iterations"]) == 6
    assert result["timing_complete"]


def test_missing_session_invalidates_the_timing_instead_of_averaging_survivors(tmp_path):
    report = _session_report(tmp_path, [[96, 104], [118, 122]])
    report["session_count"] = 3
    result = summarize_run_report(report)
    assert result["status"] == "Harness Error"
    assert result["elapsed_us"] is None
    assert not result["timing_complete"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nan", "infinity", "zero", "negative"])
def test_invalid_profiler_samples_fail_the_session(tmp_path, failure):
    report = _session_report(tmp_path, [[96, 104]])
    records = report["sessions"][0]["profile_records"]
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[1] = records[0].copy()
    else:
        records[0]["Task Duration(us)"] = {
            "nan": "nan", "infinity": "inf", "zero": "0", "negative": "-1",
        }[failure]
    result = summarize_run_report(report, verify_files=False)
    assert result["status"] == "Harness Error"
    assert result["elapsed_us"] is None
    assert result["errors"]


@pytest.mark.parametrize("changed", ["csv", "runner", "fixture", "library"])
def test_recomputed_reports_reject_changed_measurement_inputs(tmp_path, changed):
    report = _session_report(tmp_path, [[96, 104]])
    session = report["sessions"][0]
    paths = {
        "csv": Path(session["profile_records"][0]["csv_path"]),
        "runner": Path(session["runner_report"]),
        "fixture": Path(report["case_dir"]) / "case.json",
        "library": Path(report["build_dir"]) / "libcust_opapi.so",
    }
    with paths[changed].open("ab") as stream:
        stream.write(b" ")
    result = summarize_run_report(report)
    assert result["status"] == "Harness Error"


def test_one_numerically_failed_session_fails_the_whole_run(tmp_path):
    report = _session_report(tmp_path, [[96, 104], [98, 102]])
    session = report["sessions"][1]
    runner = Path(session["runner_report"])
    execution = json.loads(runner.read_text())
    execution["status"] = "Wrong Answer"
    execution["iterations"][0]["comparisons"][2].update(passed=False, mismatch_count=1)
    runner.write_text(json.dumps(execution))
    session["runner_report_sha256"] = file_digest(runner)
    result = summarize_run_report(report)
    assert result["status"] == "Wrong Answer"
    assert result["elapsed_us"] == 100
    assert result["session_statuses"] == ["Pass", "Wrong Answer"]
    assert result["error_ratio"]["first_failing_output"] == "dWeights"
    assert result["error_ratio_per_iteration"][2]["first_failing"] == 1 / 8


def test_malformed_sample_summary_cannot_replace_raw_csv_values(tmp_path):
    report = _session_report(tmp_path, [[150, 150], [150, 150]])
    report["elapsed_us"] = 100
    report["sessions"][0]["profile_records"][0]["Task Duration(us)"] = "100"
    result = summarize_run_report(report)
    assert result["status"] == "Harness Error"
    assert result["elapsed_us"] is None


def test_session_statistics_report_local_repeatability():
    result = session_statistics([33.2, 35.4, 40.6])
    assert result["sessions"] == 3 and result["median"] == pytest.approx(35.4)
    assert result["spread"] == pytest.approx((40.6 - 33.2) / ((33.2 + 35.4 + 40.6) / 3))
    assert session_statistics([]) is None
    assert timing_statistics([28, 36, 28, 36, 28])["phase_balanced_mean"] == 32


def test_report_cli_recomputes_without_executing_a_device(tmp_path, monkeypatch, capsys):
    report = _session_report(tmp_path, [[96, 104], [118, 122], [99, 101]])
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(report | {"elapsed_us": 999}))
    monkeypatch.setattr(sys, "argv", ["verify_result", "--run-report", str(source),
                                     "--aggregation", "total"])
    assert verify_result.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["elapsed_us"] == 100
    assert result["selected_aggregation"] == "total"
    assert result["selected_error_ratio"] == 0
