"""Exercise suite selection, compact fixtures and device-test failure reporting."""

import json
import copy
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.common.case_matrix import COMPACT_CASES, MOCK_CASES, SUITES, case_by_id
from tests.common.reference import decode_bfloat16_bits
from tests.st.scripts.compact_fixture import make_fixture
from tests.st.scripts import run_device
from tests.st.scripts.device_runtime import DeviceError
from tests.st.scripts.verify_result import error_ratios


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("suite", ("coverage", "correctness", "mock"))
def test_cli_selects_only_the_requested_suite(suite):
    result = subprocess.run(
        [sys.executable, "-m", "tests.st.scripts.gen_data", "--list", "--suite", suite],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    listed = [line.split() for line in result.stdout.splitlines()]
    assert {row[0] for row in listed} == {suite}
    assert {row[2] for row in listed} == {case["id"] for case in SUITES[suite]}


def test_cli_rejects_cross_suite_selection_without_generating_data(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tests.st.scripts.gen_data", "--suite", "coverage",
         "--case", "mock_case_1", "--output-dir", str(tmp_path)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "unknown case in suite coverage" in result.stderr
    assert not list(tmp_path.iterdir())


def test_cli_materializes_all_mock_cases(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "tests.st.scripts.gen_data", "--suite", "mock",
         "--all", "--output-dir", str(tmp_path)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert {path.name for path in tmp_path.iterdir()} == {case["id"] for case in MOCK_CASES}
    for path in tmp_path.iterdir():
        metadata = json.loads((path / "case.json").read_text())
        assert metadata["suite"] == "mock"
        assert metadata["abi"] == "compact_five_inputs"
        assert len(metadata["input_order"]) == 5
        assert len(metadata["output_order"]) == 4


@pytest.mark.parametrize("spec", COMPACT_CASES, ids=lambda spec: spec["id"])
def test_compact_fixtures_are_reproducible_finite_and_follow_input_shapes(spec, tmp_path):
    first = make_fixture(spec, tmp_path / "first")
    second = make_fixture(spec, tmp_path / "second")
    metadata = json.loads((first / "case.json").read_text())
    tensors = metadata["tensors"]
    for entry in tensors.values():
        left, right = first / entry["file"], second / entry["file"]
        assert left.read_bytes() == right.read_bytes()
        assert left.stat().st_size == entry["bytes"]
    for output, source in (("dQueryIndex", "queryIndex"), ("dKeyIndex", "keyIndex"),
                           ("dWeights", "weights")):
        assert tensors[output]["shape"] == tensors[source]["shape"]
        assert tensors[output]["logical_dtype"] == tensors[source]["logical_dtype"]
    assert tensors["loss"]["shape"] == [1]
    assert tensors["loss"]["logical_dtype"] == "float32"
    for name in metadata["output_order"]:
        entry = tensors[name]
        values = np.fromfile(first / entry["file"], dtype=entry["storage_dtype"])
        if entry["logical_dtype"] == "bfloat16":
            values = decode_bfloat16_bits(values)
        assert np.isfinite(values).all(), name


@pytest.mark.parametrize("status,exit_code", [
    ("Pass", 0), ("Wrong Answer", 1), ("Runtime Error", 1), ("Harness Error", 1),
])
def test_device_cli_propagates_test_failures(status, exit_code, tmp_path, monkeypatch):
    report = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "run_device", "--case-dir", str(tmp_path), "--build-dir", str(tmp_path),
        "--actual-dir", str(tmp_path / "actual"), "--report", str(report),
    ])
    monkeypatch.setattr(run_device, "run", lambda *args: {
        "case_id": "mock_case_1", "status": status, "stage": "complete", "acl_status": None,
    })
    assert run_device.main() == exit_code
    assert json.loads(report.read_text())["status"] == status


def test_device_runner_refuses_a_formal_case_without_a_compact_bridge(tmp_path, monkeypatch):
    from tests.st.scripts.gen_data import materialize_numeric_case

    def unexpected_runtime(*args):
        pytest.fail("the device must not be initialized for an unsupported ABI")

    monkeypatch.setattr(run_device, "Runtime", unexpected_runtime)
    _, spec = case_by_id("variable_lengths_and_padding_poison", "correctness")
    fixture = materialize_numeric_case(spec, tmp_path / "fixtures")
    with pytest.raises(ValueError, match="requires the full ABI"):
        run_device.run(fixture, tmp_path / "build", tmp_path / "actual")


def test_device_report_keeps_per_iteration_mismatch_diagnostics(tmp_path, monkeypatch):
    _, spec = case_by_id("mock_case_3", "mock")
    fixture = make_fixture(spec, tmp_path / "fixtures")
    metadata = json.loads((fixture / "case.json").read_text())
    expected = [
        np.fromfile(fixture / metadata["tensors"][name]["file"],
                    dtype=metadata["tensors"][name]["storage_dtype"]).reshape(
                        metadata["tensors"][name]["shape"])
        for name in metadata["output_order"]
    ]

    class FixtureRuntime:
        workspace_size = 0
        op_timeout_apis = []

        def __init__(self, *args):
            self.iteration = 0

        def tensor(self, raw, dtype):
            return {"raw": raw.copy(), "value": raw.copy()}

        def reset(self, tensor):
            tensor["value"] = tensor["raw"].copy()

        def stage(self, inputs, outputs, scale):
            return outputs

        def execute(self, outputs):
            for tensor, golden in zip(outputs, expected):
                tensor["value"] = golden.copy()
            output = 0 if self.iteration == 0 else 3
            outputs[output]["value"].reshape(-1)[0] += 100
            self.iteration += 1

        def read(self, tensor):
            return tensor["value"].copy(), True

        def close(self):
            pass

    monkeypatch.setattr(run_device, "Runtime", FixtureRuntime)
    report = run_device.run(fixture, tmp_path / "build", tmp_path / "actual", repeat=2)
    assert report["status"] == "Wrong Answer"
    assert [entry["error_ratio"]["first_failing_output"] for entry in report["iterations"]] == [
        "dQueryIndex", "loss",
    ]
    for entry in report["iterations"]:
        assert entry["error_ratio"] == error_ratios(entry["comparisons"])
    assert report["error_ratio_per_iteration"] == [
        entry["error_ratio"] for entry in report["iterations"]
    ]
    assert report["error_ratio"] == error_ratios(report["comparisons"])
    assert report["error_ratio"]["first_failing_output"] == "dQueryIndex"
    assert report["error_ratio"]["per_tensor"]["loss"] == 1.0


def test_device_api_failure_retains_status_without_a_fabricated_rate(tmp_path, monkeypatch):
    _, spec = case_by_id("mock_case_3", "mock")
    fixture = make_fixture(spec, tmp_path / "fixtures")

    def unavailable_runtime(*args):
        raise DeviceError("acl_init", 1)

    monkeypatch.setattr(run_device, "Runtime", unavailable_runtime)
    report = run_device.run(fixture, tmp_path / "build", tmp_path / "actual")
    assert report["status"] == "Runtime Error"
    assert report["stage"] == "acl_init" and report["acl_status"] == 1
    assert report["error_ratio"]["first_failing"] is None
    assert report["error_ratio"]["first_failing_output"] is None
    assert report["error_ratio"]["element_total"] == 0


def test_multisession_runner_uses_distinct_processes_and_keeps_failed_sessions(tmp_path, monkeypatch):
    _, spec = case_by_id("mock_case_3", "mock")
    fixture = make_fixture(spec, tmp_path / "fixtures")
    build = tmp_path / "build"
    build.mkdir()
    (build / "libcust_opapi.so").write_bytes(b"test binary")
    commands = []

    def process(command, log_path, environment, timeout):
        index = len(commands)
        commands.append(command)
        assert command[:2] == ["msprof", "op"]
        assert "--warm-up=5" in command and "--launch-count=2" in command
        assert command[command.index("--warmup") + 1] == "0"
        profile = Path(next(part.removeprefix("--output=") for part in command
                            if part.startswith("--output=")))
        runner = Path(command[command.index("--report") + 1])
        comparisons = [{"name": name, "passed": True, "mismatch_count": 0, "element_count": 8}
                       for name in run_device.OUTPUT_NAMES]
        if index == 1:
            comparisons[2].update(passed=False, mismatch_count=1)
        execution = {
            "case_id": spec["id"], "status": "Wrong Answer" if index == 1 else "Pass",
            "stage": "verification" if index == 1 else "complete", "guards": {"all": True},
            "iterations": [{"iteration": row, "comparisons": comparisons, "guards": {"all": True}}
                           for row in range(2)],
        }
        runner.write_text(json.dumps(execution))
        with (profile / "OpBasicInfo.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("Op Name", "Task Duration(us)"))
            writer.writeheader()
            for value in ([96, 104], [118, 122], [99, 101])[index]:
                writer.writerow({"Op Name": "DenseLightningIndexerGradKlLoss_kernel",
                                 "Task Duration(us)": value})
        return (1 if index == 1 else 0), False

    monkeypatch.setattr(run_device, "_execute_process", process)
    monkeypatch.setattr(run_device, "runtime_environment", lambda path: {})
    result = run_device.run_sessions(fixture, build, tmp_path / "actual",
                                     sessions=3, profiled=True, repeat=2, warmup=5)
    assert len(commands) == 3
    assert len({tuple(command) for command in commands}) == 3
    assert result["elapsed_us"] == 100
    assert result["status"] == "Wrong Answer"
    assert result["session_statuses"] == ["Pass", "Wrong Answer", "Pass"]
    assert len(result["profile_records"]) == 6
    assert len(result["iterations"]) == 6
    assert result["error_ratio"]["first_failing_output"] == "dWeights"
    assert (Path(result["run_dir"]) / "result.json").is_file()


def test_session_timeout_has_no_fabricated_kernel_duration(tmp_path, monkeypatch):
    _, spec = case_by_id("mock_case_3", "mock")
    fixture = make_fixture(spec, tmp_path / "fixtures")
    build = tmp_path / "build"
    build.mkdir()
    (build / "libcust_opapi.so").write_bytes(b"test binary")
    monkeypatch.setattr(run_device, "_execute_process", lambda *args: (-15, True))
    monkeypatch.setattr(run_device, "runtime_environment", lambda path: {})
    result = run_device.run_sessions(fixture, build, tmp_path / "actual",
                                     sessions=2, profiled=True, repeat=2)
    assert result["status"] == "Time Limit Exceeded"
    assert result["elapsed_us"] is None
    assert result["session_means_us"] == [None, None]
    assert not result["timing_complete"]


def test_process_timeout_terminates_and_reaps_the_child_group(tmp_path, monkeypatch):
    class Process:
        pid = 123456
        returncode = -15

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("test-process", timeout)
            return self.returncode

    process = Process()
    signals = []
    monkeypatch.setattr(run_device.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(run_device.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert run_device._execute_process(["test-process"], tmp_path / "process.log", {}, 1) == (-15, True)
    assert signals == [(process.pid, run_device.signal.SIGTERM)]
    assert process.waits == 2


def test_partial_mock_reruns_preserve_other_cases_and_propagate_failure():
    ids = ["mock_case_1", "mock_case_2", "mock_case_3"]
    identity = {"build": "binary", "catalog": "specs", "protocol": {"sessions": 3}}
    previous = run_device.merge_mock_results(None, identity, [
        {"case_id": ids[0], "status": "Wrong Answer", "samples_us": [10, 12]},
        {"case_id": ids[1], "status": "Pass", "samples_us": [20, 22]},
    ], ids)
    before = copy.deepcopy(previous)
    updated = run_device.merge_mock_results(previous, identity, [
        {"case_id": ids[2], "status": "Pass", "samples_us": [30, 32]},
    ], ids)
    assert updated["cases"][:2] == before["cases"]
    assert updated["complete"] and not updated["passed"]
    assert previous == before
    replaced = run_device.merge_mock_results(updated, identity, [
        {"case_id": ids[0], "status": "Pass", "samples_us": [11, 13]},
    ], ids)
    assert replaced["passed"]
    failed = run_device.merge_mock_results(replaced, identity, [
        {"case_id": ids[1], "status": "Harness Error"},
    ], ids)
    assert not failed["passed"]
    assert failed["cases"][1]["status"] == "Harness Error"
    with pytest.raises(ValueError, match="identity differs"):
        run_device.merge_mock_results(previous, identity | {"build": "another binary"}, [], ids)
    assert previous == before


@pytest.mark.parametrize("status,exit_code", [("Pass", 0), ("Wrong Answer", 1), ("Harness Error", 1)])
def test_profile_cli_uses_fixed_session_protocol_and_failure_exit(
    tmp_path, monkeypatch, status, exit_code,
):
    called = []

    def sessions(*args, **kwargs):
        called.append(kwargs)
        return {"status": status}

    monkeypatch.setattr(run_device, "run_sessions", sessions)
    monkeypatch.setattr(sys, "argv", [
        "run_device", "--profile", "--sessions", "3", "--case-dir", str(tmp_path),
        "--build-dir", str(tmp_path), "--actual-dir", str(tmp_path / "actual"),
        "--report", str(tmp_path / "report.json"), "--device-lock", str(tmp_path / "device.lock"),
    ])
    assert run_device.main() == exit_code
    assert called[0]["sessions"] == 3
    assert called[0]["repeat"] == 5 and called[0]["warmup"] == 5
    assert called[0]["profiled"]
