"""Run compact fixtures or the Mock suite, with independent profiling sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np

from tests.st.scripts.device_runtime import DEFAULT_OP_WAIT_SECONDS, DeviceError, Runtime
from tests.st.scripts.gen_data import COMPACT_INPUT_ORDER, OUTPUT_NAMES
from tests.st.scripts.verify_result import (
    error_ratios, file_digest, read_profile_records, summarize_run_report, verify_case,
)


ROOT = Path(__file__).resolve().parents[3]


def validate_fixture(case_dir: Path):
    metadata = json.loads((case_dir / "case.json").read_text())
    if metadata["abi"] != "compact_five_inputs" and not metadata.get(
            "abi_variants", {}).get("current_compact_schema", {}).get("applicable", False):
        raise ValueError("fixture requires the full ABI; no compact bridge is declared")
    # No silently dropping formal-only semantics when bridging the compact ABI.
    for name in ("queryRope", "keyRope"):
        if name in metadata["tensors"]:
            entry = metadata["tensors"][name]
            values = np.fromfile(case_dir / entry["file"], dtype=entry["storage_dtype"])
            if np.any(values):
                raise ValueError("nonzero RoPE requires the unconfirmed full evaluator ABI")
    if any(name in metadata["tensors"] for name in
           ("actual_seq_lengths_query", "actual_seq_lengths_key")):
        raise ValueError("explicit variable lengths require the unconfirmed full evaluator ABI")
    return metadata


def run(case_dir: Path, build_dir: Path, actual_dir: Path, warmup=0, repeat=1,
        device=0, op_wait_seconds=DEFAULT_OP_WAIT_SECONDS, guard_bytes=64):
    if warmup < 0 or repeat < 1 or op_wait_seconds < 0 or guard_bytes < 0:
        raise ValueError("warmup/timeout must be nonnegative and repeat must be positive")
    metadata = validate_fixture(case_dir)
    actual_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "case_id": metadata.get("case_id", metadata.get("id", case_dir.name)),
        "status": "Harness Error", "stage": "initialize", "acl_status": None,
        "warmup": warmup, "repeat": repeat, "device": device,
        "build_dir": str(build_dir),
        "op_wait_seconds": op_wait_seconds,
        "guard_bytes": guard_bytes,
        "error_ratio": error_ratios(()),
        "iterations": [], "comparisons": [], "guards": {},
    }
    runtime = None
    try:
        runtime = Runtime(build_dir, device, op_wait_seconds, guard_bytes)
        inputs = []
        for name in COMPACT_INPUT_ORDER:
            entry = metadata["tensors"][name]
            raw = np.fromfile(case_dir / entry["file"], dtype=entry["storage_dtype"])
            if raw.nbytes != entry["bytes"]:
                raise ValueError(f"invalid input byte size: {name}")
            inputs.append(runtime.tensor(raw.reshape(entry["shape"]), entry["logical_dtype"]))
        outputs = []
        for name in OUTPUT_NAMES:
            entry = metadata["tensors"][name]
            storage = np.dtype(entry["storage_dtype"])
            fill = 0x7FC1 if entry["logical_dtype"] == "bfloat16" else np.nan
            raw = np.full(entry["shape"], fill, dtype=storage)
            outputs.append(runtime.tensor(raw, entry["logical_dtype"]))
        scale = metadata["attributes"]["scaleValue"]
        for _ in range(warmup):
            runtime.execute(runtime.stage(inputs, outputs, scale))
        iterations = result["iterations"]
        guards = result["guards"]
        comparisons = {}
        for iteration in range(repeat):
            for tensor in outputs:
                runtime.reset(tensor)
            runtime.execute(runtime.stage(inputs, outputs, scale))
            iteration_dir = actual_dir / str(iteration)
            iteration_dir.mkdir(parents=True, exist_ok=True)
            current_guards = {}
            for name, tensor in zip(OUTPUT_NAMES, outputs):
                raw, current_guards[name] = runtime.read(tensor)
                raw.tofile(iteration_dir / (name + ".bin"))
                guards[name] = guards.get(name, True) and current_guards[name]
            current_comparisons = [asdict(item) for item in verify_case(case_dir, iteration_dir)]
            iterations.append({"iteration": iteration, "actual_dir": str(iteration_dir),
                               "guards": current_guards, "comparisons": current_comparisons,
                               "error_ratio": error_ratios(current_comparisons)})
            for item in current_comparisons:
                name = item["name"]
                if name not in comparisons or item["mismatch_count"] > comparisons[name]["mismatch_count"]:
                    comparisons[name] = item
            result["comparisons"] = list(comparisons.values())
            result["error_ratio"] = error_ratios(result["comparisons"])
            result["error_ratio_per_iteration"] = [item["error_ratio"] for item in iterations]
        result["workspace_bytes"] = runtime.workspace_size
        result["op_timeout_apis"] = runtime.op_timeout_apis
        for name, tensor in zip(COMPACT_INPUT_ORDER, inputs):
            raw, guards[name] = runtime.read(tensor)
            guards[name] = guards[name] and bool(np.array_equal(raw, tensor["raw"], equal_nan=True))
        result["guards"] = guards
        result["iterations"] = iterations
        result["comparisons"] = list(comparisons.values())
        result["error_ratio"] = error_ratios(result["comparisons"])
        result["error_ratio_per_iteration"] = [item["error_ratio"] for item in iterations]
        passed = all(guards.values()) and all(
            comparison["passed"] for iteration in iterations for comparison in iteration["comparisons"])
        result["status"] = "Pass" if passed else "Wrong Answer"
        result["stage"] = "complete" if passed else "verification"
    except DeviceError as error:
        result.update(status="Runtime Error", stage=error.stage, acl_status=error.status,
                      error=str(error))
    except Exception as error:
        result.update(status="Harness Error", error=str(error))
    finally:
        if runtime is not None:
            runtime.close()
    return result


def _write_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _digest_files(root: Path, paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def build_identity(build_dir: Path) -> str:
    library = build_dir / "libcust_opapi.so"
    if not library.is_file():
        raise ValueError(f"missing operator library: {library}")
    artifacts = {library}
    for directory in (build_dir / "op_host", build_dir / "tmp/vendors/custom"):
        artifacts.update(path for path in directory.rglob("*")
                         if path.is_file() and path.suffix in (".so", ".o", ".json", ".ini"))
    return _digest_files(build_dir, artifacts)


def fixture_identity(case_dir: Path) -> str:
    return _digest_files(case_dir, (path for path in case_dir.rglob("*") if path.is_file()))


def runtime_environment(build_dir: Path) -> dict:
    env = dict(os.environ)
    candidates = [
        env.get("ASCEND_HOME_PATH"), env.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/cann-8.5.0", "/home/developer/Ascend/cann-8.5.2",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    toolkit = next((Path(value).resolve() for value in candidates
                    if value and (Path(value) / "include/acl/acl.h").is_file()), None)
    if toolkit is None:
        raise ValueError("CANN toolkit not found")
    paths = [build_dir, build_dir / "op_host", toolkit / "lib64",
             Path("/usr/local/Ascend/driver/lib64/driver"),
             Path("/usr/local/Ascend/driver/lib64/common")]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, paths)) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["ASCEND_CUSTOM_OPP_PATH"] = str(build_dir / "tmp/vendors/custom")
    env["ASCEND_OPP_PATH"] = str(toolkit / "opp")
    env["ASCEND_HOME_PATH"] = env["ASCEND_TOOLKIT_HOME"] = str(toolkit)
    env["PATH"] = str(toolkit / "bin") + ":" + env.get("PATH", "")
    return env


def _execute_process(command, log_path: Path, environment: dict, timeout: float):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return process.returncode, True
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def run_session(case_dir, build_dir, output, *, profiled, repeat, warmup, device,
                timeout, op_wait_seconds, guard_bytes):
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = output / "runner.json"
    command = [
        sys.executable, "-m", "tests.st.scripts.run_device", "--case-dir", str(case_dir),
        "--build-dir", str(build_dir), "--actual-dir", str(output / "actual"),
        "--report", str(report), "--repeat", str(repeat),
        "--warmup", str(0 if profiled else warmup), "--device", str(device),
        "--op-wait-seconds", str(op_wait_seconds),
        "--guard-bytes", str(guard_bytes),
    ]
    profile = output / "profile"
    if profiled:
        profile.mkdir(mode=0o700)
        command = ["msprof", "op", "--output=" + str(profile),
                   "--warm-up=" + str(warmup), "--launch-count=" + str(repeat),
                   "--aic-metrics=BasicInfo", *command]
    process_error = None
    try:
        returncode, timed_out = _execute_process(
            command, output / "process.log", runtime_environment(build_dir), timeout)
    except (OSError, ValueError) as error:
        returncode, timed_out, process_error = None, False, str(error)
    try:
        result = json.loads(report.read_text())
        result.update(runner_report=str(report), runner_report_sha256=file_digest(report),
                      execution_status=result["status"])
    except (OSError, KeyError, ValueError) as error:
        result = {"case_id": case_dir.name, "status": "Harness Error", "stage": "runner_report",
                  "error": str(error), "iterations": [], "comparisons": [], "guards": {}}
    result.update(process_returncode=returncode, timed_out=timed_out, process_error=process_error,
                  command=command, process_log=str(output / "process.log"), repeat=repeat,
                  warmup=warmup, profile_dir=str(profile), profile_records=read_profile_records(profile))
    return result


def run_sessions(case_dir, build_dir, actual_dir, *, sessions=1, profiled=False,
                repeat=5, warmup=5, device=0, timeout=120,
                op_wait_seconds=DEFAULT_OP_WAIT_SECONDS, guard_bytes=64):
    if min(sessions, repeat) < 1 or warmup < 0 or op_wait_seconds < 0 or guard_bytes < 0:
        raise ValueError("sessions/repeat must be positive; warmup/timeout must be nonnegative")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("process timeout must be finite and positive")
    metadata = validate_fixture(case_dir)
    before = {"build": build_identity(build_dir), "fixture": fixture_identity(case_dir)}
    actual_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=actual_dir)).resolve()
    result = {"schema_version": 1, "kind": "device_sessions",
              "case_id": metadata.get("case_id", metadata.get("name", case_dir.name)),
              "profiled": profiled, "session_count": sessions, "repeat": repeat,
              "warmup": warmup, "device": device, "identity": before,
              "build_dir": str(build_dir), "case_dir": str(case_dir),
              "run_dir": str(root), "sessions": []}
    for index in range(sessions):
        session = run_session(case_dir, build_dir, root / f"session-{index + 1}",
                              profiled=profiled, repeat=repeat, warmup=warmup, device=device,
                              timeout=timeout, op_wait_seconds=op_wait_seconds,
                              guard_bytes=guard_bytes)
        result["sessions"].append(session)
        _write_json(root / "result.json", summarize_run_report(result))
    result = summarize_run_report(result)
    if before != {"build": build_identity(build_dir), "fixture": fixture_identity(case_dir)}:
        result.update(status="Harness Error", stage="identity",
                      errors=[*result["errors"], "build or fixture changed during execution"])
    _write_json(root / "result.json", result)
    return result


def merge_mock_results(previous, identity, fresh, case_ids):
    """Merge a partial local rerun only when inputs, binary and protocol match."""
    if previous is not None and (previous.get("kind") != "mock_suite"
                                 or previous.get("identity") != identity):
        raise ValueError("report identity differs; use a separate --report path")
    kept = {case["case_id"]: case for case in (previous or {}).get("cases", [])}
    kept.update({case["case_id"]: case for case in fresh})
    if set(kept) - set(case_ids):
        raise ValueError("report contains a case outside the Mock suite")
    cases = [kept[name] for name in case_ids if name in kept]
    return {"schema_version": 1, "kind": "mock_suite", "identity": identity,
            "case_ids": list(case_ids), "cases": cases, "complete": len(cases) == len(case_ids),
            "passed": bool(cases) and all(case["status"] == "Pass" for case in cases)}


def run_mock_suite(build_dir, actual_dir, report_path, *, names=(), **protocol):
    from tests.common.case_matrix import CASES_DIR, MOCK_CASES, case_by_id
    from tests.st.scripts.compact_fixture import make_fixture

    ids = tuple(case["id"] for case in MOCK_CASES)
    selected = tuple(names) or ids
    if len(set(selected)) != len(selected) or set(selected) - set(ids):
        raise ValueError("select distinct Case IDs from --suite mock")
    identity = {"build": build_identity(build_dir), "catalog": file_digest(CASES_DIR / "mock.json"),
                "tests": _digest_files(ROOT, (path for path in (ROOT / "tests").rglob("*.py"))),
                "protocol": protocol}
    previous = json.loads(report_path.read_text()) if report_path.is_file() else None
    merge_mock_results(previous, identity, [], ids)
    if previous is not None:
        previous = summarize_run_report(previous)
    actual_dir.mkdir(parents=True, exist_ok=True)
    fixtures = Path(tempfile.mkdtemp(prefix="fixtures-", dir=actual_dir)).resolve()
    fresh = []
    for name in selected:
        _, spec = case_by_id(name, "mock")
        fixture = make_fixture(spec, fixtures)
        result = run_sessions(fixture, build_dir, actual_dir / name, **protocol)
        fresh.append(result)
        combined = merge_mock_results(previous, identity, fresh, ids)
        _write_json(report_path, combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-dir", type=Path)
    selection.add_argument("--suite", choices=("mock",))
    parser.add_argument("--case", action="append", default=[], help="Mock Case ID; repeatable")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--actual-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--profile", action="store_true", help="measure independent msprof sessions")
    parser.add_argument("--timeout", type=float, default=120, help="process timeout per session")
    parser.add_argument("--device-lock", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--op-wait-seconds", type=int, default=DEFAULT_OP_WAIT_SECONDS,
                        help="operator timeout in seconds; 0 leaves runtime defaults")
    parser.add_argument("--guard-bytes", type=int, default=64,
                        help="guard width around tensors; 0 uses unshifted buffers for timing")
    args = parser.parse_args()
    args.warmup = args.warmup if args.warmup is not None else (5 if args.profile else 0)
    args.repeat = args.repeat if args.repeat is not None else (5 if args.profile else 1)
    if (args.warmup < 0 or args.repeat < 1 or args.sessions < 1 or
            args.op_wait_seconds < 0 or args.guard_bytes < 0):
        parser.error("warmup/timeout must be nonnegative and repeat must be positive")
    if args.case and not args.suite:
        parser.error("--case requires --suite mock")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.profile and (args.repeat > 5000 or args.warmup > 500):
        parser.error("msprof supports repeat <= 5000 and warmup <= 500")
    if args.suite or args.profile or args.sessions > 1:
        protocol = dict(sessions=args.sessions, profiled=args.profile, repeat=args.repeat,
                        warmup=args.warmup, device=args.device, timeout=args.timeout,
                        op_wait_seconds=args.op_wait_seconds, guard_bytes=args.guard_bytes)
        lock_path = args.device_lock or Path(f"/tmp/cannjudge/device-{args.device}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.suite:
                result = run_mock_suite(args.build_dir.resolve(), args.actual_dir.resolve(),
                                        args.report.resolve(), names=args.case, **protocol)
            else:
                result = run_sessions(args.case_dir.resolve(), args.build_dir.resolve(),
                                      args.actual_dir.resolve(), **protocol)
                _write_json(args.report, result)
        passed = result.get("passed", result.get("status") == "Pass")
        print(json.dumps({"report": str(args.report), "passed": passed}))
        return 0 if passed else 1
    result = run(args.case_dir.resolve(), args.build_dir.resolve(), args.actual_dir.resolve(),
                 args.warmup, args.repeat, args.device, args.op_wait_seconds,
                 args.guard_bytes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in
        ("case_id", "status", "stage", "acl_status")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "Pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
