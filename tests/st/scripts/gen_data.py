#!/usr/bin/env python3
"""Generate raw ST tensors and runner metadata from the shared case matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.common.case_matrix import (  # noqa: E402
    REFERENCE_CONVENTIONS,
    SUITE_NAMES,
    all_specs,
    case_by_id,
    suite_for_case,
)
from tests.common.reference import (  # noqa: E402
    DEFAULT_SCALE,
    actual_lengths,
    encode_external,
    make_inputs,
    reference,
)


DEFAULT_OUTPUT_DIR = Path("/tmp/dense_lightning_indexer_grad_kl_loss_cases")
DEFAULT_MAX_ELEMENTS = 20_000_000
FORMAL_INPUT_ORDER = (
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
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
)
COMPACT_INPUT_ORDER = ("query", "key", "queryIndex", "keyIndex", "weights")
OUTPUT_NAMES = ("dQueryIndex", "dKeyIndex", "dWeights", "loss")


def _encode(values: np.ndarray, logical_dtype: str) -> tuple[np.ndarray, str]:
    encoded, storage = encode_external(values, logical_dtype)
    return np.ascontiguousarray(encoded), storage


def _tensor_metadata(
    array: np.ndarray,
    logical_dtype: str,
    storage: str,
    relative_path: Path,
    role: str,
) -> dict[str, Any]:
    return {
        "file": relative_path.as_posix(),
        "role": role,
        "shape": list(array.shape),
        "logical_dtype": logical_dtype,
        "storage": storage,
        "storage_dtype": str(array.dtype),
        "numel": int(array.size),
        "bytes": int(array.nbytes),
        "layout": "contiguous_nd",
        "byte_order": "little_endian",
    }


def _comparison(name: str, logical_dtype: str) -> dict[str, Any]:
    if name == "loss":
        return {"rtol": 5.0e-4, "atol": 5.0e-5, "require_finite": True}
    if logical_dtype == "float16":
        return {"rtol": 5.0e-3, "atol": 5.0e-3, "require_finite": True}
    if logical_dtype == "bfloat16":
        return {"rtol": 2.0e-2, "atol": 2.0e-2, "require_finite": True}
    return {"rtol": 2.0e-4, "atol": 2.0e-5, "require_finite": True}


def _numeric_payload(
    spec: dict[str, Any], include_debug: bool
) -> tuple[dict[str, tuple[np.ndarray, str, str]], dict[str, str]]:
    inputs = make_inputs(spec)
    golden = reference(inputs, spec)
    input_dtype = str(spec.get("input_dtype", "float16"))
    weights_dtype = str(spec.get("weights_dtype", input_dtype))
    payload: dict[str, tuple[np.ndarray, str, str]] = {}
    storage_by_name: dict[str, str] = {}

    formal_inputs = {
        "query": (inputs["query"], input_dtype),
        "key": (inputs["key"], input_dtype),
        "queryIndex": (inputs["query_index"], input_dtype),
        "keyIndex": (inputs["key_index"], input_dtype),
        "weights": (inputs["weights"], weights_dtype),
        "queryRope": (inputs["query_rope"], input_dtype),
        "keyRope": (inputs["key_rope"], input_dtype),
    }
    for name, (values, logical_dtype) in formal_inputs.items():
        encoded, storage = _encode(values, logical_dtype)
        payload[name] = (encoded, logical_dtype, "input")
        storage_by_name[name] = storage

    statistics = {
        "softmaxMax": golden.softmax_max,
        "softmaxSum": golden.softmax_sum,
        "softmaxMaxIndex": golden.softmax_max_index,
        "softmaxSumIndex": golden.softmax_sum_index,
    }
    for name, values in statistics.items():
        payload[name] = (np.ascontiguousarray(values, dtype=np.float32), "float32", "input")
        storage_by_name[name] = "float32"

    actual_q, actual_k = actual_lengths(spec)
    if spec.get("actual_seq_lengths_query") is not None:
        payload["actual_seq_lengths_query"] = (
            np.ascontiguousarray(actual_q, dtype=np.int64),
            "int64",
            "input",
        )
        storage_by_name["actual_seq_lengths_query"] = "int64"
    if spec.get("actual_seq_lengths_key") is not None:
        payload["actual_seq_lengths_key"] = (
            np.ascontiguousarray(actual_k, dtype=np.int64),
            "int64",
            "input",
        )
        storage_by_name["actual_seq_lengths_key"] = "int64"

    outputs = {
        "dQueryIndex": (golden.d_query_index, input_dtype),
        "dKeyIndex": (golden.d_key_index, input_dtype),
        "dWeights": (golden.d_weights, weights_dtype),
        "loss": (golden.loss, "float32"),
    }
    for name, (values, logical_dtype) in outputs.items():
        encoded, storage = _encode(values, logical_dtype)
        payload[name] = (encoded, logical_dtype, "golden")
        storage_by_name[name] = storage

    if include_debug:
        debug = {
            "targetProbability": golden.target_probability.astype(np.float32),
            "indexProbability": golden.index_probability.astype(np.float32),
            "validMask": golden.valid_mask,
        }
        for name, values in debug.items():
            logical_dtype = "bool" if values.dtype == np.bool_ else "float32"
            payload[name] = (np.ascontiguousarray(values), logical_dtype, "debug")
            storage_by_name[name] = logical_dtype
    return payload, storage_by_name


def materialize_numeric_case(
    spec: dict[str, Any], output_root: Path, include_debug: bool = False
) -> Path:
    """Write one success case as inputs/*.bin, golden/*.bin, and case.json."""

    case_dir = output_root / spec["id"]
    payload, storage_by_name = _numeric_payload(spec, include_debug)
    tensor_metadata: dict[str, dict[str, Any]] = {}

    for name, (array, logical_dtype, role) in payload.items():
        subdir = {"input": "inputs", "golden": "golden", "debug": "debug"}[role]
        relative_path = Path(subdir) / f"{name}.bin"
        path = case_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        array.tofile(path)
        tensor_metadata[name] = _tensor_metadata(
            array, logical_dtype, storage_by_name[name], relative_path, role
        )
    input_order = [name for name in FORMAL_INPUT_ORDER if name in payload]
    compact_applicable = "compact_schema_bridge" in spec.get("tags", [])

    metadata = {
        "name": spec["id"],
        "suite": suite_for_case(spec["id"]),
        "kind": "numeric",
        "expected": "success",
        "abi": "formal_full_abi",
        "spec": spec,
        "attributes": {
            "scaleValue": float(spec.get("scale_value", DEFAULT_SCALE)),
            "layout": spec.get("layout", "BSND"),
            "sparseMode": int(spec.get("sparse_mode", 3)),
            "pre_tokens": int(spec.get("pre_tokens", 2**63 - 1)),
            "next_tokens": int(spec.get("next_tokens", 2**63 - 1)),
        },
        "input_order": input_order,
        "output_order": list(OUTPUT_NAMES),
        "abi_variants": {
            "formal_full_abi": {
                "applicable": True,
                "input_order": input_order,
                "output_order": list(OUTPUT_NAMES),
            },
            "current_compact_schema": {
                "applicable": compact_applicable,
                "input_order": list(COMPACT_INPUT_ORDER),
                "output_name_map": {
                    "dQueryIndex": "d_query_index",
                    "dKeyIndex": "d_key_index",
                    "dWeights": "d_weights",
                    "loss": "loss",
                },
                "attribute_name": "scale_value",
                "recompute_forward_statistics": True,
                "reason": (
                    "Only full-length, zero-RoPE bridge cases can represent the compact "
                    "schema without discarding documented semantics."
                ),
            },
        },
        "omitted_optional_inputs": [
            name
            for name in ("actual_seq_lengths_query", "actual_seq_lengths_key")
            if spec.get(name) is None
        ],
        "tensors": tensor_metadata,
        "comparisons": {
            name: _comparison(name, tensor_metadata[name]["logical_dtype"])
            for name in OUTPUT_NAMES
        },
        "runner_checks": {
            "prefill_outputs_with_nan": True,
            "all_outputs_must_be_written": True,
            "guard_regions": {
                "enabled": True,
                "bytes_before": 64,
                "bytes_after": 64,
                "canary_byte": "0xA5",
            },
            "masked_and_padded_outputs_must_be_zero": True,
        },
        "reference_conventions": REFERENCE_CONVENTIONS,
        "abi_note": (
            "The current Host template is compact. Use this full-ABI fixture only after "
            "the evaluator interface is confirmed. Cases tagged compact_schema_bridge "
            "also provide a five-input recomputation oracle."
        ),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return case_dir


def materialize_spec_case(group: str, spec: dict[str, Any], output_root: Path) -> Path:
    case_dir = output_root / spec["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    expectation = {
        "shape_only_cases": "shape_or_performance_run",
        "invalid_cases": "rejection",
        "runtime_only_cases": "runtime_check",
        "abi_pending_cases": "abi_confirmation",
    }[group]
    metadata = {
        "name": spec["id"],
        "suite": suite_for_case(spec["id"]),
        "kind": group,
        "expected": expectation,
        "spec": spec,
        "materialized": False,
    }
    (case_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return case_dir


def largest_tensor_numel(spec: dict[str, Any]) -> int:
    shape = spec["shape"]
    if spec.get("kind") == "compact":
        b, s1, s2, n1, nidx1, dim = shape
        return max(b * s1 * n1 * dim, b * s2 * n1 * dim,
                   b * s1 * nidx1 * dim, b * s2 * dim)
    b, s1, s2 = shape["B"], shape["S1"], shape["S2"]
    n1, nidx1 = shape["N1"], shape["Nidx1"]
    return max(
        b * s1 * n1 * 128,
        b * s2 * n1 * 128,
        b * s1 * nidx1 * 128,
        b * s2 * 128,
        b * n1 * s1,
    )


def _print_cases(suite: str) -> None:
    for group, spec in all_specs(suite):
        if group == "numeric_cases":
            shape = spec["shape"]
            shape_text = (
                f"B={shape['B']} S1={shape['S1']} S2={shape['S2']} "
                f"N1={shape['N1']} Nidx1={shape['Nidx1']}"
            )
        elif group == "compact_cases":
            shape_text = f"compact BSND dimensions={spec['shape']}"
        else:
            shape_text = "spec-only"
        print(f"{suite_for_case(spec['id']):11s} {group:18s} {spec['id']:46s} {shape_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", dest="case_ids", help="case ID; repeatable")
    selection.add_argument("--all", action="store_true", help="generate all executable cases in the suite")
    selection.add_argument("--list", action="store_true", help="list matrix without writing")
    parser.add_argument("--suite", choices=("all", *SUITE_NAMES), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-spec-only",
        action="store_true",
        help="with --all, also write shape/invalid/runtime/ABI metadata cases",
    )
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=DEFAULT_MAX_ELEMENTS,
        help="skip numeric cases above this largest-tensor size; 0 disables the cap",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list or (not args.case_ids and not args.all):
        _print_cases(args.suite)
        return 0

    if args.case_ids:
        selected = []
        for case_id in args.case_ids:
            try:
                selected.append(case_by_id(case_id, args.suite))
            except KeyError:
                print(f"unknown case in suite {args.suite}: {case_id}", file=sys.stderr)
                return 2
    else:
        selected = [(group, spec) for group, spec in all_specs(args.suite)
                    if args.include_spec_only or group in ("numeric_cases", "compact_cases")]

    generated = 0
    skipped = 0
    for group, spec in selected:
        if group in ("numeric_cases", "compact_cases"):
            max_numel = largest_tensor_numel(spec)
            if args.max_elements and max_numel > args.max_elements:
                print(
                    f"skipped {spec['id']}: largest tensor {max_numel} > {args.max_elements}"
                )
                skipped += 1
                continue
            if group == "compact_cases":
                from tests.st.scripts.compact_fixture import make_fixture
                path = make_fixture(spec, args.output_dir)
            else:
                path = materialize_numeric_case(spec, args.output_dir, args.include_debug)
        else:
            path = materialize_spec_case(group, spec, args.output_dir)
        print(f"generated {spec['id']} -> {path}")
        generated += 1
    print(f"summary: generated={generated}, skipped={skipped}, output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
