"""Generate fixed compact-ABI fixtures and their float32 reference outputs.

Compact regressions and Mock Cases declare their own layout, causal and loss
conventions. They do not replace the formal full-ABI test contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.common.case_matrix import suite_for_case
from tests.common.reference import encode_external, quantize
from tests.st.scripts.gen_data import _comparison, _tensor_metadata, COMPACT_INPUT_ORDER, OUTPUT_NAMES


def compact_reference(inputs, scale, causal=True, loss_epsilon=None):
    q, k, qi, ki, weights = [np.asarray(value, dtype=np.float32) for value in inputs]
    batch, query_length, _, _ = q.shape
    key_length = k.shape[1]
    dq, dk, dw = [np.zeros_like(value) for value in (qi, ki, weights)]
    loss = np.float32(0)
    for b in range(batch):
        for row in range(query_length):
            visible = max(0, min(key_length, row + key_length - query_length + 1)) if causal else key_length
            if visible == 0:
                continue
            scores = np.einsum("hd,khd->hk", q[b, row], k[b, :visible], optimize=True) * np.float32(scale)
            probability = np.exp(scores - scores.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            target = probability.sum(axis=0)
            target /= target.sum()
            similarity = qi[b, row] @ ki[b, :visible, 0].T
            positive = np.maximum(similarity, np.float32(0))
            logits = weights[b, row] @ positive
            shifted = logits - logits.max()
            prediction = np.exp(shifted)
            normalizer = prediction.sum()
            prediction /= normalizer
            active = target > 0
            if loss_epsilon is None:
                loss += np.sum(target[active] * (np.log(target[active]) -
                                shifted[active] + np.log(normalizer)), dtype=np.float32)
            else:
                epsilon = np.float32(loss_epsilon)
                if not np.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("loss_epsilon must be finite and positive")
                loss += np.sum(target * np.log((target + epsilon) / (prediction + epsilon)), dtype=np.float32)
            delta = prediction - target
            dw[b, row] = positive @ delta
            ds = weights[b, row, :, None] * delta * (similarity > 0)
            dq[b, row] = ds @ ki[b, :visible, 0]
            dk[b, :visible, 0] += ds.T @ qi[b, row]
    return dq, dk, dw, np.asarray([loss], dtype=np.float32)


def make_fixture(spec: dict, output_root: Path):
    batch, s1, s2, main_heads, index_heads, dim = spec["shape"]
    if min(spec["shape"]) <= 0:
        raise ValueError("probe dimensions must be positive")
    dtype = spec.get("dtype", "float16")
    weight_dtype = spec.get("weights_dtype", dtype)
    scale = spec.get("scale", 1 / np.sqrt(dim))
    rng = np.random.default_rng(spec.get("seed", 42))
    shapes = [(batch, s1, main_heads, dim), (batch, s2, main_heads, dim),
              (batch, s1, index_heads, dim), (batch, s2, 1, dim), (batch, s1, index_heads)]
    deviations = spec.get("deviations", (0.18, 0.18, 0.12, 0.12, 0.2))
    distributions = spec.get("distributions", ("normal",) * len(shapes))
    if len(distributions) != len(shapes):
        raise ValueError("one distribution is required for each input")
    inputs = []
    for shape, deviation, distribution in zip(shapes, deviations, distributions):
        if distribution == "normal":
            value = rng.normal(0, deviation, shape)
        elif distribution == "uniform":
            value = rng.uniform(-deviation, deviation, shape)
        else:
            raise ValueError(f"unknown distribution {distribution}")
        inputs.append(value.astype(np.float32))
    pattern = spec.get("pattern", "random")
    if pattern == "documented_example":
        for array in inputs:
            array.fill(0.1)
        for array in (inputs[1], inputs[3]):
            array[:] = np.arange(1, s2 + 1, dtype=np.float32)[None, :, None, None] * 0.1
    elif pattern == "separated_extremes":
        inputs[0].fill(0.5)
        inputs[2].fill(0.5)
        inputs[4].fill(0.25)
        positions = np.linspace(-16.0, 16.0, s2, dtype=np.float32)
        inputs[1][:] = positions[None, :, None, None]
        inputs[3][:] = -positions[None, :, None, None]
    elif pattern == "zeros":
        for array in inputs:
            array.fill(0)
    elif pattern != "random":
        raise ValueError(f"unknown pattern {pattern}")
    if "query_index_prefix" in spec:
        prefix = np.asarray(spec["query_index_prefix"], dtype=np.float32)
        if prefix.ndim != 1 or prefix.size > inputs[2].size:
            raise ValueError("query_index_prefix must fit queryIndex")
        inputs[2].flat[:prefix.size] = prefix
    inputs = [quantize(array, dtype if i < 4 else weight_dtype) for i, array in enumerate(inputs)]
    outputs = list(compact_reference(inputs, scale, spec.get("causal", True), spec.get("loss_epsilon")))
    layout = spec.get("layout", "BSND")
    if layout == "BNSD":
        inputs[:4] = [array.transpose(0, 2, 1, 3) for array in inputs[:4]]
        outputs[:2] = [array.transpose(0, 2, 1, 3) for array in outputs[:2]]
        if spec.get("weight_layout", "BSN") == "BNS":
            inputs[4] = inputs[4].transpose(0, 2, 1)
            outputs[2] = outputs[2].transpose(0, 2, 1)
    elif layout == "SND":
        if batch != 1:
            raise ValueError("SND probes require batch=1")
        inputs = [array[0] for array in inputs]
        outputs[:3] = [array[0] for array in outputs[:3]]
    elif layout != "BSND":
        raise ValueError(f"unknown layout {layout}")
    case_dir = output_root / spec["id"]
    tensors = {}
    for role, names, values in (("input", COMPACT_INPUT_ORDER, inputs), ("golden", OUTPUT_NAMES, outputs)):
        for name, array in zip(names, values):
            logical_dtype = "float32" if name == "loss" else weight_dtype if name in ("weights", "dWeights") else dtype
            raw, storage = encode_external(array, logical_dtype)
            raw = np.ascontiguousarray(raw)
            relative = Path(role) / (name + ".bin")
            path = case_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.tofile(path)
            tensors[name] = _tensor_metadata(raw, logical_dtype, storage, relative, role)
    metadata = {
        "case_id": spec["id"], "abi": "compact_five_inputs", "expected": "success",
        "suite": suite_for_case(spec["id"]), "spec": spec,
        "input_order": COMPACT_INPUT_ORDER, "output_order": OUTPUT_NAMES,
        "attributes": {"scaleValue": scale}, "tensors": tensors,
        "comparisons": {name: _comparison(name, tensors[name]["logical_dtype"]) for name in OUTPUT_NAMES},
    }
    (case_dir / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return case_dir
