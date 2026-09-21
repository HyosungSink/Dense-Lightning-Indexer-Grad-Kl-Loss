"""Independent float32 reference model and deterministic test-data helpers.

This module deliberately does not import or call the operator implementation.  It models
the complete ABI described by the problem statement so that it remains useful while the
Host schema is still the compact five-input template.

The statement gives queryRope/keyRope as already materialized tensors but does not spell
out their element-wise fusion.  The isolated convention used here is the common separated
RoPE score form

    main_score = scale * (query @ key.T + query_rope @ key_rope.T)

Changing that convention after evaluator clarification only requires changing
``_main_scores`` and the explicit RoPE tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


D = 128
DR = 64
SUPPORTED_N1 = (32, 64, 128)
SUPPORTED_NIDX1 = (8, 16, 32, 64)
SUPPORTED_INPUT_DTYPES = ("float16", "bfloat16")
SUPPORTED_WEIGHT_DTYPES = ("float16", "bfloat16", "float32")
MAX_SEQUENCE = 128 * 1024
MAX_BATCH = 256
DEFAULT_SCALE = float(1.0 / np.sqrt(D))


@dataclass(frozen=True)
class ReferenceResult:
    d_query_index: np.ndarray
    d_key_index: np.ndarray
    d_weights: np.ndarray
    loss: np.ndarray
    softmax_max: np.ndarray
    softmax_sum: np.ndarray
    softmax_max_index: np.ndarray
    softmax_sum_index: np.ndarray
    target_probability: np.ndarray
    index_probability: np.ndarray
    valid_mask: np.ndarray
    d_key_index_per_query: np.ndarray


def emulate_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16, retaining float32 storage for NumPy math."""

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).copy()
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding_bias
    rounded &= np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def quantize(values: np.ndarray, dtype: str) -> np.ndarray:
    """Return logical values rounded to the requested external dtype."""

    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        return emulate_bfloat16(values)
    if dtype == "float32":
        return values.copy()
    raise ValueError(f"unsupported dtype: {dtype}")


def bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Encode logical bfloat16 values as raw uint16 words for portable ST files."""

    rounded = emulate_bfloat16(values)
    return (rounded.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def decode_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _shape(spec: Mapping[str, Any], name: str, default: int) -> int:
    return int(spec.get("shape", {}).get(name, default))


def validate_case_spec(spec: Mapping[str, Any]) -> None:
    """Validate one full-ABI case specification against documented constraints."""

    b = _shape(spec, "B", 1)
    s1 = _shape(spec, "S1", 1)
    s2 = _shape(spec, "S2", 1)
    n1 = _shape(spec, "N1", 32)
    n2 = _shape(spec, "N2", n1)
    nidx1 = _shape(spec, "Nidx1", 8)
    nidx2 = _shape(spec, "Nidx2", 1)
    d = _shape(spec, "D", D)
    dr = _shape(spec, "Dr", DR)

    if not 1 <= b <= MAX_BATCH:
        raise ValueError("B must be in [1, 256]")
    if not 1 <= s1 <= MAX_SEQUENCE:
        raise ValueError("S1 must be in [1, 128K]")
    if not 1 <= s2 <= MAX_SEQUENCE:
        raise ValueError("S2 must be in [1, 128K]")
    if n1 not in SUPPORTED_N1:
        raise ValueError("N1 must be one of 32, 64, 128")
    if n2 != n1:
        raise ValueError("N2 must equal N1")
    if nidx1 not in SUPPORTED_NIDX1:
        raise ValueError("Nidx1 must be one of 8, 16, 32, 64")
    if nidx2 != 1:
        raise ValueError("Nidx2 must equal 1")
    if d != D:
        raise ValueError("D must equal 128")
    if dr != DR:
        raise ValueError("Dr must equal 64")

    tensor_dtypes = spec.get("tensor_dtypes")
    if tensor_dtypes is None:
        input_dtype = str(spec.get("input_dtype", "float16"))
        tensor_dtypes = {name: input_dtype for name in ("query", "key", "query_index", "key_index")}
    else:
        tensor_dtypes = dict(tensor_dtypes)
    if set(tensor_dtypes) != {"query", "key", "query_index", "key_index"}:
        raise ValueError("all four tensor dtypes must be specified")
    if any(dtype not in SUPPORTED_INPUT_DTYPES for dtype in tensor_dtypes.values()):
        raise ValueError("main/index tensors must be float16 or bfloat16")
    if len(set(tensor_dtypes.values())) != 1:
        raise ValueError("query/key/queryIndex/keyIndex dtypes must match")

    input_dtype = next(iter(tensor_dtypes.values()))
    weights_dtype = str(spec.get("weights_dtype", input_dtype))
    if weights_dtype not in SUPPORTED_WEIGHT_DTYPES:
        raise ValueError("weights dtype must be float16, bfloat16, or float32")
    if weights_dtype != "float32" and weights_dtype != input_dtype:
        raise ValueError("non-float32 weights dtype must match the other tensors")

    if spec.get("layout", "BSND") != "BSND":
        raise ValueError("layout must be BSND")
    if int(spec.get("sparse_mode", 3)) != 3:
        raise ValueError("sparseMode must equal 3")
    max_token = 2**63 - 1
    if int(spec.get("pre_tokens", max_token)) != max_token:
        raise ValueError("pre_tokens must retain its default maximum")
    if int(spec.get("next_tokens", max_token)) != max_token:
        raise ValueError("next_tokens must retain its default maximum")

    for field, limit in (("actual_seq_lengths_query", s1), ("actual_seq_lengths_key", s2)):
        lengths = spec.get(field)
        if lengths is None:
            continue
        if len(lengths) != b:
            raise ValueError(f"{field} must contain B entries")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in lengths):
            raise ValueError(f"{field} must contain int64-compatible integers")
        if any(value < 1 or value > limit for value in lengths):
            raise ValueError(f"{field} values must be in [1, padded sequence length]")


def actual_lengths(spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    b = _shape(spec, "B", 1)
    s1 = _shape(spec, "S1", 1)
    s2 = _shape(spec, "S2", 1)
    actual_q = spec.get("actual_seq_lengths_query")
    actual_k = spec.get("actual_seq_lengths_key")
    q = np.full(b, s1, dtype=np.int64) if actual_q is None else np.asarray(actual_q, dtype=np.int64)
    k = np.full(b, s2, dtype=np.int64) if actual_k is None else np.asarray(actual_k, dtype=np.int64)
    return q, k


def right_down_causal_mask(
    batch: int,
    s1: int,
    s2: int,
    actual_q: np.ndarray,
    actual_k: np.ndarray,
) -> np.ndarray:
    """Build a per-batch right-aligned causal mask.

    The documented formula is generalized to variable lengths as
    ``j <= i + actual_k[b] - actual_q[b]``.  When S1>S2, early valid query
    rows can consequently have no visible key.  The reference treats such rows as
    inactive so every reported probability/loss remains finite.
    """

    mask = np.zeros((batch, s1, s2), dtype=np.bool_)
    for b in range(batch):
        for i in range(int(actual_q[b])):
            visible = min(int(actual_k[b]), i + int(actual_k[b]) - int(actual_q[b]) + 1)
            if visible > 0:
                mask[b, i, :visible] = True
    return mask


def _random_array(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def make_inputs(spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Materialize deterministic, dtype-quantized logical inputs for a numeric case."""

    validate_case_spec(spec)
    b = _shape(spec, "B", 1)
    s1 = _shape(spec, "S1", 1)
    s2 = _shape(spec, "S2", 1)
    n1 = _shape(spec, "N1", 32)
    nidx1 = _shape(spec, "Nidx1", 8)
    input_dtype = str(spec.get("input_dtype", "float16"))
    weights_dtype = str(spec.get("weights_dtype", input_dtype))
    rng = np.random.default_rng(int(spec.get("seed", 0)))

    arrays = {
        "query": _random_array(rng, (b, s1, n1, D), 0.18),
        "key": _random_array(rng, (b, s2, n1, D), 0.18),
        "query_index": _random_array(rng, (b, s1, nidx1, D), 0.12),
        "key_index": _random_array(rng, (b, s2, 1, D), 0.12),
        "weights": _random_array(rng, (b, s1, nidx1), 0.2),
        "query_rope": np.zeros((b, s1, n1, DR), dtype=np.float32),
        "key_rope": np.zeros((b, s2, n1, DR), dtype=np.float32),
    }

    pattern = str(spec.get("pattern", "random"))
    if pattern == "zeros":
        for array in arrays.values():
            array.fill(0.0)
    elif pattern == "documented_example":
        arrays["query"].fill(0.1)
        arrays["query_index"].fill(0.1)
        arrays["weights"].fill(0.1)
        for j in range(s2):
            arrays["key"][:, j].fill((j + 1) * 0.1)
            arrays["key_index"][:, j].fill((j + 1) * 0.1)
    elif pattern == "extreme_logits":
        arrays["query"].fill(0.5)
        arrays["query_index"].fill(0.5)
        arrays["weights"].fill(0.25)
        # A score span above 2*90 forces real float32 exp underflow at the tail.
        # Reverse indexer positions so p and q disagree instead of both collapsing
        # onto the same key, which also stresses finite log-softmax KL evaluation.
        positions = np.linspace(-16.0, 16.0, s2, dtype=np.float32)
        for j, value in enumerate(positions):
            arrays["key"][:, j].fill(value)
            arrays["key_index"][:, j].fill(-value)
    elif pattern == "relu_boundary":
        arrays["query_index"].fill(0.0)
        arrays["key_index"].fill(0.0)
        arrays["query_index"][..., 0] = 1.0
        arrays["query_index"][..., 1] = 1.0
        prototypes = ((1.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (1.0, -0.999))
        for j in range(s2):
            first, second = prototypes[j % len(prototypes)]
            arrays["key_index"][:, j, 0, 0] = first
            arrays["key_index"][:, j, 0, 1] = second
    elif pattern == "zero_weights":
        arrays["weights"].fill(0.0)
    elif pattern not in ("random", "padding_poison"):
        raise ValueError(f"unknown input pattern: {pattern}")

    rope_mode = str(spec.get("rope", "zero"))
    if rope_mode == "random":
        arrays["query_rope"] = _random_array(rng, (b, s1, n1, DR), 0.15)
        arrays["key_rope"] = _random_array(rng, (b, s2, n1, DR), 0.15)
    elif rope_mode == "dominant":
        arrays["query"].fill(0.0)
        arrays["key"].fill(0.0)
        arrays["query_rope"].fill(0.5)
        for j, value in enumerate(np.linspace(-1.0, 1.0, s2, dtype=np.float32)):
            arrays["key_rope"][:, j].fill(value)
    elif rope_mode != "zero":
        raise ValueError(f"unknown RoPE mode: {rope_mode}")

    actual_q, actual_k = actual_lengths(spec)
    if pattern == "padding_poison":
        poison = np.float32(2048.0)
        for batch_index in range(b):
            aq = int(actual_q[batch_index])
            ak = int(actual_k[batch_index])
            arrays["query"][batch_index, aq:].fill(poison)
            arrays["query_index"][batch_index, aq:].fill(poison)
            arrays["weights"][batch_index, aq:].fill(poison)
            arrays["query_rope"][batch_index, aq:].fill(poison)
            arrays["key"][batch_index, ak:].fill(poison)
            arrays["key_index"][batch_index, ak:].fill(poison)
            arrays["key_rope"][batch_index, ak:].fill(poison)

    for name in ("query", "key", "query_index", "key_index", "query_rope", "key_rope"):
        arrays[name] = quantize(arrays[name], input_dtype)
    arrays["weights"] = quantize(arrays["weights"], weights_dtype)
    return arrays


def _main_scores(
    query: np.ndarray,
    key: np.ndarray,
    query_rope: np.ndarray,
    key_rope: np.ndarray,
    scale: float,
) -> np.ndarray:
    content = np.einsum("nd,knd->nk", query, key, dtype=np.float32, optimize=True)
    rope = np.einsum("nr,knr->nk", query_rope, key_rope, dtype=np.float32, optimize=True)
    return np.asarray((content + rope) * np.float32(scale), dtype=np.float32)


def reference(
    inputs: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    supplied_stats: Mapping[str, np.ndarray] | None = None,
) -> ReferenceResult:
    """Evaluate loss, gradients, masks, probabilities, and forward statistics."""

    validate_case_spec(spec)
    bsz = _shape(spec, "B", 1)
    s1 = _shape(spec, "S1", 1)
    s2 = _shape(spec, "S2", 1)
    n1 = _shape(spec, "N1", 32)
    nidx1 = _shape(spec, "Nidx1", 8)
    scale = float(spec.get("scale_value", DEFAULT_SCALE))
    actual_q, actual_k = actual_lengths(spec)
    mask = right_down_causal_mask(bsz, s1, s2, actual_q, actual_k)

    q = np.asarray(inputs["query"], dtype=np.float32)
    k = np.asarray(inputs["key"], dtype=np.float32)
    qi = np.asarray(inputs["query_index"], dtype=np.float32)
    ki = np.asarray(inputs["key_index"], dtype=np.float32)
    weights = np.asarray(inputs["weights"], dtype=np.float32)
    qr = np.asarray(inputs["query_rope"], dtype=np.float32)
    kr = np.asarray(inputs["key_rope"], dtype=np.float32)

    d_qi = np.zeros((bsz, s1, nidx1, D), dtype=np.float32)
    d_ki = np.zeros((bsz, s2, 1, D), dtype=np.float32)
    d_weights = np.zeros((bsz, s1, nidx1), dtype=np.float32)
    d_ki_per_query = np.zeros((bsz, s1, s2, D), dtype=np.float32)
    p_all = np.zeros((bsz, s1, s2), dtype=np.float32)
    pred_all = np.zeros((bsz, s1, s2), dtype=np.float32)
    main_max = np.full((bsz, n1, s1, 1), -np.inf, dtype=np.float32)
    main_sum = np.zeros((bsz, n1, s1, 1), dtype=np.float32)
    index_max = np.full((bsz, 1, s1), -np.inf, dtype=np.float32)
    index_sum = np.zeros((bsz, 1, s1), dtype=np.float32)
    loss = np.float32(0.0)

    for batch_index in range(bsz):
        for query_index in range(int(actual_q[batch_index])):
            key_positions = np.flatnonzero(mask[batch_index, query_index])
            if key_positions.size == 0:
                continue
            key_count = int(key_positions.size)
            main_scores = _main_scores(
                q[batch_index, query_index],
                k[batch_index, :key_count],
                qr[batch_index, query_index],
                kr[batch_index, :key_count],
                scale,
            )
            index_similarity = np.einsum(
                "hd,kd->hk",
                qi[batch_index, query_index],
                ki[batch_index, :key_count, 0],
                dtype=np.float32,
                optimize=True,
            )
            relu_similarity = np.maximum(index_similarity, np.float32(0.0))
            index_scores = np.einsum(
                "h,hk->k",
                weights[batch_index, query_index],
                relu_similarity,
                dtype=np.float32,
                optimize=True,
            )

            computed_main_max = np.max(main_scores, axis=1, keepdims=True)
            computed_main_sum = np.sum(
                np.exp(main_scores - computed_main_max, dtype=np.float32),
                axis=1,
                keepdims=True,
                dtype=np.float32,
            )
            computed_index_max = np.max(index_scores).astype(np.float32)
            computed_index_sum = np.sum(
                np.exp(index_scores - computed_index_max, dtype=np.float32), dtype=np.float32
            ).astype(np.float32)
            main_max[batch_index, :, query_index, 0] = computed_main_max[:, 0]
            main_sum[batch_index, :, query_index, 0] = computed_main_sum[:, 0]
            index_max[batch_index, 0, query_index] = computed_index_max
            index_sum[batch_index, 0, query_index] = computed_index_sum

            if supplied_stats is None:
                used_main_max = computed_main_max
                used_main_sum = computed_main_sum
                used_index_max = computed_index_max
                used_index_sum = computed_index_sum
            else:
                used_main_max = np.asarray(
                    supplied_stats["softmax_max"][batch_index, :, query_index, 0], dtype=np.float32
                )[:, None]
                used_main_sum = np.asarray(
                    supplied_stats["softmax_sum"][batch_index, :, query_index, 0], dtype=np.float32
                )[:, None]
                used_index_max = np.float32(
                    supplied_stats["softmax_max_index"][batch_index, 0, query_index]
                )
                used_index_sum = np.float32(
                    supplied_stats["softmax_sum_index"][batch_index, 0, query_index]
                )

            main_probability = np.exp(main_scores - used_main_max, dtype=np.float32) / used_main_sum
            target_raw = np.sum(main_probability, axis=0, dtype=np.float32)
            target = target_raw / np.sum(target_raw, dtype=np.float32)
            predicted = np.exp(index_scores - used_index_max, dtype=np.float32) / used_index_sum
            p_all[batch_index, query_index, :key_count] = target
            pred_all[batch_index, query_index, :key_count] = predicted

            positive_target = target > 0
            log_predicted = index_scores - (
                np.float32(used_index_max) + np.log(np.float32(used_index_sum)).astype(np.float32)
            )
            row_loss = np.sum(
                target[positive_target]
                * (np.log(target[positive_target]).astype(np.float32) - log_predicted[positive_target]),
                dtype=np.float32,
            )
            loss = np.float32(loss + row_loss)

            d_index_score = predicted - target
            d_weights[batch_index, query_index] = np.einsum(
                "k,hk->h", d_index_score, relu_similarity, dtype=np.float32, optimize=True
            )
            d_similarity = (
                weights[batch_index, query_index, :, None]
                * d_index_score[None, :]
                * (index_similarity > 0)
            ).astype(np.float32)
            d_qi[batch_index, query_index] = np.einsum(
                "hk,kd->hd",
                d_similarity,
                ki[batch_index, :key_count, 0],
                dtype=np.float32,
                optimize=True,
            )
            key_contribution = np.einsum(
                "hk,hd->kd",
                d_similarity,
                qi[batch_index, query_index],
                dtype=np.float32,
                optimize=True,
            )
            d_ki_per_query[batch_index, query_index, :key_count] = key_contribution
            d_ki[batch_index, :key_count, 0] += key_contribution

    return ReferenceResult(
        d_query_index=d_qi,
        d_key_index=d_ki,
        d_weights=d_weights,
        loss=np.asarray([loss], dtype=np.float32),
        softmax_max=main_max,
        softmax_sum=main_sum,
        softmax_max_index=index_max,
        softmax_sum_index=index_sum,
        target_probability=p_all,
        index_probability=pred_all,
        valid_mask=mask,
        d_key_index_per_query=d_ki_per_query,
    )


def result_stats(result: ReferenceResult) -> dict[str, np.ndarray]:
    return {
        "softmax_max": result.softmax_max,
        "softmax_sum": result.softmax_sum,
        "softmax_max_index": result.softmax_max_index,
        "softmax_sum_index": result.softmax_sum_index,
    }


def encode_external(values: np.ndarray, dtype: str) -> tuple[np.ndarray, str]:
    """Encode an external tensor and return its storage-description string."""

    if dtype == "float16":
        return np.asarray(values, dtype=np.float16), "float16"
    if dtype == "float32":
        return np.asarray(values, dtype=np.float32), "float32"
    if dtype == "bfloat16":
        return bfloat16_bits(values), "bfloat16_raw_uint16"
    raise ValueError(f"unsupported external dtype: {dtype}")
