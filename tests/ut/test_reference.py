from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.common.case_matrix import NUMERIC_CASES
from tests.common.reference import (
    actual_lengths,
    make_inputs,
    reference,
    result_stats,
)


CASES = {case["id"]: case for case in NUMERIC_CASES}


def _assert_probabilities(result, spec):
    actual_q, _ = actual_lengths(spec)
    for batch_index, query_length in enumerate(actual_q):
        for query_index in range(result.valid_mask.shape[1]):
            has_support = bool(result.valid_mask[batch_index, query_index].any())
            p_sum = np.sum(result.target_probability[batch_index, query_index], dtype=np.float32)
            q_sum = np.sum(result.index_probability[batch_index, query_index], dtype=np.float32)
            if query_index < query_length and has_support:
                np.testing.assert_allclose(p_sum, 1.0, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(q_sum, 1.0, rtol=2e-5, atol=2e-6)
            else:
                assert p_sum == 0.0
                assert q_sum == 0.0


@pytest.mark.parametrize("spec", NUMERIC_CASES, ids=lambda case: case["id"])
def test_numeric_case_reference_is_finite_normalized_and_has_contract_shapes(spec):
    inputs = make_inputs(spec)
    result = reference(inputs, spec)
    shape = spec["shape"]
    b, s1, s2 = shape["B"], shape["S1"], shape["S2"]
    n1, nidx1 = shape["N1"], shape["Nidx1"]
    assert result.d_query_index.shape == (b, s1, nidx1, 128)
    assert result.d_key_index.shape == (b, s2, 1, 128)
    assert result.d_weights.shape == (b, s1, nidx1)
    assert result.loss.shape == (1,)
    assert result.loss.dtype == np.float32
    assert result.softmax_max.shape == (b, n1, s1, 1)
    assert result.softmax_sum.shape == (b, n1, s1, 1)
    assert result.softmax_max_index.shape == (b, 1, s1)
    assert result.softmax_sum_index.shape == (b, 1, s1)
    for output in (result.d_query_index, result.d_key_index, result.d_weights, result.loss):
        assert np.isfinite(output).all()
    _assert_probabilities(result, spec)


@pytest.mark.parametrize("spec", NUMERIC_CASES, ids=lambda case: case["id"])
def test_supplied_softmax_statistics_reconstruct_the_same_result(spec):
    inputs = make_inputs(spec)
    direct = reference(inputs, spec)
    reconstructed = reference(inputs, spec, supplied_stats=result_stats(direct))
    np.testing.assert_allclose(reconstructed.target_probability, direct.target_probability, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(reconstructed.index_probability, direct.index_probability, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(reconstructed.d_query_index, direct.d_query_index, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(reconstructed.d_key_index, direct.d_key_index, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(reconstructed.d_weights, direct.d_weights, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(reconstructed.loss, direct.loss, rtol=3e-6, atol=3e-7)


def test_documented_example_matches_published_loss():
    spec = CASES["documented_example"]
    result = reference(make_inputs(spec), spec)
    np.testing.assert_allclose(result.loss[0], 0.4409, rtol=0.0, atol=5e-5)


def test_uniform_case_has_zero_kl_and_zero_gradients():
    spec = CASES["smoke_uniform_minimum"]
    result = reference(make_inputs(spec), spec)
    assert result.loss[0] == 0.0
    assert not np.any(result.d_query_index)
    assert not np.any(result.d_key_index)
    assert not np.any(result.d_weights)


def test_right_down_causal_boundaries_for_both_length_orders():
    short_query = CASES["right_down_mask_boundary"]
    result = reference(make_inputs(short_query), short_query)
    assert result.valid_mask[0].sum(axis=1).tolist() == [4, 5, 6, 7]

    long_query = CASES["query_longer_than_key"]
    result = reference(make_inputs(long_query), long_query)
    assert result.valid_mask[0].sum(axis=1).tolist() == [0, 0, 0, 1, 2, 3]
    assert not np.any(result.d_query_index[0, :3])
    assert not np.any(result.d_weights[0, :3])


def test_padding_poison_is_ignored_and_padded_outputs_are_zero():
    spec = CASES["variable_lengths_and_padding_poison"]
    poisoned = make_inputs(spec)
    cleaned = {name: value.copy() for name, value in poisoned.items()}
    actual_q, actual_k = actual_lengths(spec)
    for batch_index in range(spec["shape"]["B"]):
        aq, ak = int(actual_q[batch_index]), int(actual_k[batch_index])
        for name in ("query", "query_index", "weights", "query_rope"):
            cleaned[name][batch_index, aq:] = 0
        for name in ("key", "key_index", "key_rope"):
            cleaned[name][batch_index, ak:] = 0

    poison_result = reference(poisoned, spec)
    clean_result = reference(cleaned, spec)
    for name in ("d_query_index", "d_key_index", "d_weights", "loss"):
        np.testing.assert_array_equal(getattr(poison_result, name), getattr(clean_result, name))
    for batch_index in range(spec["shape"]["B"]):
        aq, ak = int(actual_q[batch_index]), int(actual_k[batch_index])
        assert not np.any(poison_result.d_query_index[batch_index, aq:])
        assert not np.any(poison_result.d_weights[batch_index, aq:])
        assert not np.any(poison_result.d_key_index[batch_index, ak:])


def test_dkey_is_the_sum_of_all_query_contributions():
    spec = CASES["dkey_cross_query_accumulation"]
    result = reference(make_inputs(spec), spec)
    summed = np.sum(result.d_key_index_per_query, axis=1, dtype=np.float32)
    np.testing.assert_allclose(summed, result.d_key_index[:, :, 0], rtol=2e-6, atol=2e-7)
    assert np.count_nonzero(np.linalg.norm(result.d_key_index_per_query, axis=-1)) > spec["shape"]["S2"]


def test_relu_zero_uses_zero_subgradient():
    spec = CASES["relu_zero_and_signs"]
    result = reference(make_inputs(spec), spec)
    assert not np.any(result.d_key_index_per_query[:, :, 0])


def test_zero_weights_stop_index_gradients_but_not_weight_gradient():
    spec = CASES["zero_weights_chain_rule"]
    result = reference(make_inputs(spec), spec)
    assert result.loss[0] > 0
    assert not np.any(result.d_query_index)
    assert not np.any(result.d_key_index)
    assert np.any(result.d_weights)


def test_probability_tail_case_hits_float32_zero_boundary_with_finite_kl():
    spec = CASES["probability_tail_stability"]
    result = reference(make_inputs(spec), spec)
    active = result.valid_mask
    assert np.any(result.target_probability[active] == 0)
    assert np.any(result.index_probability[active] == 0)
    assert np.isfinite(result.loss).all()


def test_nonzero_rope_changes_target_distribution():
    spec = CASES["rope_dominates_main_score"]
    inputs = make_inputs(spec)
    with_rope = reference(inputs, spec)
    without_rope_inputs = {name: value.copy() for name, value in inputs.items()}
    without_rope_inputs["query_rope"].fill(0)
    without_rope_inputs["key_rope"].fill(0)
    without_rope = reference(without_rope_inputs, spec)
    assert np.max(np.abs(with_rope.target_probability - without_rope.target_probability)) > 1e-3
    assert not np.allclose(with_rope.loss, without_rope.loss)


def test_equivalent_shifted_forward_statistics_are_consumed_correctly():
    spec = CASES["custom_scale_value"]
    inputs = make_inputs(spec)
    direct = reference(inputs, spec)
    stats = {name: value.copy() for name, value in result_stats(direct).items()}
    main_shift = np.float32(0.75)
    index_shift = np.float32(-0.5)
    active_main = stats["softmax_sum"] > 0
    active_index = stats["softmax_sum_index"] > 0
    stats["softmax_max"][active_main] += main_shift
    stats["softmax_sum"][active_main] *= np.exp(-main_shift).astype(np.float32)
    stats["softmax_max_index"][active_index] += index_shift
    stats["softmax_sum_index"][active_index] *= np.exp(-index_shift).astype(np.float32)
    shifted = reference(inputs, spec, supplied_stats=stats)
    np.testing.assert_allclose(shifted.target_probability, direct.target_probability, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(shifted.index_probability, direct.index_probability, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(shifted.loss, direct.loss, rtol=3e-6, atol=3e-7)


def _numerical_derivative(inputs, spec, tensor_name, index, epsilon=2e-3):
    plus = {name: value.copy() for name, value in inputs.items()}
    minus = {name: value.copy() for name, value in inputs.items()}
    plus[tensor_name][index] += np.float32(epsilon)
    minus[tensor_name][index] -= np.float32(epsilon)
    plus_loss = float(reference(plus, spec).loss[0])
    minus_loss = float(reference(minus, spec).loss[0])
    return (plus_loss - minus_loss) / (2.0 * epsilon)


def test_selected_gradients_match_finite_differences():
    spec = CASES["finite_difference_probe"]
    inputs = make_inputs(spec)
    analytic = reference(inputs, spec)
    probes = (
        ("query_index", analytic.d_query_index),
        ("key_index", analytic.d_key_index),
        ("weights", analytic.d_weights),
    )
    for tensor_name, gradient in probes:
        index = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
        numerical = _numerical_derivative(inputs, spec, tensor_name, index)
        expected = float(gradient[index])
        np.testing.assert_allclose(numerical, expected, rtol=4e-2, atol=2e-4)


def test_loss_reduction_is_a_sum_not_a_mean():
    base_spec = CASES["finite_difference_probe"]
    base_inputs = make_inputs(base_spec)
    base = reference(base_inputs, base_spec)
    doubled_spec = copy.deepcopy(base_spec)
    doubled_spec["id"] = "doubled_for_loss_sum"
    doubled_spec["shape"]["B"] = 2
    doubled_inputs = {name: np.repeat(value, 2, axis=0) for name, value in base_inputs.items()}
    doubled = reference(doubled_inputs, doubled_spec)
    np.testing.assert_allclose(doubled.loss, base.loss * np.float32(2.0), rtol=2e-6, atol=2e-7)
