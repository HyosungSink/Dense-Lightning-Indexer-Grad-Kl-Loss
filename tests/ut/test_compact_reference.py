"""Compact-ABI reference invariants, independent of the device kernel."""

from copy import deepcopy

import numpy as np
import pytest

from tests.common.case_matrix import numeric_case_by_id
from tests.common.reference import make_inputs, reference
from tests.st.scripts.compact_fixture import compact_reference


@pytest.mark.parametrize("case_id", ["documented_example", "head_grid_n32_i8", "head_grid_n128_i64"])
def test_compact_reference_matches_full_reference_without_optional_semantics(case_id):
    spec = numeric_case_by_id(case_id)
    inputs = make_inputs(spec)
    full = reference(inputs, spec)
    compact = compact_reference([inputs[name] for name in
        ("query", "key", "query_index", "key_index", "weights")], 1 / np.sqrt(128))
    for actual, expected in zip(compact, (full.d_query_index, full.d_key_index, full.d_weights, full.loss)):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("causal", [False, True])
def test_general_compact_gradient_finite_differences(causal):
    rng = np.random.default_rng(4)
    shapes = [(1, 2, 2, 5), (1, 4, 2, 5), (1, 2, 3, 5), (1, 4, 1, 5), (1, 2, 3)]
    inputs = [rng.normal(0, 0.5, shape).astype(np.float32) for shape in shapes]
    gradients = compact_reference(inputs, 0.4, causal=causal)
    for which, index in [(2, (0, 1, 2, 3)), (3, (0, 2, 0, 4)), (4, (0, 0, 1))]:
        step = 1e-3
        plus, minus = deepcopy(inputs), deepcopy(inputs)
        plus[which][index] += step
        minus[which][index] -= step
        numerical = (compact_reference(plus, 0.4, causal=causal)[3][0] -
                     compact_reference(minus, 0.4, causal=causal)[3][0]) / (2 * step)
        np.testing.assert_allclose(numerical, gradients[which - 2][index], rtol=0.03, atol=3e-4)


def test_epsilon_regularization_changes_only_tail_loss():
    inputs = [np.ones((1, 1, 1, 1), dtype=np.float32),
              np.asarray([-100, 100], dtype=np.float32).reshape(1, 2, 1, 1),
              np.ones((1, 1, 1, 1), dtype=np.float32),
              np.asarray([100, -100], dtype=np.float32).reshape(1, 2, 1, 1),
              np.ones((1, 1, 1), dtype=np.float32)]
    exact = compact_reference(inputs, 1.0)
    regularized = compact_reference(inputs, 1.0, loss_epsilon=1e-12)
    for left, right in zip(exact[:3], regularized[:3]):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(exact[3], [100.0], atol=1e-4)
    np.testing.assert_allclose(regularized[3], [-np.log(1e-12)], rtol=1e-6)


@pytest.mark.parametrize("causal", [False, True])
def test_single_key_probability_support_has_zero_loss_and_all_gradients(causal):
    rng = np.random.default_rng(813)
    shapes = [(2, 3, 7, 17), (2, 1, 7, 17), (2, 3, 5, 17),
              (2, 1, 1, 17), (2, 3, 5)]
    inputs = [rng.normal(0, 2, shape).astype(np.float32) for shape in shapes]
    for value in compact_reference(inputs, 0.3, causal=causal, loss_epsilon=1e-12):
        np.testing.assert_array_equal(value, np.zeros_like(value))
