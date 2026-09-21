# Test coverage audit

The audit distinguishes executable numeric coverage, metadata/validation coverage, and
checks that require an implemented operator or clarified evaluator ABI.

Case definitions are maintained in three disjoint suites: Coverage
(`st/cases/coverage.json`), Correctness (`st/cases/correctness.json`) and
Mock Cases (`st/cases/mock.json`). The first two retain all 63 formal-contract
descriptors, with 22 compact-ABI regressions additionally included under
Correctness. Mock Cases contains seven fixed compact fixtures. Compact cases
are not counted as evidence of formal full-ABI support.

| Document requirement | Coverage | Cases/checks |
|---|---|---|
| BSND/ND interpretation, D=128, Dr=64 | Numeric + rejection | every numeric case; `invalid_d`, `invalid_dr`, `invalid_layout` |
| B in 1..256 | Boundary numeric | `smoke_uniform_minimum`, `batch_maximum_numeric`; B=0/257 rejection |
| S1/S2 in 1..128K, non-empty | Numeric + shape boundary | minimum smoke, `max_s1_shape_only`, `max_s2_shape_only`; empty/128K+1 rejection |
| S1=S2, S1<S2, S1>S2 | Numeric | smoke, documented/non-aligned cases, `query_longer_than_key` |
| N1={32,64,128}, N2=N1 | Full Cartesian numeric + rejection | twelve `head_grid_*` cases; invalid N1/N2 cases |
| Nidx1={8,16,32,64}, Nidx2=1 | Full Cartesian numeric + rejection | twelve `head_grid_*` cases; invalid Nidx1/Nidx2 cases |
| float16/bfloat16 tensors; float16/bfloat16/float32 weights | All valid dtype paths + rejection | three `dtype_*` cases plus fp16 baseline; mismatch/invalid cases |
| Output shapes and dtype following inputs; float32 loss | Numeric/ST export check | all numeric shape checks; `test_raw_bfloat16_fixture_*` |
| Target p, index q, KL sum, three gradients | Numeric | all numeric cases; published example and zero-loss smoke |
| Stable Softmax and probability tails | Adversarial numeric | `probability_tail_stability` |
| Supplied main/index Softmax max/sum | Reconstruction numeric | every numeric case plus algebraically shifted-statistics check |
| ReLU negative/positive and S=0 subgradient | Adversarial numeric | `relu_zero_and_signs` |
| Zero weights chain rule | Adversarial numeric | `zero_weights_chain_rule`: dQ/dK=0 while dW remains active |
| Finite-difference gradients | Numeric | `finite_difference_probe` for queryIndex/keyIndex/weights |
| `dKeyIndex` accumulates across queries | Numeric decomposition | `dkey_cross_query_accumulation` |
| rightDownCausal, exact boundary, masked positions | Numeric | `right_down_mask_boundary`, non-aligned and unequal cases |
| actual lengths optional/default and explicit per batch | Numeric | default cases; `variable_lengths_and_padding_poison` |
| Padding ignored and outputs fully initialized | Poison numeric | `variable_lengths_and_padding_poison`; empty-row case |
| Non-aligned tails | Numeric + large shape | `non_aligned_tail_tiles`, `large_mixed_shape_only` |
| RoPE absent/zero and nonzero contribution | Numeric, convention pending | zero-rope baseline, variable/random RoPE, `rope_dominates_main_score` |
| scaleValue propagation | Numeric | default cases and `custom_scale_value` |
| layout=BSND, sparseMode=3, default pre/next tokens only | Valid + rejection | all numeric; invalid layout/mode/token cases |
| Sum reduction over queries/batches | Numeric | `test_loss_reduction_is_a_sum_not_a_mean`, batch maximum |
| Atlas A2/A3 only; reject 950PR/950DT | Runtime specification | four platform `runtime_only_cases` |
| Default nondeterminism and deterministic context option | Runtime specification | two determinism `runtime_only_cases` |
| Streaming/online behavior at 128K | Shape/performance specification | `max_s2_shape_only`; requires performance instrumentation after implementation |

## Audit result

All explicitly enumerated shapes, dtypes, head counts, mathematical relationships,
masking, variable-length, padding, reduction, and error constraints have a numeric or
specification case. The following cannot yet be closed as authoritative executable tests:

1. **Full versus compact ABI:** current Host code exposes five tensors and one attribute,
   while the statement exposes statistics, RoPE, sequence lengths, layout, and sparseMode.
   The conflicting float32-input and float16-loss paths are quarantined under
   `abi_pending_cases`, not accepted as formal valid cases.
2. **RoPE fusion algebra:** separate RoPE tensor shapes are documented, but the exact
   element-wise rotation/fusion equation is not. Tests currently use additive RoPE dot
   products in main score and keep that rule isolated.
3. **S1>S2 empty causal rows:** the statement gives its mask formula under an S2>=S1
   assumption while allowing unequal lengths generally. Tests use the natural
   right-aligned generalization and skip empty rows.
4. **Hardware/determinism/performance:** these require device execution and access to
   each target/unsupported platform; CPU tests do not establish this coverage.

These are documented test oracles awaiting evaluator evidence, not silently omitted
coverage.
