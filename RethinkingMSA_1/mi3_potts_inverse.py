"""Clean-room RNA q=5 Potts inverse sampler inspired by mi3-GPU.

The implementation intentionally does not copy mi3-GPU source code. It follows
the same statistical structure at RNA scale: construct target bivariate
marginals, fit pair couplings with MCMC plus damped pairwise Newton updates, and
sample a synthetic MSA from the fitted Potts model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


ALPHABET = "AUCG-"
GAP_IDX = 4
N_STATES = len(ALPHABET)
EPS = 1e-12
CANONICAL_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G")}


@dataclass(frozen=True)
class PairIndex:
    i: np.ndarray
    j: np.ndarray
    lookup: np.ndarray

    @property
    def size(self) -> int:
        return int(self.i.size)


@dataclass
class Mi3PottsResult:
    msa_num: np.ndarray
    target_bimarg: np.ndarray
    final_model_bimarg: np.ndarray
    final_J: np.ndarray
    history: list[dict[str, float]]
    baseline_fitness: dict[str, float]
    best_fitness: dict[str, float]


def make_pair_index(length: int) -> PairIndex:
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    lookup = np.full((length, length), -1, dtype=np.int32)
    pair_idx = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            pairs_i.append(i)
            pairs_j.append(j)
            lookup[i, j] = pair_idx
            lookup[j, i] = pair_idx
            pair_idx += 1
    return PairIndex(
        i=np.asarray(pairs_i, dtype=np.int32),
        j=np.asarray(pairs_j, dtype=np.int32),
        lookup=lookup,
    )


def query_to_numeric(query: str) -> np.ndarray:
    return np.asarray([ALPHABET.find(ch) if ch in ALPHABET else GAP_IDX for ch in query], dtype=np.int8)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, EPS)
    return probs / probs.sum(axis=-1, keepdims=True)


def query_prior_probs(query: str, same_prob: float = 0.70, mutation_prob_each: float = 0.09, gap_prob: float = 0.03) -> np.ndarray:
    probs = np.empty((len(query), N_STATES), dtype=np.float64)
    for pos, base in enumerate(query):
        row = np.full(N_STATES, mutation_prob_each, dtype=np.float64)
        row[GAP_IDX] = gap_prob
        base_idx = ALPHABET.find(base)
        if 0 <= base_idx < 4:
            row[base_idx] = same_prob
        else:
            row[:4] = (1.0 - gap_prob) / 4.0
        probs[pos] = row / row.sum()
    return probs


def mixed_single_site_prior(
    query: str,
    alphabet_probs: np.ndarray,
    query_prior_mix: float,
    same_prob: float = 0.70,
    mutation_prob_each: float = 0.09,
    gap_prob: float = 0.03,
) -> np.ndarray:
    background = normalize_probs(np.asarray(alphabet_probs, dtype=np.float64))[None, :]
    background = np.repeat(background, len(query), axis=0)
    q_prior = query_prior_probs(query, same_prob, mutation_prob_each, gap_prob)
    mix = float(np.clip(query_prior_mix, 0.0, 1.0))
    return normalize_probs((1.0 - mix) * background + mix * q_prior)


def _is_valid_pair(a: int, b: int) -> bool:
    if a == GAP_IDX or b == GAP_IDX:
        return False
    pair = (ALPHABET[a], ALPHABET[b])
    return pair in CANONICAL_PAIRS or pair in WOBBLE_PAIRS


def _is_canonical(a: int, b: int) -> bool:
    return a != GAP_IDX and b != GAP_IDX and (ALPHABET[a], ALPHABET[b]) in CANONICAL_PAIRS


def _is_wobble(a: int, b: int) -> bool:
    return a != GAP_IDX and b != GAP_IDX and (ALPHABET[a], ALPHABET[b]) in WOBBLE_PAIRS


def pair_kernel(qi: int, qj: int) -> np.ndarray:
    kernel = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    query_pair = _is_valid_pair(qi, qj)
    for a in range(N_STATES):
        for b in range(N_STATES):
            if a == GAP_IDX or b == GAP_IDX:
                score = -1.50
            elif query_pair:
                if _is_canonical(a, b):
                    score = 1.25
                elif _is_wobble(a, b):
                    score = 0.90
                else:
                    score = -1.05
                one_changed = (a == qi) ^ (b == qj)
                both_changed = (a != qi) and (b != qj)
                if both_changed and _is_valid_pair(a, b):
                    score += 0.40
                elif one_changed:
                    score -= 0.65
            else:
                if a == qi and b == qj:
                    score = 0.35
                elif a != qi and b != qj:
                    score = 0.75 if _is_valid_pair(a, b) else 0.40
                elif (a == qi) ^ (b == qj):
                    score = -0.55
                else:
                    score = -0.25
            kernel[a, b] = score
    return kernel - kernel.mean()


def mutual_information_from_joint(joint: np.ndarray) -> float:
    joint = normalize_probs(joint.reshape(-1)).reshape(N_STATES, N_STATES)
    fi = joint.sum(axis=1, keepdims=True)
    fj = joint.sum(axis=0, keepdims=True)
    return float(np.sum(joint * np.log2((joint + EPS) / (fi @ fj + EPS))))


def _scalar_map_scale(target_map: np.ndarray) -> float:
    values = target_map[np.triu_indices(target_map.shape[0], k=1)]
    positives = values[np.isfinite(values) & (values > 0)]
    if positives.size == 0:
        return 1.0
    return float(max(np.percentile(positives, 95), EPS))


def _solve_lambda_for_mi(independent: np.ndarray, preferred: np.ndarray, target_mi: float) -> float:
    max_mi = mutual_information_from_joint(preferred)
    if target_mi <= EPS or max_mi <= EPS:
        return 0.0
    if target_mi >= max_mi:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        joint = (1.0 - mid) * independent + mid * preferred
        if mutual_information_from_joint(joint) < target_mi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def construct_bivariate_target_from_scalar_map(
    target_map: np.ndarray,
    query: str,
    alphabet_probs: np.ndarray,
    query_prior_mix: float = 0.35,
    pair_beta_max: float = 8.0,
    same_prob: float = 0.70,
    mutation_prob_each: float = 0.09,
    gap_prob: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, PairIndex]:
    target_map = np.asarray(target_map, dtype=np.float64)
    if target_map.shape != (len(query), len(query)):
        raise ValueError(f"target map shape {target_map.shape} does not match query length {len(query)}")
    clean_map = 0.5 * (target_map + target_map.T)
    clean_map = np.maximum(clean_map, 0.0)
    np.fill_diagonal(clean_map, 0.0)
    scale = _scalar_map_scale(clean_map)
    single = mixed_single_site_prior(query, alphabet_probs, query_prior_mix, same_prob, mutation_prob_each, gap_prob)
    pair_index = make_pair_index(len(query))
    query_idx = query_to_numeric(query)
    target_bimarg = np.empty((pair_index.size, N_STATES * N_STATES), dtype=np.float64)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        independent = np.outer(single[i], single[j])
        kernel = pair_kernel(int(query_idx[i]), int(query_idx[j]))
        preferred = independent * np.exp(float(pair_beta_max) * kernel)
        preferred = normalize_probs(preferred.reshape(-1)).reshape(N_STATES, N_STATES)
        strength = float(np.clip(clean_map[i, j] / scale, 0.0, 1.0))
        target_mi = strength * mutual_information_from_joint(preferred)
        lam = _solve_lambda_for_mi(independent, preferred, target_mi)
        joint = normalize_probs(((1.0 - lam) * independent + lam * preferred).reshape(-1))
        target_bimarg[pair_idx] = joint
    return target_bimarg, single, pair_index


def apply_covariation_preserving_pseudocount(
    bimarg: np.ndarray,
    single_probs: np.ndarray,
    pair_index: PairIndex,
    target_neff: float,
    pc: float = 0.5,
) -> np.ndarray:
    bimarg = normalize_probs(np.asarray(bimarg, dtype=np.float64))
    single_probs = normalize_probs(np.asarray(single_probs, dtype=np.float64))
    mu = N_STATES * float(pc) / max(float(target_neff) + N_STATES * float(pc), EPS)
    out = np.empty_like(bimarg)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        joint = bimarg[pair_idx].reshape(N_STATES, N_STATES)
        fi = single_probs[i]
        fj = single_probs[j]
        additive = np.add.outer(fi, fj) / N_STATES
        pc_joint = (1.0 - mu) ** 2 * joint + (1.0 - mu) * mu * additive + (mu / N_STATES) ** 2
        out[pair_idx] = normalize_probs(pc_joint.reshape(-1))
    return out


def compute_bimarg_numeric(
    msa_num: np.ndarray,
    pair_index: PairIndex | None = None,
    weights: np.ndarray | None = None,
    length: int | None = None,
) -> np.ndarray:
    msa_num = np.asarray(msa_num, dtype=np.int8)
    if msa_num.ndim != 2:
        raise ValueError("msa_num must have shape [N,L]")
    if length is None:
        length = int(msa_num.shape[1])
    if pair_index is None:
        pair_index = make_pair_index(length)
    if msa_num.shape[1] != length:
        raise ValueError(f"MSA length {msa_num.shape[1]} != expected {length}")
    if weights is None:
        weights = np.ones(msa_num.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (msa_num.shape[0],):
        raise ValueError(f"weights shape {weights.shape} does not match number of sequences {msa_num.shape[0]}")
    total = float(weights.sum())
    if total <= EPS:
        weights = np.ones(msa_num.shape[0], dtype=np.float64)
        total = float(weights.sum())
    out = np.empty((pair_index.size, N_STATES * N_STATES), dtype=np.float64)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        pair_state = msa_num[:, i].astype(np.int16) * N_STATES + msa_num[:, j].astype(np.int16)
        counts = np.bincount(pair_state, weights=weights, minlength=N_STATES * N_STATES).astype(np.float64)
        out[pair_idx] = counts / total
    return out


def bimarg_to_mi_apc(bimarg: np.ndarray, length: int, pair_index: PairIndex | None = None) -> np.ndarray:
    if pair_index is None:
        pair_index = make_pair_index(length)
    mi = np.zeros((length, length), dtype=np.float64)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        value = mutual_information_from_joint(bimarg[pair_idx])
        mi[i, j] = mi[j, i] = max(value, 0.0)
    denom = max(length - 1, 1)
    row_mean = mi.sum(axis=1) / denom
    total_mean = mi.sum() / max(length * (length - 1), 1)
    if total_mean <= EPS:
        return np.zeros_like(mi)
    apc = mi - np.outer(row_mean, row_mean) / total_mean
    apc = np.maximum(apc, 0.0)
    np.fill_diagonal(apc, 0.0)
    return apc


def initialize_independent_couplings(single_probs: np.ndarray, pair_index: PairIndex) -> np.ndarray:
    single_probs = normalize_probs(single_probs)
    fields = -np.log(single_probs + EPS)
    denom = max(single_probs.shape[0] - 1, 1)
    J = np.empty((pair_index.size, N_STATES, N_STATES), dtype=np.float64)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        J[pair_idx] = fields[i, :, None] / denom + fields[j, None, :] / denom
    return J.reshape(pair_index.size, N_STATES * N_STATES)


def sample_independent(single_probs: np.ndarray, depth: int, rng: np.random.Generator) -> np.ndarray:
    single_probs = normalize_probs(single_probs)
    length = single_probs.shape[0]
    out = np.empty((depth, length), dtype=np.int8)
    for pos in range(length):
        out[:, pos] = rng.choice(N_STATES, size=depth, replace=True, p=single_probs[pos]).astype(np.int8)
    return out


def _sample_rows_from_logits(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    return (rng.random(logits.shape[0])[:, None] > cdf).sum(axis=1).astype(np.int8)


def mcmc_sample_potts(
    J: np.ndarray,
    single_probs: np.ndarray,
    pair_index: PairIndex,
    depth: int,
    rng: np.random.Generator,
    sweeps: int = 10,
    temperature: float = 1.0,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    length = single_probs.shape[0]
    J3 = np.asarray(J, dtype=np.float64).reshape(pair_index.size, N_STATES, N_STATES)
    if initial_states is None or initial_states.shape != (depth, length):
        states = sample_independent(single_probs, depth, rng)
    else:
        states = np.asarray(initial_states, dtype=np.int8).copy()
    temp = max(float(temperature), EPS)
    for _ in range(max(0, int(sweeps))):
        for pos in rng.permutation(length):
            energy = np.zeros((depth, N_STATES), dtype=np.float64)
            for other in range(length):
                if other == pos:
                    continue
                pair_idx = int(pair_index.lookup[pos, other])
                other_states = states[:, other]
                if pos < other:
                    energy += J3[pair_idx, :, other_states]
                else:
                    energy += J3[pair_idx, other_states, :]
            states[:, pos] = _sample_rows_from_logits(-energy / temp, rng)
    return states


def sequence_energies(msa_num: np.ndarray, J: np.ndarray, pair_index: PairIndex) -> np.ndarray:
    msa_num = np.asarray(msa_num, dtype=np.int8)
    J2 = np.asarray(J, dtype=np.float64)
    energies = np.zeros(msa_num.shape[0], dtype=np.float64)
    for pair_idx, (i, j) in enumerate(zip(pair_index.i, pair_index.j)):
        flat = msa_num[:, i].astype(np.int16) * N_STATES + msa_num[:, j].astype(np.int16)
        energies += J2[pair_idx, flat]
    return energies


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights * weights))
    if denom <= EPS:
        return 0.0
    return float(weights.sum() ** 2 / denom)


def zwanzig_newton_steps(
    J: np.ndarray,
    sampled_num: np.ndarray,
    query_idx: np.ndarray,
    target_bimarg: np.ndarray,
    pair_index: PairIndex,
    gamma: float = 0.02,
    damping: float = 0.001,
    max_steps: int = 64,
    frac_neff: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    J_base = np.asarray(J, dtype=np.float64)
    J_current = J_base.copy()
    combined = np.vstack([query_idx[None, :], sampled_num])
    base_weights = np.ones(combined.shape[0], dtype=np.float64)
    model = compute_bimarg_numeric(combined, pair_index, weights=base_weights)
    history: list[dict[str, float]] = []
    n_synth = max(sampled_num.shape[0], 1)
    for step in range(1, max(0, int(max_steps)) + 1):
        delta = float(gamma) * (model - target_bimarg) / (model + float(damping))
        J_current = J_current + delta
        delta_E = sequence_energies(sampled_num, J_current - J_base, pair_index)
        weights_synth = np.exp(float(delta_E.min()) - delta_E)
        neff = effective_sample_size(weights_synth)
        combined_weights = np.concatenate([np.ones(1, dtype=np.float64), weights_synth])
        model = compute_bimarg_numeric(combined, pair_index, weights=combined_weights)
        tensor_mae = float(np.mean(np.abs(model - target_bimarg)))
        history.append({"newton_step": float(step), "neff": float(neff), "neff_fraction": float(neff / n_synth), "tensor_mae": tensor_mae})
        if neff < float(frac_neff) * n_synth:
            break
    return J_current, model, history


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(a.std()) <= EPS or float(b.std()) <= EPS:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top_l_overlap(target: np.ndarray, generated: np.ndarray) -> float:
    length = min(target.shape[0], generated.shape[0])
    top_k = min(length, length * (length - 1) // 2)
    if top_k <= 0:
        return 0.0
    idx = np.triu_indices(length, k=1)
    t_values = target[:length, :length][idx]
    g_values = generated[:length, :length][idx]
    t_order = np.argsort(t_values)[::-1][:top_k]
    g_order = np.argsort(g_values)[::-1][:top_k]
    return float(len(set(t_order.tolist()) & set(g_order.tolist())) / top_k)


def map_fitness(target_map: np.ndarray, generated_map: np.ndarray, tensor_mae: float = 0.0) -> dict[str, float]:
    pearson = _safe_pearson(_upper_values(target_map), _upper_values(generated_map))
    top_l = _top_l_overlap(target_map, generated_map)
    mae = float(np.mean(np.abs(_upper_values(target_map) / _scalar_map_scale(target_map) - _upper_values(generated_map) / _scalar_map_scale(target_map))))
    loss = (1.0 - pearson) + mae + 0.25 * (1.0 - top_l) + tensor_mae
    return {
        "loss": float(loss),
        "pearson": float(pearson),
        "top_l_overlap": float(top_l),
        "selected_edge_mae": float(mae),
        "tensor_mae": float(tensor_mae),
    }


def fit_mi3_potts_inverse(
    target_map: np.ndarray,
    query: str,
    alphabet_probs: np.ndarray,
    synthetic_depth: int,
    rng: np.random.Generator,
    args: Any,
) -> Mi3PottsResult:
    target_bimarg, single_probs, pair_index = construct_bivariate_target_from_scalar_map(
        target_map,
        query,
        alphabet_probs,
        query_prior_mix=float(getattr(args, "mi3_query_prior_mix", 0.35)),
        pair_beta_max=float(getattr(args, "mi3_pair_beta_max", 8.0)),
        same_prob=float(getattr(args, "same_prob", 0.70)),
        mutation_prob_each=float(getattr(args, "mutation_prob_each", 0.09)),
        gap_prob=float(getattr(args, "gap_prob", 0.03)),
    )
    target_bimarg = apply_covariation_preserving_pseudocount(
        target_bimarg,
        single_probs,
        pair_index,
        target_neff=max(float(synthetic_depth + 1), 1.0),
        pc=float(getattr(args, "mi3_target_pseudocount", 0.5)),
    )
    query_idx = query_to_numeric(query)
    init_mode = str(getattr(args, "mi3_init", "independent"))
    if init_mode == "zero":
        J = np.zeros((pair_index.size, N_STATES * N_STATES), dtype=np.float64)
    elif init_mode == "independent":
        J = initialize_independent_couplings(single_probs, pair_index)
    else:
        raise ValueError(f"Unknown mi3 init mode: {init_mode}")

    sample = sample_independent(single_probs, synthetic_depth, rng)
    combined = np.vstack([query_idx[None, :], sample])
    model_bimarg = compute_bimarg_numeric(combined, pair_index)
    generated_map = bimarg_to_mi_apc(model_bimarg, len(query), pair_index)
    tensor_mae = float(np.mean(np.abs(model_bimarg - target_bimarg)))
    baseline_fitness = map_fitness(target_map, generated_map, tensor_mae)
    best_sample = sample.copy()
    best_model_bimarg = model_bimarg.copy()
    best_J = J.copy()
    best_fitness = dict(baseline_fitness)
    history: list[dict[str, float]] = [
        {"round": 0.0, **baseline_fitness, "neff": float(synthetic_depth), "newton_steps": 0.0}
    ]

    rounds = max(0, int(getattr(args, "mi3_mcmc_rounds", 15)))
    for round_idx in range(1, rounds + 1):
        sample = mcmc_sample_potts(
            J,
            single_probs,
            pair_index,
            synthetic_depth,
            rng,
            sweeps=int(getattr(args, "mi3_mcmc_sweeps", 10)),
            temperature=float(getattr(args, "mi3_temperature", 1.0)),
            initial_states=sample,
        )
        combined = np.vstack([query_idx[None, :], sample])
        model_bimarg = compute_bimarg_numeric(combined, pair_index)
        generated_map = bimarg_to_mi_apc(model_bimarg, len(query), pair_index)
        tensor_mae = float(np.mean(np.abs(model_bimarg - target_bimarg)))
        fitness = map_fitness(target_map, generated_map, tensor_mae)
        if fitness["loss"] < best_fitness["loss"]:
            best_sample = sample.copy()
            best_model_bimarg = model_bimarg.copy()
            best_J = J.copy()
            best_fitness = dict(fitness)

        J, model_bimarg, newton_history = zwanzig_newton_steps(
            J,
            sample,
            query_idx,
            target_bimarg,
            pair_index,
            gamma=float(getattr(args, "mi3_gamma", 0.02)),
            damping=float(getattr(args, "mi3_damping", 0.001)),
            max_steps=int(getattr(args, "mi3_newton_steps", 64)),
            frac_neff=float(getattr(args, "mi3_frac_neff", 0.90)),
        )
        last_neff = newton_history[-1]["neff"] if newton_history else float(synthetic_depth)
        history.append(
            {
                "round": float(round_idx),
                **fitness,
                "neff": float(last_neff),
                "newton_steps": float(len(newton_history)),
                "post_newton_tensor_mae": float(newton_history[-1]["tensor_mae"] if newton_history else tensor_mae),
            }
        )

    final_combined = np.vstack([query_idx[None, :], best_sample])
    final_model_bimarg = compute_bimarg_numeric(final_combined, pair_index)
    return Mi3PottsResult(
        msa_num=best_sample,
        target_bimarg=target_bimarg,
        final_model_bimarg=final_model_bimarg,
        final_J=best_J,
        history=history,
        baseline_fitness=baseline_fitness,
        best_fitness=best_fitness,
    )


def result_to_dict(result: Mi3PottsResult) -> dict[str, Any]:
    return {
        "msa_num": result.msa_num,
        "target_bimarg": result.target_bimarg,
        "final_model_bimarg": result.final_model_bimarg,
        "final_J": result.final_J,
        "history": result.history,
        "baseline_fitness": result.baseline_fitness,
        "best_fitness": result.best_fitness,
    }
