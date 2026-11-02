"""Targeted validation of the original community/network/music analysis.

All tuning rules are graph-only. Calibration separates a true invariant null,
a continuous-degree-only stress case, and a planted music positive control.
Recovery comparisons separate equal K information from fixed-resolution Louvain.
Perturbation comparisons share both deleted edges and evaluated users.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.stats import t as student_t
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

import lastfm_stats as stats

CORE = ['Spectral', 'SCORE', 'Louvain']
RESOLUTIONS = (1., .5, .25, .125, 2., .75, 1.5, .375, .625, .875, .0625,
               .3125, .4375, .5625, .6875, 1.25, .9375, 1.75)


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 236
    bin_B: int = 499
    calibration_R: int = 200
    calibration_B: int = 199
    calibration_n: int = 324
    recovery_R: int = 30
    recovery_n: int = 600
    legacy_R: int = 20
    perturbation_R: int = 20
    seed_R: int = 10
    n_init: int = 20

    @classmethod
    def for_profile(cls, profile):
        if profile == 'smoke':
            return cls(bin_B=5, calibration_R=4, calibration_B=9, calibration_n=180,
                       recovery_R=2, recovery_n=300, legacy_R=2, perturbation_R=2, seed_R=3)
        if profile != 'full':
            raise ValueError('Unknown validation profile')
        return cls()


def batch_permutation_test(similarity, partitions, strata, B, seed=236):
    """Same transformations for all methods; both statistics are linear in W.

    Given preserved community sizes, fixed coefficients applied to Z.T S Z
    exactly reproduce the existing equal-community and pair-weighted statistics.
    This batches thin products and avoids repeating permutation draws.
    """
    names = list(partitions)
    labels = np.column_stack([np.asarray(partitions[m], dtype=int) for m in names])
    n = len(labels)
    if similarity.shape != (n, n) or len(strata) != n or B < 2:
        raise ValueError('Incompatible permutation inputs')
    memberships = [stats.onehot(labels[:, j]) for j in range(len(names))]
    widths = [z.shape[1] for z in memberships]
    offsets = np.r_[0, np.cumsum(widths)]
    z = np.column_stack(memberships)
    coefficients = np.zeros((len(names), 2, z.shape[1], z.shape[1]))
    for j, membership in enumerate(memberships):
        sizes = membership.sum(0)
        k = len(sizes)
        valid = sizes >= 2
        if k < 2 or not valid.any():
            raise ValueError('Each method needs multiple communities and a nonsingleton')
        equal = -1 / (k * (k - 1) * np.outer(sizes, sizes))
        np.fill_diagonal(equal, 0)
        diagonal = np.zeros(k)
        diagonal[valid] = 1 / (valid.sum() * sizes[valid] * (sizes[valid]-1))
        np.fill_diagonal(equal, diagonal)
        same_count = np.sum(sizes * (sizes-1))
        different_count = n * (n-1) - same_count
        pair = np.full((k, k), -1 / different_count)
        np.fill_diagonal(pair, 1 / same_count)
        lo, hi = offsets[j], offsets[j+1]
        coefficients[j, 0, lo:hi, lo:hi] = equal
        coefficients[j, 1, lo:hi, lo:hi] = pair

    def values():
        w = z.T @ (similarity @ z)
        return np.einsum('msij,ij->ms', coefficients, w, optimize=False)

    observed = values()
    draws = np.empty((B, len(names), 2))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rows = np.repeat(np.arange(n), len(names))
    for b in range(B):
        order = np.arange(n)
        for group in groups:
            order[group] = rng.permutation(group)
        z.fill(0)
        z[rows, (labels[order] + offsets[:-1]).ravel()] = 1
        draws[b] = values()
    results = {}
    for j, name in enumerate(names):
        result = stats.summarize_null(observed[j, 0], draws[:, j, 0])
        result['pair_result'] = stats.summarize_null(observed[j, 1], draws[:, j, 1])
        results[name] = result
    return results


def louvain_labels(graph, resolution, seed, nodelist=None):
    nodelist = list(graph) if nodelist is None else nodelist
    position = {v:i for i, v in enumerate(nodelist)}
    labels = np.empty(len(nodelist), dtype=int)
    for k, community in enumerate(nx.community.louvain_communities(graph, resolution=resolution, seed=seed)):
        labels[[position[v] for v in community]] = k
    return stats.relabel_by_size(labels)


def match_louvain_K(graph, K, seed, nodelist=None, cache=None):
    """Choose resolution by cluster count only; never inspect planted labels."""
    cache = {} if cache is None else dict(cache)
    trace = []
    for gamma in RESOLUTIONS:
        if gamma not in cache:
            cache[gamma] = louvain_labels(graph, gamma, seed, nodelist)
        labels = cache[gamma]
        count = len(np.unique(labels))
        trace.append(dict(resolution=gamma, estimated_K=count))
        if count == K:
            return labels, gamma, trace
    gamma = min(RESOLUTIONS, key=lambda g:(abs(len(np.unique(cache[g]))-K), abs(np.log(g))))
    return cache[gamma], gamma, trace


def fit_core(adjacency, K, seed=236, n_init=20, graph=None, nodelist=None,
             resolution=.25, assist_louvain=False):
    values, vectors = stats.leading_eigs(adjacency, min(K+2, adjacency.shape[0]-1), seed=seed)
    labels = {
        'Spectral':stats.kmeans(vectors[:, :K], K, seed, n_init),
        'SCORE':stats.kmeans(stats.score_ratios(vectors, K), K, seed, n_init),
    }
    graph = nx.from_scipy_sparse_array(adjacency) if graph is None else graph
    fixed = louvain_labels(graph, resolution, seed, nodelist)
    info = dict(fixed_labels=fixed, selected_resolution=resolution, search_trace=[])
    if assist_louvain:
        selected, gamma, trace = match_louvain_K(graph, K, seed, nodelist, cache={resolution:fixed})
        labels['Louvain'] = selected
        info.update(selected_resolution=gamma, search_trace=trace)
    else:
        labels['Louvain'] = fixed
    return labels, info


def bernoulli_dcbm(n, K, mean_degree, degree_sd, within_between_ratio, seed):
    """Independently specified balanced Bernoulli DCBM, not fitted to LastFM.

    Weights have block mean one and bounds [.35, 2.5]. Scaling gives the exact
    requested expected mean degree before sampling. Probabilities are validated,
    never clipped. The same seed couples degree regimes via common random numbers.
    """
    if n % K:
        raise ValueError('Balanced design requires n divisible by K')
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(K), n//K)[rng.permutation(n)]
    normal = rng.normal(size=n)
    theta = np.ones(n)
    if degree_sd:
        raw = np.exp(degree_sd * normal)
        for k in range(K):
            members = truth == k
            x = raw[members]
            scale = brentq(lambda a:np.clip(a*x, .35, 2.5).mean()-1, 0, 2.5/x.min())
            theta[members] = np.clip(scale*x, .35, 2.5)
    intensity = np.outer(theta, theta) * np.where(truth[:, None] == truth[None, :], within_between_ratio, 1)
    np.fill_diagonal(intensity, 0)
    probability = intensity * (n * mean_degree / intensity.sum())
    if probability.max() > 1 or probability.min() < 0:
        raise ValueError('Bernoulli design produced an invalid probability; do not clip it')
    u, v = np.triu_indices(n, 1)
    keep = rng.random(len(u)) < probability[u, v]
    graph = sp.coo_matrix((np.ones(2*keep.sum()), (np.r_[u[keep], v[keep]], np.r_[v[keep], u[keep]])), shape=(n, n)).tocsr()
    return graph, truth, dict(expected_mean_degree=float(probability.sum()/n), maximum_probability=float(probability.max()),
        theta_min=float(theta.min()), theta_max=float(theta.max()), theta_sd=float(theta.std()))


def largest_component(adjacency):
    _, groups = sp.csgraph.connected_components(adjacency, directed=False)
    return np.flatnonzero(groups == np.bincount(groups).argmax())


def common_connected_core(graphs):
    """Prespecified common connected population; never redraw a graph."""
    retained = np.arange(graphs[0].shape[0])
    for _ in range(graphs[0].shape[0]):
        previous = len(retained)
        for adjacency in graphs:
            retained = retained[largest_component(adjacency[retained][:, retained])]
        if len(retained) == previous:
            return retained
    raise RuntimeError('Common connected-core selection did not converge')


def monte_carlo_mean(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    sd = float(values.std(ddof=1)) if count > 1 else np.nan
    mcse = sd / np.sqrt(count)
    half = float(student_t.ppf(.975, count-1)*mcse) if count > 1 else np.nan
    return dict(mean=float(values.mean()), sd=sd, mcse=mcse, ci_low=float(values.mean()-half),
                ci_high=float(values.mean()+half), replications=count)


def _save(frame, directory, stem):
    frame.to_csv(Path(directory) / (stem+'.csv'), index=False, float_format='%.12g')


def _scheme(degree, activity, name):
    if name == 'degree_exact_activity_q5':
        return stats.combine_strata(degree, stats.make_strata(activity, q=5))
    return stats.make_strata(degree, activity, q=int(name.rsplit('q', 1)[1]))


def _residual_fraction(values, labels, strata):
    residual = np.asarray(values, dtype=float).copy()
    for s in np.unique(strata):
        group = strata == s
        residual[group] -= residual[group].mean()
    denominator = np.square(residual).sum()
    if denominator < 1e-25:
        return 0.
    count = np.bincount(labels)
    total = np.bincount(labels, weights=residual)
    return float((total**2 / count).sum()/denominator)


def primary_binning_sensitivity(S, degree, activity, partitions, tables, outputs, config, main_results):
    rows, supports, draws = [], [], {}
    schemes = ['degree_activity_q3', 'degree_activity_q5', 'degree_activity_q7',
               'degree_activity_q10', 'degree_exact_activity_q5']
    for name in schemes:
        strata = _scheme(degree, activity, name)
        results = main_results if name == 'degree_activity_q5' else batch_permutation_test(
            S, {m:partitions[m] for m in CORE}, strata, config.bin_B, config.seed+600)
        for method in CORE:
            labels = np.asarray(partitions[method])
            detail = stats.T_stat(S, labels, True)
            frame = stats.strata_diagnostics(strata, labels, degree=degree, artist_count=activity)
            supports.append(dict(scheme=name, method=method, occupied_strata=len(frame), min_size=int(frame['size'].min()),
                median_size=float(frame['size'].median()), fraction_users_in_cells_below_10=float(frame.loc[frame['size']<10, 'size'].sum()/len(labels)),
                expected_labels_changed=float(np.average(frame.expected_changed_fraction, weights=frame['size'])),
                residual_log_degree_fraction=_residual_fraction(np.log1p(degree), labels, strata),
                residual_log_activity_fraction=_residual_fraction(np.log1p(activity), labels, strata)))
            for statistic, result in [('equal', results[method]), ('pair', results[method]['pair_result'])]:
                rows.append(dict(scheme=name, method=method, statistic=statistic,
                    **{k:v for k, v in result.items() if k not in ['null', 'pair_result']},
                    ratio=detail['ratio' if statistic=='equal' else 'pair_ratio'],
                    role='original_primary_reference' if name=='degree_activity_q5' else 'exploratory_binning_sensitivity'))
                draws[name+'__'+method+'__'+statistic] = result['null']
        print('Primary bin sensitivity:', name, flush=True)
    frame = pd.DataFrame(rows)
    frame['p_Holm_3_methods_within_scheme_statistic'] = frame.groupby(['scheme', 'statistic']).p.transform(lambda p:stats.holm_adjust(p.values))
    _save(frame, tables, 'tab23_primary_binning_sensitivity')
    support = pd.DataFrame(supports)
    _save(support, tables, 'tab24_primary_strata_support')
    np.savez_compressed(Path(outputs)/'primary_binning_nulls.npz', **draws)
    return frame, support


def _music_profiles(degree, truth, scenario, rng, feature_count=90):
    """No community term in either null/stress scenario; fixed row-count sets."""
    n = len(degree)
    degree_bin = stats.make_strata(degree, q=5)
    popularity = np.exp(np.linspace(0, -1, feature_count))
    if scenario == 'exact_binned_null':
        counts = rng.integers(10, 31, size=n)
        pool = np.arange(feature_count) % 5
        weights = popularity[None, :] * np.where(pool[None, :] == degree_bin[:, None], 8., 1.)
    elif scenario == 'smooth_degree_only_stress':
        x = np.log1p(degree)
        x = np.clip((x-x.mean())/x.std(), -2, 2)
        counts = np.clip(rng.poisson(18+3*x), 8, 35)
        weights = popularity[None, :] * np.exp(1.4*x[:, None]*np.linspace(-1, 1, feature_count))
    elif scenario == 'planted_music_positive_control':
        counts = rng.integers(10, 31, size=n)
        pool = np.arange(feature_count) % 3
        weights = popularity[None, :] * np.where(pool[None, :] == truth[:, None], 10., 1.)
    else:
        raise ValueError('Unknown calibration scenario')
    x = np.zeros((n, feature_count))
    for i in range(n):
        selected = rng.choice(feature_count, size=int(counts[i]), replace=False, p=weights[i]/weights[i].sum())
        x[i, selected] = 1
    assert np.array_equal(x.sum(1), counts)
    return sp.csr_matrix(x), counts


def calibrate_primary_test(tables, outputs, config):
    # The graph and all graph-only fitted partitions stay fixed across music
    # replications, matching inference conditional on the observed network.
    adjacency, truth, design = bernoulli_dcbm(config.calibration_n, 3,
        min(14., config.calibration_n/20), 1., 4., config.seed+70000)
    cc = largest_component(adjacency)
    adjacency, truth = adjacency[cc][:, cc].tocsr(), truth[cc]
    degree = np.asarray(adjacency.sum(1)).ravel()
    partitions, info = fit_core(adjacency, 3, config.seed, config.n_init, assist_louvain=True)
    scenarios = ['exact_binned_null', 'smooth_degree_only_stress', 'planted_music_positive_control']
    schemes = ['degree_activity_q5', 'degree_exact_activity_q5']
    records, family_records, null_store = [], [], {}
    for index, scenario in enumerate(scenarios):
        stored = np.empty((config.calibration_R, len(schemes), config.calibration_B, 3, 2))
        for rep in range(config.calibration_R):
            rng = np.random.default_rng(config.seed+71000+10000*index+rep)
            features, activity = _music_profiles(degree, truth, scenario, rng)
            similarity = stats.jaccard_matrix(features)
            for j, scheme in enumerate(schemes):
                strata = _scheme(degree, activity, scheme)
                result = batch_permutation_test(similarity, partitions, strata, config.calibration_B,
                                                seed=config.seed+100000+1000*index+rep)
                invariant = scenario == 'exact_binned_null' or (scenario=='smooth_degree_only_stress' and scheme=='degree_exact_activity_q5')
                for m, method in enumerate(CORE):
                    for s, (statistic, value) in enumerate([('equal', result[method]), ('pair', result[method]['pair_result'])]):
                        stored[rep, j, :, m, s] = value['null']
                        records.append(dict(scenario=scenario, scheme=scheme, rep=rep, method=method, statistic=statistic,
                            p=value['p'], reject=value['p']<=.05, T=value['T'], null_mean=value['null_mean'],
                            occupied_strata=len(np.unique(strata)), assumed_invariant_null=invariant,
                            interpretation='power' if scenario=='planted_music_positive_control' else ('type_I_error' if invariant else 'approximate_adjustment_stress')))
                p = [result[method]['p'] for method in CORE]
                family_records.append(dict(scenario=scenario, scheme=scheme, rep=rep,
                    any_primary_rejection=bool(np.any(stats.holm_adjust(p)<=.05)), assumed_invariant_null=invariant))
            if (rep+1) % 50 == 0:
                print('Calibration:', scenario, rep+1, '/', config.calibration_R, flush=True)
        null_store[scenario] = stored
    raw = pd.DataFrame(records)
    summaries = []
    true_checks = raw[raw.assumed_invariant_null].groupby(['scenario', 'scheme', 'method', 'statistic']).ngroups
    for (scenario, scheme, method, statistic), group in raw.groupby(['scenario', 'scheme', 'method', 'statistic'], sort=False):
        count, size = int(group.reject.sum()), len(group)
        lower, upper = stats.binomial_interval(count, size)
        simultaneous_lower, _ = stats.binomial_interval(count, size, alpha=.05/max(true_checks, 1))
        invariant = bool(group.assumed_invariant_null.iloc[0])
        summaries.append(dict(scenario=scenario, scheme=scheme, method=method, statistic=statistic,
            replications=size, inner_permutations=config.calibration_B, alpha=.05, rejection_count=count,
            rejection_fraction=count/size, pointwise_MC_low=lower, pointwise_MC_high=upper,
            assumed_invariant_null=invariant, interpretation=group.interpretation.iloc[0],
            simultaneous_MC_lower=simultaneous_lower,
            evidence_excess_size=bool(invariant and simultaneous_lower>.05)))
    family = pd.DataFrame(family_records)
    family_summary = []
    for (scenario, scheme), group in family.groupby(['scenario', 'scheme'], sort=False):
        count = int(group.any_primary_rejection.sum())
        lower, upper = stats.binomial_interval(count, len(group))
        family_summary.append(dict(scenario=scenario, scheme=scheme, replications=len(group),
            family_size=3, Holm_family_rejection_fraction=count/len(group), pointwise_MC_low=lower,
            pointwise_MC_high=upper, assumed_invariant_null=bool(group.assumed_invariant_null.iloc[0])))
    summary = pd.DataFrame(summaries)
    _save(raw, tables, 'tab25_primary_calibration_raw')
    _save(summary, tables, 'tab26_primary_calibration_summary')
    _save(pd.DataFrame(family_summary), tables, 'tab27_primary_calibration_family')
    np.savez_compressed(Path(outputs)/'primary_calibration_nulls.npz', **null_store)
    return summary, pd.DataFrame(family_summary), dict(graph_n=len(cc), full_graph_n=config.calibration_n,
        retained_fraction=len(cc)/config.calibration_n, graph_design=design,
        fitted_K={m:len(np.unique(partitions[m])) for m in CORE}, selected_louvain_resolution=info['selected_resolution'])


def _recovery_summary(raw, tables, prefix):
    summaries, differences = [], []
    for (setting, method, regime), group in raw.groupby(['setting', 'method', 'information_regime'], sort=False):
        row = dict(setting=setting, method=method, information_regime=regime,
            replications=len(group), mean_estimated_K=float(group.estimated_K.mean()),
            K_match_fraction=float(group.estimated_K.eq(group.target_K).mean()),
            mean_retained_fraction=float(group.retained_fraction.mean()))
        for metric in ['NMI', 'ARI']:
            row.update({metric+'_'+key:value for key, value in monte_carlo_mean(group[metric]).items() if key!='replications'})
        summaries.append(row)
    for setting, group in raw[raw.information_regime.eq('known_K_assisted')].groupby('setting', sort=False):
        for metric in ['NMI', 'ARI']:
            pivot = group.pivot(index='rep', columns='method', values=metric)
            for first, second in [('SCORE', 'Spectral'), ('Louvain', 'SCORE'), ('Louvain', 'Spectral')]:
                differences.append(dict(setting=setting, metric=metric, comparison=first+' minus '+second,
                    **monte_carlo_mean(pivot[first]-pivot[second])))
    summary, paired = pd.DataFrame(summaries), pd.DataFrame(differences)
    _save(raw, tables, prefix+'_raw')
    _save(summary, tables, prefix+'_summary')
    _save(paired, tables, prefix+'_paired')
    return summary, paired


def independent_recovery(tables, config):
    rows, search_rows, interactions = [], [], []
    settings = [(sd, ratio) for sd in [0., 1.2] for ratio in [3., 8.]]
    for rep in range(config.recovery_R):
        generated = [bernoulli_dcbm(config.recovery_n, 3, 18., sd, ratio,
                     config.seed+200000+rep) for sd, ratio in settings]
        assert all(np.array_equal(generated[0][1], item[1]) for item in generated)
        common = common_connected_core([item[0] for item in generated])
        for (sd, ratio), (full, truth, design) in zip(settings, generated):
            adjacency, planted = full[common][:, common].tocsr(), truth[common]
            K = len(np.unique(planted))
            labels, info = fit_core(adjacency, K, config.seed, config.n_init, assist_louvain=True)
            setting = f'degree_sd={sd:g}; within/between={ratio:g}'
            candidates = [(m, labels[m], 'known_K_assisted', info['selected_resolution'] if m=='Louvain' else np.nan) for m in CORE]
            candidates.append(('Louvain (fixed resolution)', info['fixed_labels'], 'fixed_resolution_reference', .25))
            for method, fitted, regime, gamma in candidates:
                rows.append(dict(setting=setting, rep=rep, method=method, information_regime=regime,
                    degree_sd=sd, within_between_ratio=ratio, target_K=K, estimated_K=len(np.unique(fitted)),
                    NMI=NMI(planted, fitted), ARI=ARI(planted, fitted), retained_fraction=len(common)/config.recovery_n,
                    n_full=config.recovery_n, n_common=len(common), resolution=gamma,
                    full_graph_LCC_fraction=len(largest_component(full))/config.recovery_n, **design))
            search_rows.extend(dict(setting=setting, rep=rep, **item) for item in info['search_trace'])
        if (rep+1) % 5 == 0:
            print('Independent recovery:', rep+1, '/', config.recovery_R, flush=True)
    raw = pd.DataFrame(rows)
    summary, paired = _recovery_summary(raw, tables, 'tab28_independent_recovery')
    for ratio, group in raw[raw.information_regime.eq('known_K_assisted')].groupby('within_between_ratio'):
        for metric in ['NMI', 'ARI']:
            pivot = group.pivot(index=['rep', 'degree_sd'], columns='method', values=metric)
            gains = (pivot['SCORE']-pivot['Spectral']).unstack('degree_sd')
            interactions.append(dict(within_between_ratio=ratio, metric=metric,
                contrast='(SCORE minus Spectral) heterogeneous minus homogeneous', **monte_carlo_mean(gains[1.2]-gains[0.])))
    interaction = pd.DataFrame(interactions)
    _save(interaction, tables, 'tab29_degree_correction_interaction')
    _save(pd.DataFrame(search_rows), tables, 'tab30_independent_resolution_search')
    return summary, paired, interaction


def _sample_legacy(adjacency, truth, weights, rng):
    k = int(truth.max()+1)
    z = stats.onehot(truth)
    block = z.T @ (adjacency @ z)
    members = [np.flatnonzero(truth==i) for i in range(k)]
    probabilities = [weights[g]/weights[g].sum() for g in members]
    us, vs = [], []
    for i in range(k):
        for j in range(i, k):
            count = rng.poisson(block[i, j]/2 if i==j else block[i, j])
            us.append(rng.choice(members[i], count, p=probabilities[i]))
            vs.append(rng.choice(members[j], count, p=probabilities[j]))
    u, v = np.concatenate(us), np.concatenate(vs)
    keep = u != v
    graph = sp.coo_matrix((np.ones(keep.sum()), (u[keep], v[keep])), shape=adjacency.shape).tocsr()
    graph = ((graph+graph.T)>0).astype(float).tocsr()
    cc = largest_component(graph)
    return graph[cc][:, cc].tocsr(), cc


def legacy_matched_recovery(adjacency, partitions, degree, tables, config, resolution=.25, original=None):
    original = pd.read_csv(Path(tables)/'tabA1_simulation_raw.csv') if original is None else original
    settings = [('Poisson DCBM: truncated SCORE+ truth', np.asarray(partitions['SCORE+ (truncated)']), degree),
                ('Poisson DCBM: Louvain truth', np.asarray(partitions['Louvain']), degree),
                ('Homogeneous blocks: truncated SCORE+ truth', np.asarray(partitions['SCORE+ (truncated)']), np.ones(len(degree)))]
    rows, searches = [], []
    for index, (setting, truth, weights) in enumerate(settings):
        reps = sorted(original.loc[original.setting.eq(setting), 'rep'].unique())[:config.legacy_R]
        for rep in reps:
            graph, cc = _sample_legacy(adjacency, truth, weights, np.random.default_rng(config.seed+99+1000*index+rep))
            planted = stats.relabel_by_size(truth[cc])
            old = original[original.setting.eq(setting) & original.rep.eq(rep)].set_index('method')
            assert graph.nnz//2 == int(old.m_lcc.iloc[0]) and len(cc) == int(old.n_lcc.iloc[0])
            fitted, info = fit_core(graph, int(truth.max()+1), config.seed, config.n_init,
                                    resolution=resolution, assist_louvain=True)
            for method in ['Spectral', 'SCORE']:
                assert np.isclose(NMI(planted, fitted[method]), old.loc[method, 'NMI'], atol=1e-9)
                assert np.isclose(ARI(planted, fitted[method]), old.loc[method, 'ARI'], atol=1e-9)
            assert np.isclose(NMI(planted, info['fixed_labels']), old.loc['Louvain', 'NMI'], atol=1e-9)
            candidates = [(m, fitted[m], 'known_K_assisted', info['selected_resolution'] if m=='Louvain' else np.nan) for m in CORE]
            candidates.append(('Louvain (fixed resolution)', info['fixed_labels'], 'fixed_resolution_reference', resolution))
            for method, labels, regime, gamma in candidates:
                rows.append(dict(setting=setting, rep=int(rep), method=method, information_regime=regime,
                    target_K=int(truth.max()+1), estimated_K=len(np.unique(labels)), resolution=gamma,
                    NMI=NMI(planted, labels), ARI=ARI(planted, labels), n_lcc=len(cc), retained_fraction=len(cc)/len(truth)))
            searches.extend(dict(setting=setting, rep=int(rep), **item) for item in info['search_trace'])
        print('Matched legacy recovery:', setting, len(reps), 'replicates', flush=True)
    summary, paired = _recovery_summary(pd.DataFrame(rows), tables, 'tab31_matched_legacy_recovery')
    _save(pd.DataFrame(searches), tables, 'tab32_legacy_resolution_search')
    return summary, paired


def core_stability(adjacency, similarity, graph, nodes, partitions, K, resolution, tables, outputs, config):
    edges = np.array(sp.triu(adjacency, k=1).nonzero()).T
    rng = np.random.default_rng(config.seed+400000)
    rows, seeds, stored, seed_stored = [], [], {}, {}
    for rep in range(config.perturbation_R):
        keep = rng.random(len(edges)) < .9
        kept = edges[keep]
        perturbed = sp.coo_matrix((np.ones(len(kept)), (kept[:, 0], kept[:, 1])), shape=adjacency.shape)
        perturbed = (perturbed+perturbed.T).tocsr()
        cc = largest_component(perturbed)
        sub = perturbed[cc][:, cc].tocsr()
        # Preserve the original graph's node and neighbor iteration order for Louvain.
        subgraph = graph.subgraph(nodes[cc]).copy()
        subgraph.remove_edges_from((nodes[u], nodes[v]) for u, v in edges[~keep])
        assert subgraph.number_of_edges()*2 == sub.nnz
        labels, _ = fit_core(sub, K, config.seed, config.n_init, graph=subgraph,
                             nodelist=nodes[cc], resolution=resolution)
        sub_similarity = similarity[np.ix_(cc, cc)]
        stored[f'rep_{rep:02d}_node_ids'] = nodes[cc]
        for method in CORE:
            reference = stats.relabel_by_size(np.asarray(partitions[method])[cc])
            fitted = labels[method]
            observed = stats.T_stat(sub_similarity, fitted, True)
            baseline = stats.T_stat(sub_similarity, reference, True)
            stored[f'rep_{rep:02d}_{method}'] = fitted
            rows.append(dict(rep=rep, method=method, n_lcc=len(cc), retained_fraction=len(cc)/len(nodes),
                edge_retention_fraction=keep.mean(), estimated_K=len(np.unique(fitted)), NMI=NMI(reference, fitted), ARI=ARI(reference, fitted),
                T=observed['T'], T_pair=observed['T_pair'], restricted_reference_T=baseline['T'],
                restricted_reference_T_pair=baseline['T_pair'], T_change_same_users=observed['T']-baseline['T'],
                pair_T_change_same_users=observed['T_pair']-baseline['T_pair'], ratio=observed['ratio']))
        if (rep+1) % 5 == 0:
            print('Shared edge perturbations:', rep+1, '/', config.perturbation_R, flush=True)
    raw = pd.DataFrame(rows)
    summaries = []
    for method, group in raw.groupby('method', sort=False):
        row = dict(method=method, perturbations=len(group), min_retained_fraction=group.retained_fraction.min(),
                   max_retained_fraction=group.retained_fraction.max())
        for metric in ['NMI', 'ARI', 'T_change_same_users', 'pair_T_change_same_users']:
            row.update({metric+'_mean':group[metric].mean(), metric+'_sd':group[metric].std(ddof=1),
                        metric+'_q025':group[metric].quantile(.025), metric+'_q975':group[metric].quantile(.975)})
        summaries.append(row)
    seed_partitions = {m:[] for m in CORE}
    for rep in range(config.seed_R):
        labels, _ = fit_core(adjacency, K, config.seed+rep, config.n_init, graph=graph,
                             nodelist=nodes, resolution=resolution)
        for method in CORE:
            fitted = labels[method]
            seed_partitions[method].append(fitted)
            seed_stored[f'seed_{config.seed+rep}_{method}'] = fitted
            observed = stats.T_stat(similarity, fitted, True)
            seeds.append(dict(seed=config.seed+rep, method=method, estimated_K=len(np.unique(fitted)),
                NMI_to_original=NMI(partitions[method], fitted), T=observed['T'], T_pair=observed['T_pair']))
    seed_rows = []
    for method, runs in seed_partitions.items():
        comparisons = [NMI(runs[i], runs[j]) for i in range(len(runs)) for j in range(i+1, len(runs))]
        seed_rows.append(dict(method=method, seeds=len(runs), pairwise_comparisons=len(comparisons),
            mean_pairwise_NMI=np.mean(comparisons), minimum_pairwise_NMI=np.min(comparisons),
            interpretation='Pairwise seed comparisons are dependent; no iid-pair confidence interval.'))
    seed_summary = pd.DataFrame(seed_rows)
    _save(raw, tables, 'tab33_core_perturbation_raw')
    _save(pd.DataFrame(summaries), tables, 'tab34_core_perturbation_summary')
    _save(pd.DataFrame(seeds), tables, 'tab35_core_seed_runs')
    _save(seed_summary, tables, 'tab36_core_seed_summary')
    np.savez_compressed(Path(outputs)/'core_perturbation_partitions.npz', **stored)
    np.savez_compressed(Path(outputs)/'core_seed_partitions.npz', **seed_stored)
    return raw, pd.DataFrame(summaries), pd.DataFrame(seeds), seed_summary
