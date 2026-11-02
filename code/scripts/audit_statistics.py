"""Read-only statistical diagnostics for the existing analysis.

This writes an isolated audit JSON. It does not change the fitted partitions,
main hypotheses, p-values, figures, or analysis execution manifest.
"""
from pathlib import Path
import hashlib
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from lastfm_stats import make_strata


def residual_association(values, labels):
    sizes = np.bincount(labels)
    sums = np.bincount(labels, weights=values, minlength=len(sizes))
    return float(np.sum(sums[sizes > 0] ** 2 / sizes[sizes > 0]) / np.square(values).sum())


partitions = pd.read_csv(ROOT / 'outputs/partitions.csv', index_col='node_id')
nodes = partitions.index.to_numpy()
edge_frame = pd.read_csv(ROOT / 'data/lastfm_edges.csv')
u, v = edge_frame.iloc[:, 0].to_numpy(), edge_frame.iloc[:, 1].to_numpy()
keep = np.isin(u, nodes) & np.isin(v, nodes) & (u != v)
u, v = u[keep], v[keep]
indices_u, indices_v = np.searchsorted(nodes, u), np.searchsorted(nodes, v)
degree = np.bincount(np.r_[indices_u, indices_v], minlength=len(nodes))
likes = pd.read_csv(ROOT / 'data/lastfm_features.csv').drop_duplicates()
activity = likes.groupby('node_id').size().reindex(nodes).to_numpy()
strata = make_strata(degree, activity, q=5)
groups = [np.flatnonzero(strata == group) for group in np.unique(strata)]
covariates = {'log1p_degree': np.log1p(degree), 'log1p_artist_count': np.log1p(activity)}
residuals = {}
for name, values in covariates.items():
    residual = values.copy()
    for group in groups:
        residual[group] -= residual[group].mean()
    residuals[name] = residual

diagnostics = []
for method in ['Spectral', 'SCORE', 'Louvain']:
    labels = partitions[method].to_numpy()
    rng = np.random.default_rng(236 + 900)
    observed = {name:residual_association(values, labels) for name, values in residuals.items()}
    null = {name:[] for name in residuals}
    for _ in range(999):
        permuted = labels.copy()
        for group in groups:
            permuted[group] = labels[rng.permutation(group)]
        for name, values in residuals.items():
            null[name].append(residual_association(values, permuted))
    for name, values in residuals.items():
        reference = np.asarray(null[name])
        diagnostics.append(dict(method=method, covariate=name,
            observed_residual_between_community_variance_fraction=observed[name],
            permutation_q025=float(np.quantile(reference, .025)),
            permutation_q975=float(np.quantile(reference, .975)), draws=len(reference),
            interpretation='Descriptive residual covariate/label association after removing stratum means; '
                'this diagnoses remaining within-bin structure, not validity or invalidity of the music test.'))

graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(zip(u, v))
network_summaries = []
for method in ['Spectral', 'SCORE', 'Louvain']:
    labels = partitions[method].to_numpy()
    sizes = np.bincount(labels)
    communities = [set(nodes[labels == k]) for k in range(len(sizes))]
    pair_counts = sizes * (sizes - 1)
    network_summaries.append(dict(method=method,
        largest_community_user_fraction=float(sizes.max()/len(nodes)),
        largest_community_share_of_within_pairs=float(pair_counts.max()/pair_counts.sum()),
        modularity_gamma_1=float(nx.community.modularity(graph, communities, resolution=1)),
        interpretation='Graph summaries are descriptive and evaluated on the fitted graph; '
            'modularity is not a neutral out-of-sample accuracy metric.'))

simulation = pd.read_csv(ROOT / 'tables/tabA1_simulation_raw.csv')
simulation_fairness = []
for (setting, method), group in simulation.groupby(['setting', 'method'], sort=False):
    if method not in ['Spectral', 'SCORE', 'Louvain']:
        continue
    simulation_fairness.append(dict(setting=setting, method=method,
        planted_K=int(group.planted_K.iloc[0]), mean_estimated_K=float(group.estimated_K.mean()),
        fraction_replicates_matching_planted_K=float(group.estimated_K.eq(group.planted_K).mean()),
        mean_retained_fraction=float(group.retained_fraction.mean()),
        algorithm_receives_planted_K=method != 'Louvain'))

core_paired_comparisons = []
for setting, group in simulation.groupby('setting', sort=False):
    pivot = group.pivot(index='rep', columns='method', values='NMI')
    for first, second in [('SCORE', 'Spectral'), ('Louvain', 'SCORE'), ('Louvain', 'Spectral')]:
        differences = pivot[first] - pivot[second]
        mcse = float(differences.std(ddof=1) / np.sqrt(len(differences)))
        half = float(student_t.ppf(.975, len(differences)-1) * mcse)
        core_paired_comparisons.append(dict(setting=setting, comparison=first+' minus '+second,
            mean_NMI_difference=float(differences.mean()), mcse=mcse,
            pointwise_MC_interval_low=float(differences.mean()-half),
            pointwise_MC_interval_high=float(differences.mean()+half),
            interpretation='Monte Carlo uncertainty under the existing generator and information allocation; '
                'not a population-network interval or an unbiased ranking of algorithms.'))

largest_degree_group = max(groups, key=lambda group:degree[group].max())
result = dict(
    scope='Statistical review and follow-up on targeted validation; fitted primary partitions and hypotheses unchanged.',
    assessment='Substantial statistical content, with remaining inferential and comparison-design limitations.',
    main_strata=dict(occupied=len(groups),
        degree_min_in_one_max_degree_cell=int(degree[largest_degree_group].min()),
        degree_max_in_that_cell=int(degree[largest_degree_group].max())),
    residual_covariate_diagnostics=diagnostics,
    network_summaries=network_summaries,
    simulation_fairness=simulation_fairness,
    core_paired_simulation_comparisons=core_paired_comparisons,
    prioritized_actions=[
        dict(priority='High', action='Calibrate the primary degree/activity procedure and assess bin sensitivity for all three core methods.',
            rationale='The existing country-only calibration does not address the current primary reference; bins leave residual covariate structure.'),
        dict(priority='High', action='Add a matched-information recovery comparison and an independently specified planted network design.',
            rationale='Separate true-K assistance and fitted-partition generator dependence from recovery ability; report LCC retention.'),
        dict(priority='Medium', action='Apply the same graph perturbations and seed checks to Spectral, SCORE and Louvain.',
            rationale='Current edge-deletion stability focuses on the SCORE+ extension rather than the core method comparison.'),
        dict(priority='Interpretation', action='Keep both weighting schemes and state the target comparison before presenting a yes/no conclusion.',
            rationale='Spectral changes sign; equal-community and equal-user-pair statistics are distinct estimands.'),
    ],
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=[
        'Primary-reference calibration is conditional on one generated graph and cannot establish observational exchangeability.',
        'Shared perturbations and seeds assess algorithmic sensitivity, not uncertainty over a population network.',
        'Primary equal-community and pair-weighted contrasts target different populations of comparisons.',
        'Simulation performance is conditional on planted-partition calibration, known-K assistance, and LCC selection.',
        'Software/result-integrity checks do not establish exchangeability, model fit, or population-effect coverage.',
    ],
)
validation_path = ROOT/'outputs/statistical_validation_manifest.json'
if validation_path.exists():
    followup=json.loads(validation_path.read_text())
    result['validation_followup']=dict(
        manifest='outputs/statistical_validation_manifest.json',
        configuration=followup['configuration'], warnings=followup['warnings'],
        completed_actions=['Primary degree/activity calibration and five nuisance references for core methods.',
            'Matched-K recovery on reproduced legacy graphs and independent Bernoulli designs.',
            'Shared graph perturbations and seed checks for Spectral, SCORE and Louvain.'],
        interpretation='Original fairness diagnostics above describe Appendix A before K assistance; Appendix D addresses this comparison. Remaining limits and flags must accompany report claims.')
    for action in result['prioritized_actions']:
        action['status']='addressed in Appendix D' if action['priority']!='Interpretation' else 'ongoing reporting qualification'
    result['assessment']='Targeted statistical additions completed; inference remains conditional and exploratory.'
destination = ROOT / 'outputs/statistical_audit.json'
destination.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
