"""Run the statistical additions from saved analysis outputs.

Default output: project tables/, outputs/, figures/.
--smoke output: tmp/validation_smoke/ only; never use it in the report.
The main notebook invokes the same functions without rereading its matrices.
"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import networkx as nx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import lastfm_stats as stats
import statistical_validation as validation
from validation_visualizations import generate_validation_figures, write_validation_manifest
from visualizations import _load_matrices

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
config = validation.ValidationConfig.for_profile('smoke' if args.smoke else 'full')
saved_config = json.loads((ROOT/'outputs/run_manifest.json').read_text())['configuration']
selected_K, selected_resolution = saved_config['K_selected'], saved_config['gamma']
run = ROOT/'tmp/validation_smoke' if args.smoke else ROOT
for directory in ['tables', 'outputs', 'figures']:
    (run/directory).mkdir(parents=True, exist_ok=True)

with threadpool_limits(limits=4):
    partitions = pd.read_csv(ROOT/'outputs/partitions.csv', index_col='node_id')
    nodes = partitions.index.to_numpy()
    adjacency, similarity = _load_matrices(ROOT, nodes)
    degree = np.asarray(adjacency.sum(1)).ravel()
    data = ROOT/'data'
    snap = (data/'lastfm_asia_edges.csv').exists()
    if snap:
        features = json.loads((data/'lastfm_asia_features.json').read_text())
        activity = np.array([len(set(features[str(u)])) for u in nodes])
    else:
        filename = 'lastfm_features.csv' if (data/'lastfm_features.csv').exists() else 'lastfm_features.csv.gz'
        likes = pd.read_csv(data/filename).drop_duplicates()
        activity = likes.groupby('node_id').size().reindex(nodes).to_numpy()
    # Restore the original-ID graph and adjacency iteration order for stability.
    edge_frame = pd.read_csv(data/('lastfm_asia_edges.csv' if snap else 'lastfm_edges.csv'))
    target = pd.read_csv(data/('lastfm_asia_target.csv' if snap else 'lastfm_target.csv'))
    graph = nx.Graph()
    graph.add_nodes_from(target.iloc[:, 0])
    canonical = np.sort(edge_frame.to_numpy(), axis=1)
    graph.add_edges_from(canonical[canonical[:, 0] != canonical[:, 1]])
    graph = graph.subgraph(nodes).copy()
    cached = np.load(ROOT/'outputs/permutation_nulls.npz')
    main_results = {}
    for method in validation.CORE:
        detail = stats.T_stat(similarity, partitions[method].to_numpy(), True)
        result = stats.summarize_null(detail['T'], cached['degree_activity_q5__'+method+'__equal'])
        result['pair_result'] = stats.summarize_null(detail['T_pair'], cached['degree_activity_q5__'+method+'__pair'])
        main_results[method] = result
    tables, outputs = run/'tables', run/'outputs'
    print('Stage 1: primary binning sensitivity', flush=True)
    validation.primary_binning_sensitivity(similarity, degree, activity, partitions, tables, outputs, config, main_results)
    print('Stage 2: primary-null calibration and stress tests', flush=True)
    _, _, calibration_design = validation.calibrate_primary_test(tables, outputs, config)
    print('Stage 3: independent matched-information recovery', flush=True)
    validation.independent_recovery(tables, config)
    print('Stage 4: known-K assistance on reproduced legacy graphs', flush=True)
    old = pd.read_csv(ROOT/'tables/tabA1_simulation_raw.csv')
    validation.legacy_matched_recovery(adjacency, partitions, degree, tables, config,
                                       resolution=selected_resolution, original=old)
    print('Stage 5: shared graph perturbations and seed checks', flush=True)
    validation.core_stability(adjacency, similarity, graph, nodes, partitions, selected_K,
                              selected_resolution, tables, outputs, config)
    generate_validation_figures(run)
    write_validation_manifest(run, config, calibration_design, 'smoke' if args.smoke else 'full')
print('Validation computations complete:', run)
