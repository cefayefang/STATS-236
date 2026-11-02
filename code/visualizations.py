"""Report figures for the original network/community/music research questions.

Run independently with `.venv/bin/python code/visualizations.py`, or call
`generate_figures` from the notebook with its already-computed matrices.
All figures use the saved partitions; music is never used to refit communities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp/matplotlib'))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from lastfm_stats import T_stat

CORE = ['Spectral', 'SCORE', 'Louvain']
COLORS = {'Spectral': '#eda100', 'SCORE': '#1baf7a', 'Louvain': '#eb6834'}
INK, MUTED, GRID = '#17212b', '#586473', '#e3e7eb'


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _graph_signature(nodes, adjacency):
    return hashlib.sha256(nodes.tobytes() + adjacency.indices.tobytes() + adjacency.indptr.tobytes()).hexdigest()


def _load_matrices(root, nodes):
    """Same ordered retained users and binary artist sets as the analysis."""
    from lastfm_stats import jaccard_matrix

    data = root / 'data'
    snap = (data / 'lastfm_asia_edges.csv').exists()
    edges = pd.read_csv(data / ('lastfm_asia_edges.csv' if snap else 'lastfm_edges.csv')).to_numpy()
    target = pd.read_csv(data / ('lastfm_asia_target.csv' if snap else 'lastfm_target.csv'))
    graph = nx.Graph()
    graph.add_nodes_from(target.iloc[:, 0])
    canonical = np.sort(edges, axis=1)
    graph.add_edges_from(canonical[canonical[:, 0] != canonical[:, 1]])
    graph = graph.subgraph(nodes).copy()
    if not nx.is_connected(graph):
        raise ValueError('Saved analysis population is not connected in the current graph')
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, format='csr').astype(float)
    if snap:
        features = json.loads((data / 'lastfm_asia_features.json').read_text())
        likes = pd.DataFrame([(int(u), artist) for u, artists in features.items() for artist in artists],
                             columns=['node_id', 'artist_id'])
    else:
        filename = 'lastfm_features.csv' if (data / 'lastfm_features.csv').exists() else 'lastfm_features.csv.gz'
        likes = pd.read_csv(data / filename).rename(columns={'feature_id': 'artist_id'})[['node_id', 'artist_id']]
    likes = likes[likes.node_id.isin(nodes)].drop_duplicates()
    positions = pd.Series(np.arange(len(nodes)), index=nodes)
    features = sp.csr_matrix((np.ones(len(likes)), (likes.node_id.map(positions), likes.artist_id)))
    return adjacency, jaccard_matrix(features)


def _aligned_labels(labels, reference):
    """A bijective visual relabeling; neither partition nor any metric changes."""
    k = max(labels.max(), reference.max()) + 1
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (labels, reference), 1)
    row, column = linear_sum_assignment(-counts)
    mapping = np.empty(k, dtype=int)
    mapping[row] = column
    matched = mapping[labels]
    assert len(np.unique(matched)) == len(np.unique(labels))
    assert np.array_equal(np.sort(np.unique(matched, return_counts=True)[1]),
                          np.sort(np.unique(labels, return_counts=True)[1]))
    return matched


def _network_comparison(nodes, adjacency, positions, labels):
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 5.8))
    palette = np.asarray(plt.get_cmap('tab20').colors)[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1]]
    reference = labels['SCORE'].to_numpy()
    k = int(max(labels[method].max() for method in CORE)) + 1
    if k > len(palette):
        palette = np.asarray(plt.get_cmap('turbo')(np.linspace(.03, .97, k)))
    u, v = sp.triu(adjacency, k=1).nonzero()
    segments = np.stack([positions[u], positions[v]], axis=1)
    lower, upper = positions.min(0), positions.max(0)
    pad = (upper - lower) * .035
    for ax, method in zip(axes, CORE):
        original = labels[method].to_numpy()
        matched = _aligned_labels(original, reference)
        sizes = np.bincount(original)
        ax.add_collection(LineCollection(segments, colors='#707982', linewidths=.11,
                                        alpha=.15, rasterized=True, zorder=1))
        ax.scatter(positions[:, 0], positions[:, 1], c=palette[matched], s=3.6,
                   alpha=.9, linewidths=0, rasterized=True, zorder=2)
        ax.set(xlim=(lower[0]-pad[0], upper[0]+pad[0]), ylim=(lower[1]-pad[1], upper[1]+pad[1]))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{method}\nK={len(sizes)}; largest community {sizes.max()/len(nodes):.1%}',
                     color=COLORS[method], fontsize=12, pad=12)
    handles = [Line2D([], [], marker='o', ls='', color=palette[i], markersize=5,
                       label=f'SCORE C{i}') for i in range(k)]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=6, fontsize=8,
                  title='Colors matched to SCORE by maximum membership overlap (visual aid only)', title_fontsize=8)
    figure.suptitle('The same friendship network, three community assignments', fontsize=16, weight='bold', y=.985)
    figure.text(.5, .01, f'LastFM Asia (March 2020 snapshot): {len(nodes):,} users and {len(u):,} edges. Positions and node sizes are identical. '
                'Matching colors does not imply identical communities.', ha='center', fontsize=8, color=MUTED)
    figure.subplots_adjust(left=.015, right=.985, top=.84, bottom=.20, wspace=.025)
    return figure


def _music_heatmaps(labels, details):
    figure = plt.figure(figsize=(14.6, 7.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[4, 1], hspace=.09, wspace=.20,
                              left=.055, right=.92, bottom=.14, top=.85)
    matrices = {}
    for method in CORE:
        block = details[method]['between'].copy()
        np.fill_diagonal(block, details[method]['within'])
        matrices[method] = block
    limit = max(float(block.max()) for block in matrices.values())
    for j, method in enumerate(CORE):
        matrix = matrices[method]
        k = len(matrix)
        ax = figure.add_subplot(grid[0, j])
        image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=limit, interpolation='nearest')
        ax.grid(False)
        ax.set_title(f'{method}\nwithin / between mean = {details[method]["ratio"]:.2f}', fontsize=11, color=COLORS[method])
        ax.set_xticks(range(k), [])
        ax.set_yticks(range(k), [f'C{i}' for i in range(k)], fontsize=7)
        if j == 0:
            ax.set_ylabel('Community (largest first)')
        for i in range(k):
            ax.add_patch(mpl.patches.Rectangle((i-.5, i-.5), 1, 1, fill=False, lw=.8, edgecolor=INK))
            ax.text(i, i, f'{matrix[i, i]:.3f}', ha='center', va='center', fontsize=6.1,
                    color='white' if matrix[i, i] > limit*.56 else INK)
        size_ax = figure.add_subplot(grid[1, j])
        sizes = np.bincount(labels[method])
        size_ax.bar(range(k), sizes, color=COLORS[method], width=.76)
        size_ax.set_yscale('log')
        size_ax.set(ylim=(4, len(labels)), xlim=(-.5, k-.5), xlabel='Community ID (method specific)')
        size_ax.set_xticks(range(k), [str(i) for i in range(k)], fontsize=7)
        size_ax.set_yticks([10, 100, 1000], ['10', '100', '1,000'], fontsize=7)
        if j == 0:
            size_ax.set_ylabel('Users (log scale)', fontsize=9)
        size_ax.grid(axis='x', visible=False)
    color_ax = figure.add_axes([.94, .39, .012, .38])
    figure.colorbar(image, cax=color_ax, label='Mean Jaccard similarity')
    figure.suptitle('Music similarity inside and between graph-derived communities', fontsize=16, weight='bold', y=.97)
    figure.text(.5, .025, 'LastFM Asia (March 2020 snapshot); same color scale in all panels. Diagonal boxes: within-community means; '
                'off-diagonal cells: between-community means.\nCommunity IDs are method specific. Lower panels show sizes; cells are descriptive means, not significance tests.',
                ha='center', fontsize=8, color=MUTED, linespacing=1.5)
    return figure


def _weighting_comparison(inference):
    figure, ax = plt.subplots(figsize=(11.5, 5.8))
    reference = inference[inference.strata.eq('degree_activity_q5') & inference.Method.isin(CORE)]
    for j, method in enumerate(CORE):
        rows = reference[reference.Method.eq(method)].set_index('statistic')
        equal, pair = rows.loc['equal'], rows.loc['pair']
        ax.plot([equal['T'], pair['T']], [j-.14, j+.14], color=COLORS[method], alpha=.45, lw=2)
        for statistic, row, y, marker in [('equal', equal, j-.14, 'D'), ('pair', pair, j+.14, 'o')]:
            ax.plot([row.null_q025, row.null_q975], [y, y], color='#aeb6bd', lw=6, solid_capstyle='butt', zorder=2)
            ax.plot(row.null_mean, y, '|', color=INK, ms=9, zorder=3)
            ax.scatter(row['T'], y, color=COLORS[method], marker=marker, s=70, edgecolor='white', linewidth=.8, zorder=4)
            ax.annotate(f'{row["T"]:+.4f}', (row['T'], y), xytext=(0, -16 if statistic=='pair' else 11),
                        textcoords='offset points', ha='center', fontsize=9, color=COLORS[method], weight='bold')
    ax.axvline(0, color=INK, lw=1, ls='--', zorder=0)
    ax.set_yticks(range(3), CORE, fontsize=11)
    ax.invert_yaxis()
    limits = np.r_[reference['T'], reference.null_q025, reference.null_q975, 0]
    span = float(limits.max() - limits.min())
    ax.set(ylim=(2.65, -.65), xlim=(limits.min()-.12*span, limits.max()+.12*span), xlabel='Within minus between mean Jaccard similarity (T)',
           ylabel='Community detection method')
    ax.grid(axis='y', visible=False)
    handles = [Line2D([], [], marker='D', ls='', color=INK, label='Equal-community weighting'),
               Line2D([], [], marker='o', ls='', color=INK, label='Equal-user-pair weighting'),
               Line2D([], [], color='#aeb6bd', lw=5, label='Central 95% permutation range')]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .09), ncol=3, fontsize=8)
    figure.suptitle('Does the conclusion depend on how communities are weighted?', fontsize=15, weight='bold', y=.97)
    sign_table = reference.pivot(index='Method', columns='statistic', values='T')
    sign_reversal = sign_table.loc['Spectral', 'equal'] > 0 > sign_table.loc['Spectral', 'pair']
    others_positive = (sign_table.loc[['SCORE', 'Louvain']] > 0).all().all()
    subtitle = ('Spectral changes sign; SCORE and Louvain retain positive contrasts under both estimands.'
                if sign_reversal and others_positive else 'The two weighting schemes target different community/music comparisons.')
    figure.text(.5, .90, subtitle,
                ha='center', color=MUTED, fontsize=10)
    b = int(reference.B.iloc[0])
    figure.text(.5, .025, f'LastFM Asia (March 2020 snapshot); B={b}. Grey ranges are permutation references, not effect confidence intervals.\n'
                 'A negative observed contrast is descriptive; the reported one-sided test targets a positive association.',
                 ha='center', fontsize=8, color=MUTED, linespacing=1.5)
    figure.subplots_adjust(left=.12, right=.98, top=.82, bottom=.25)
    return figure


def generate_figures(project_root, run_root=None, similarity=None, adjacency=None, nodes=None, show=False):
    root = Path(project_root).resolve()
    run = Path(run_root).resolve() if run_root is not None else root
    figures = run / 'figures'
    outputs = run / 'outputs'
    tables = run / 'tables'
    labels = pd.read_csv(outputs / 'partitions.csv', index_col='node_id')
    saved_nodes = labels.index.to_numpy()
    if nodes is not None and not np.array_equal(nodes, saved_nodes):
        raise ValueError('Plot population/order differs from the saved fitted partitions')
    nodes = saved_nodes
    if not np.all(nodes[:-1] < nodes[1:]):
        raise ValueError('Retained node IDs must be ordered and unique')
    if (similarity is None) != (adjacency is None):
        raise ValueError('Supply both adjacency and similarity, or neither')
    if similarity is None:
        adjacency, similarity = _load_matrices(root, nodes)
    layout = np.load(outputs / 'layout.npy')
    layout_meta = json.loads((outputs / 'layout_metadata.json').read_text())
    if layout_meta['signature'] != _graph_signature(nodes, adjacency) or layout.shape != (len(nodes), 2):
        raise ValueError('Network layout does not match the graph and ordered retained users')
    summary = pd.read_csv(tables / 'tab06_main_test.csv', index_col='Method')
    inference = pd.read_csv(tables / 'tab16_inference_all_references.csv')
    details = {method: T_stat(similarity, labels[method].to_numpy(), full=True) for method in CORE}
    for method in CORE:
        for metric in ['S_same', 'S_diff', 'T']:
            if not np.isclose(details[method][metric], summary.loc[method, metric], atol=1e-11, rtol=1e-10):
                raise ValueError(f'Plot statistic disagrees with the executed analysis: {method}/{metric}')
    style = {'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':10,
             'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':MUTED,
             'axes.labelcolor':INK, 'text.color':INK, 'xtick.color':MUTED, 'ytick.color':MUTED,
             'axes.grid':True, 'axes.axisbelow':True, 'grid.color':GRID, 'grid.linewidth':.6,
             'legend.frameon':False, 'savefig.facecolor':'white'}
    with mpl.rc_context(style):
        rendered = [
            ('fig11_core_networks', _network_comparison(nodes, adjacency, layout, labels)),
            ('fig12_core_music_heatmaps', _music_heatmaps(labels, details)),
            ('fig13_weighting_sensitivity', _weighting_comparison(inference)),
        ]
        for stem, figure in rendered:
            figure.savefig(figures / (stem + '.png'), dpi=300, bbox_inches='tight')
            figure.savefig(figures / (stem + '.pdf'), dpi=300, bbox_inches='tight')
        # This is a figure review booklet, not report prose. Existing PNGs retain
        # their original labels/captions; the new three figures remain vector PDFs.
        atlas = figures / 'visualization_review.pdf'
        pages = [rendered[0], rendered[1], ('fig06_permutation_nulls', None), rendered[2],
                 ('fig07_method_comparison', None), ('fig03_K_selection', None)]
        page_notes = {
            'fig06_permutation_nulls': 'Source: LastFM Asia (March 2020 snapshot). Music evaluates graph-only communities. '
                'Permutation ranges describe the conditional null, not uncertainty about a population effect.',
            'fig07_method_comparison': 'Source: LastFM Asia (March 2020 snapshot). NMI measures partition agreement, not accuracy. '
                'The hub/size comparison is descriptive; real community labels are unknown.',
            'fig03_K_selection': 'Source: LastFM Asia (March 2020 snapshot). Tuning uses the graph alone. '
                'Stability error bars are SD across edge-deletion perturbations, not confidence intervals for true K.',
        }
        with PdfPages(atlas, metadata={'Title':'STAT 236 - Community Detection Figure Review',
                                      'Author':'Faye Fang and Grace Chen'}) as pdf:
            for stem, figure in pages:
                if figure is not None:
                    pdf.savefig(figure, dpi=300, bbox_inches='tight')
                else:
                    page, ax = plt.subplots(figsize=(14.8, 6.4))
                    ax.imshow(plt.imread(figures / (stem + '.png')))
                    ax.axis('off')
                    page.subplots_adjust(left=.015, right=.985, bottom=.12, top=.97)
                    page.text(.5, .025, page_notes[stem], ha='center', color=MUTED, fontsize=8, wrap=True)
                    pdf.savefig(page, dpi=250, bbox_inches='tight')
                    plt.close(page)
        guide = pd.DataFrame([
            ('fig11_core_networks', 'Community structure', 'Same graph and layout, colored by three graph-only partitions',
             'Matched colors are a visual relabeling, not shared or true community identities.', 'Main results'),
            ('fig12_core_music_heatmaps', 'Social circle = music circle?', 'Shared-scale block music similarity, with community sizes',
             'Cells are descriptive means; community IDs are method specific.', 'Main results or appendix'),
            ('fig06_permutation_nulls', 'Social circle = music circle?', 'Within/between mean Jaccard and degree/activity permutation reference',
             'Null ranges are not population effect confidence intervals.', 'Main results'),
            ('fig13_weighting_sensitivity', 'Does the method matter?', 'Equal-community versus equal-user-pair contrasts',
             'These are different estimands; a negative contrast does not establish a negative causal effect.', 'Main results or sensitivity'),
            ('fig07_method_comparison', 'Does the method matter?', 'Partition agreement and hub placement for the original three methods',
             'NMI measures agreement, not accuracy against known labels.', 'Methods comparison'),
            ('fig03_K_selection', 'Community structure', 'Graph-only eigengap selection and edge-deletion stability',
             'Eigengap is a heuristic; error bars are perturbation SD, not confidence intervals for true K.', 'Methodology'),
        ], columns=['figure', 'research_question', 'what_it_shows', 'statistical_caveat', 'suggested_placement'])
        guide.to_csv(tables / 'tab22_visualization_guide.csv', index=False)
        artifact_names = [stem + suffix for stem, _ in rendered for suffix in ['.png', '.pdf']] + [atlas.name]
        manifest = dict(core_methods=CORE, population_size=len(nodes), edge_count=adjacency.nnz//2,
            data_source='LastFM Asia; analysis population, graph-only fitted partitions and liked-artist sets',
            source_sha256=_hash(__file__), source_file='code/visualizations.py',
            research_questions=['Community structure from follows', 'Within/between music similarity', 'Method-choice sensitivity'],
            color_alignment='Maximum overlap with SCORE using optimal bijective linear assignment; visual relabeling only',
            metrics_checked_against='tab06_main_test.csv', figure_review_pages=len(pages),
            guide='tables/tab22_visualization_guide.csv', guide_sha256=_hash(tables / 'tab22_visualization_guide.csv'),
            artifact_sha256={name:_hash(figures/name) for name in artifact_names})
        (outputs / 'visualization_manifest.json').write_text(json.dumps(manifest, indent=2))
        if not show:
            for _, figure in rendered:
                plt.close(figure)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Use the isolated smoke outputs')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = generate_figures(project, project/'tmp/smoke' if args.smoke else project)
    print(f"Exported three research figures and a {result['figure_review_pages']}-page figure review; metrics agree with analysis.")
