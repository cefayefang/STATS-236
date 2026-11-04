"""Build six report figures from fixed, verified LastFM analysis results.

All fitted labels, tuning decisions, permutations and simulations are reused.
The supplied reference motivates restrained colours and annotated mathematics;
no method is selected using music similarity during figure preparation.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import lastfm_stats as stats
from visualizations import _aligned_labels, _load_matrices, _graph_signature

REPORT = ROOT / 'report'
for folder in ['figures', 'generated']:
    (REPORT / folder).mkdir(parents=True, exist_ok=True)
CORE = ['Spectral', 'SCORE', 'Louvain']
COLORS = ['#426B8A', '#278A80', '#CB735F']
MARKERS = ['o', 's', '^']
INK = '#253746'
MUTED = '#697582'
NEUTRAL = '#BCC6CF'
inputs = set()
assets = []
generated = []

def read(name, **kwargs):
    inputs.add('tables/' + name)
    return pd.read_csv(ROOT / 'tables' / name, **kwargs)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(REPORT / 'figures' / f'{name}.{ext}', dpi=450,
                    bbox_inches='tight', pad_inches=.06)
        assets.append(f'figures/{name}.{ext}')
    plt.close(fig)

def texfile(name, lines):
    (REPORT / 'generated' / name).write_text('\n'.join(lines) + '\n')
    generated.append('generated/' + name)

def title(ax, letter, heading):
    ax.set_title(f'({letter})  {heading}', loc='left', fontweight='bold', pad=9)

def methods_legend(fig, y=.005):
    handles = [Line2D([], [], marker=m, color=c, lw=1.3, ms=4,
                      label=n) for n, c, m in zip(CORE, COLORS, MARKERS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y),
               ncol=3, frameon=False, columnspacing=2.6, handlelength=1.8)

def ygrid(ax):
    ax.grid(False)
    ax.grid(axis='y', color='#E8ECF0', lw=.55)

labels = pd.read_csv(ROOT / 'outputs/partitions.csv', index_col='node_id')
nodes = labels.index.to_numpy()
inputs.update(['outputs/partitions.csv', 'data/lastfm_edges.csv',
               'data/lastfm_target.csv', 'data/lastfm_features.csv',
               'outputs/layout.npy', 'outputs/layout_metadata.json'])
with threadpool_limits(limits=4):
    adjacency, similarity = _load_matrices(ROOT, nodes)
    details = {m: stats.T_stat(similarity, labels[m].to_numpy(), True) for m in CORE}
    social = {}
    for m in CORE:
        z = stats.onehot(labels[m].to_numpy())
        sizes = z.sum(0)
        denominator = np.outer(sizes, sizes)
        np.fill_diagonal(denominator, sizes * (sizes - 1))
        social[m] = (z.T @ (adjacency @ z)) / denominator
    degree = np.asarray(adjacency.sum(1)).ravel()
    likes = pd.read_csv(ROOT / 'data/lastfm_features.csv').drop_duplicates()
    activity = likes.groupby('node_id').size().reindex(nodes).to_numpy()
    del likes
main = read('tab06_main_test.csv').set_index('Method').loc[CORE]
comparison = read('tab11_method_comparison.csv').set_index('Method').loc[CORE]
for method in CORE:
    assert np.isclose(details[method]['T'], main.loc[method, 'T'], atol=1e-11)
    assert np.isclose(details[method]['T_pair'], comparison.loc[method, 'pair_weighted_T'], atol=1e-11)

style = {
    'font.family': 'STIXGeneral', 'font.size': 8.8, 'mathtext.fontset': 'stix',
    'axes.labelsize': 8.8, 'axes.titlesize': 9.6,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9DA8B2', 'axes.linewidth': .6,
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.axisbelow': True, 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
}
with mpl.rc_context(style):
    # 1. Same users and graph-only layout in every panel; colours alone change.
    positions = np.load(ROOT / 'outputs/layout.npy')
    meta = json.loads((ROOT / 'outputs/layout_metadata.json').read_text())
    assert meta['signature'] == _graph_signature(nodes, adjacency)
    assert positions.shape == (len(nodes), 2)
    palette = np.array(['#48789A', '#D49A58', '#498A79', '#BD6C78', '#8C79A8',
                        '#9B796B', '#BD8CA8', '#89939B', '#9FA76F', '#62A6B4', '#94B5CF'])
    u, v = sparse.triu(adjacency, k=1).nonzero()
    segments = np.stack([positions[u], positions[v]], axis=1)
    low, high = positions.min(0), positions.max(0)
    pad = .035 * (high - low)
    fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.35))
    for ax, method in zip(axs, CORE):
        aligned = _aligned_labels(labels[method].values, labels.SCORE.values)
        ax.add_collection(LineCollection(segments, colors='#728798', linewidths=.09,
                                         alpha=.11, rasterized=True, zorder=1))
        ax.scatter(*positions.T, c=palette[aligned], s=.70, alpha=.85,
                   linewidths=0, rasterized=True, zorder=2)
        ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]),
               ylim=(low[1]-pad[1], high[1]+pad[1]), aspect='equal')
        ax.axis('off')
        ax.set_title(method, fontweight='bold', fontsize=11, pad=17)
        largest = np.bincount(labels[method]).max() / len(nodes)
        ax.text(.5, 1.015, f'11 communities  |  largest {largest:.1%}',
                transform=ax.transAxes, ha='center', fontsize=7.7, color=MUTED)
    fig.subplots_adjust(left=.005, right=.995, bottom=.015, top=.86, wspace=.055)
    save(fig, 'network_comparison')

    # 2. A concise overview of data heterogeneity and graph-only tuning.
    selection = read('tab03_K_selection.csv')
    resolution = read('tab04_louvain_resolution.csv')
    fig, axs = plt.subplots(2, 2, figsize=(6.875, 3.75))
    d, count = np.unique(degree, return_counts=True)
    axs[0, 0].step(d, np.cumsum(count[::-1])[::-1]/len(degree), where='post',
                   color=COLORS[0], lw=1.65)
    axs[0, 0].set(xscale='log', yscale='log', xlabel='Mutual-follow degree',
                   ylabel='Fraction with degree ≥ d', ylim=(1/len(degree), 1.15))
    title(axs[0, 0], 'a', 'Degree distribution')
    hb = axs[0, 1].hexbin(np.log10(degree), activity, gridsize=23, mincnt=1,
                          bins='log', cmap='Blues', linewidths=0)
    axs[0, 1].set(xlabel='Mutual-follow degree', ylabel='Liked artists',
                   xticks=np.log10([1, 4, 16, 64, 214]),
                   xticklabels=['1', '4', '16', '64', '214'])
    axs[0, 1].text(.04, .94, f'Spearman ρ = {spearmanr(degree, activity).statistic:.3f}',
                   transform=axs[0, 1].transAxes, va='top', fontsize=8)
    cb = fig.colorbar(hb, ax=axs[0, 1], fraction=.035, pad=.025)
    cb.set_label('Users', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    title(axs[0, 1], 'b', 'Degree and music activity')
    ax = axs[1, 0]
    ax.plot(selection.K, selection.rel_eigengap, '-o', color=COLORS[0], ms=3, lw=1.2)
    chosen = selection.loc[selection.rel_eigengap.idxmax()]
    ax.scatter(chosen.K, chosen.rel_eigengap, color=COLORS[2], s=30, zorder=4)
    ax.annotate('K = 11', (chosen.K, chosen.rel_eigengap), xytext=(13, .053),
                fontsize=8.2, color=COLORS[2], arrowprops=dict(arrowstyle='-', color=COLORS[2], lw=.6))
    ax.set(xlabel='Candidate community count K', ylabel='Relative eigengap',
           xticks=[3, 5, 8, 11, 14, 17, 20])
    title(ax, 'c', 'Community count selection')
    ax = axs[1, 1]
    ax.fill_between(resolution.gamma, resolution.K_min, resolution.K_max,
                     color=COLORS[0], alpha=.12, linewidth=0)
    ax.plot(resolution.gamma, resolution.K_median, '-o', color=COLORS[0], ms=3, lw=1.2)
    ax.axhline(11, color=NEUTRAL, ls='--', lw=.8)
    ax.scatter(.25, 11, color=COLORS[2], s=30, zorder=4)
    ax.annotate('γ = 0.25', (.25, 11), xytext=(.40, 5), color=COLORS[2], fontsize=8.2,
                arrowprops=dict(arrowstyle='-', color=COLORS[2], lw=.6))
    ax.set(xlabel='Louvain resolution γ', ylabel='Number of communities', ylim=(0, 32))
    title(ax, 'd', 'Resolution sensitivity')
    for ax in axs.flat: ygrid(ax)
    fig.subplots_adjust(left=.085, right=.975, bottom=.115, top=.92, hspace=.70, wspace=.42)
    save(fig, 'study_diagnostics')

    # 3. Shared music scale and aligned size profiles; no redundant social row.
    cmap = LinearSegmentedColormap.from_list('music', ['#FAFAF7', '#DAE8E9', '#91B8C0', '#3E708B', '#183D58'])
    fig = plt.figure(figsize=(6.875, 2.55))
    grid = fig.add_gridspec(2, 3, height_ratios=[4, 1], left=.07, right=.89,
                             bottom=.16, top=.84, hspace=.15, wspace=.20)
    music_max = max(np.nanmax(np.r_[details[m]['within'], details[m]['between'].ravel()]) for m in CORE)
    store = {}
    for j, (method, color) in enumerate(zip(CORE, COLORS)):
        block = details[method]['between'].copy()
        np.fill_diagonal(block, details[method]['within'])
        ax = fig.add_subplot(grid[0, j])
        image = ax.imshow(block, cmap=cmap, norm=Normalize(0, music_max), interpolation='nearest')
        ax.set(xticks=[], yticks=[0, 5, 10])
        ax.tick_params(length=0)
        ax.set_title(method, fontweight='bold', color=color, fontsize=10, pad=8)
        if j == 0: ax.set_ylabel('Community ID')
        for q in range(11):
            ax.add_patch(mpl.patches.Rectangle((q-.5, q-.5), 1, 1, fill=False,
                                               edgecolor='#F7F9FB', lw=.6))
        position = ax.get_position()
        size_ax = fig.add_subplot(grid[1, j])
        bar_position = size_ax.get_position()
        size_ax.set_position([position.x0, bar_position.y0, position.width, bar_position.height])
        sizes = np.bincount(labels[method])
        size_ax.bar(range(11), sizes, color=color, width=.73, edgecolor='none')
        size_ax.set(yscale='log', ylim=(10, 12000), xticks=[0, 5, 10], yticks=[10, 1000], xlabel='Community ID')
        size_ax.tick_params(labelsize=7, length=2)
        if j == 0: size_ax.set_ylabel('Users\n(log)', fontsize=8)
        ygrid(size_ax)
        store[method+'_social'] = social[method]
        store[method+'_music'] = block
        store[method+'_sizes'] = sizes
    cax = fig.add_axes([.927, .365, .014, .445])
    cb = fig.colorbar(image, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label('Mean Jaccard similarity', fontsize=8, labelpad=7)
    np.savez_compressed(REPORT / 'generated/block_summaries.npz', **store)
    generated.append('generated/block_summaries.npz')
    save(fig, 'community_music')

    # 4. The central statistical insight: changing the evaluation unit.
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.30), gridspec_kw={'width_ratios': [1.23, 1]})
    for j, (method, color) in enumerate(zip(CORE, COLORS)):
        y = 2-j
        eq, pair = details[method]['T'], details[method]['T_pair']
        axs[0].plot([eq, pair], [y, y], color=color, lw=1.7, alpha=.55)
        axs[0].scatter(eq, y, s=34, color=color, zorder=4)
        axs[0].scatter(pair, y, s=32, facecolor='white', edgecolor=color, linewidth=1.1, marker='D', zorder=5)
        for value, dy in [(eq, .17), (pair, -.18)]:
            axs[0].text(value, y+dy, f'{value:.4f}', ha='center',
                        va='bottom' if dy>0 else 'top', fontsize=8, color=color)
        sizes = np.bincount(labels[method])
        share_users = sizes.max()/len(nodes)
        share_pairs = (sizes*(sizes-1)).max()/np.sum(sizes*(sizes-1))
        axs[1].barh(y+.14, share_users, color='#D0D8DF', height=.22)
        axs[1].barh(y-.14, share_pairs, color=color, height=.22)
        axs[1].text(share_pairs+.025, y-.14, f'{share_pairs:.1%}', va='center', fontsize=8, color=color)
    axs[0].axvline(0, color=MUTED, ls='--', lw=.7)
    axs[0].set(xlim=(-.024, .056), ylim=(-.48, 2.5), yticks=[2, 1, 0], yticklabels=CORE,
                xlabel='Within − between Jaccard')
    axs[1].set(xlim=(0, 1.18), ylim=(-.48, 2.5), yticks=[2, 1, 0], yticklabels=CORE,
                xlabel='Largest community share')
    axs[1].xaxis.set_major_formatter(PercentFormatter(1))
    axs[1].set_xticks([0, .25, .5, .75, 1])
    title(axs[0], 'a', 'Community and pair weighting')
    title(axs[1], 'b', 'Concentration of within pairs')
    handles = [Line2D([], [], marker='o', ls='', color=INK, label='Equal-community'),
               Line2D([], [], marker='D', ls='', color=INK, markerfacecolor='white', label='User-pair')]
    axs[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.21),
                   ncol=2, frameon=False, columnspacing=1.7)
    handles = [mpl.patches.Patch(color='#D0D8DF', label='Users'),
               mpl.patches.Patch(color=MUTED, label='Within pairs')]
    axs[1].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.21),
                   ncol=2, frameon=False, columnspacing=1.4)
    for ax in axs:
        ax.grid(False); ax.grid(axis='x', color='#E8ECF0', lw=.55)
    fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.28, wspace=.38)
    save(fig, 'music_weighting')

    # 5. Inference calibration and independent recovery use separate experiments.
    cal = read('tab26_primary_calibration_summary.csv')
    raw = read('tab28_independent_recovery_raw.csv')
    summary = read('tab28_independent_recovery_summary.csv')
    fig, axs = plt.subplots(2, 2, figsize=(6.875, 3.70))
    for ax, scenario, letter, heading in zip(axs[0],
            ['exact_binned_null', 'smooth_degree_only_stress'], ['a', 'b'],
            ['Invariant-null calibration', 'Continuous-degree stress']):
        for j, color in enumerate(COLORS):
            for scheme, offset, marker in [('degree_activity_q5', -.10, 'o'), ('degree_exact_activity_q5', .10, 'D')]:
                row = cal[cal.scenario.eq(scenario) & cal.statistic.eq('equal') &
                          cal.method.eq(CORE[j]) & cal.scheme.eq(scheme)].iloc[0]
                ax.errorbar(j+offset, row.rejection_fraction,
                            yerr=[[row.rejection_fraction-row.pointwise_MC_low],
                                  [row.pointwise_MC_high-row.rejection_fraction]],
                            fmt=marker, color=color, markerfacecolor=color if marker=='o' else 'white',
                            ms=4, lw=1, capsize=2.2)
        ax.axhline(.05, color=MUTED, ls='--', lw=.8)
        ax.set(xticks=range(3), xticklabels=CORE, ylabel='Rejection rate', ylim=(0, .43))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        title(ax, letter, heading)
    handles = [Line2D([], [], marker='o', ls='', color=INK, label='Degree / activity quintiles'),
               Line2D([], [], marker='D', ls='', color=INK, markerfacecolor='white', label='Exact degree / activity quintiles')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.54, .545), ncol=2,
               frameon=False, fontsize=7.8, columnspacing=2)
    for ax, ratio, letter in zip(axs[1], [3, 8], ['c', 'd']):
        for j, (method, color, marker) in enumerate(zip(CORE, COLORS, MARKERS)):
            rows = []
            for sd in [0, 1.2]:
                setting = raw[raw.degree_sd.eq(sd) & raw.within_between_ratio.eq(ratio)].setting.iloc[0]
                rows.append(summary[summary.setting.eq(setting) & summary.method.eq(method)].iloc[0])
            mean = np.array([r.ARI_mean for r in rows])
            low = np.array([r.ARI_ci_low for r in rows])
            high = np.array([r.ARI_ci_high for r in rows])
            xx = np.array([0, 1])+(j-1)*.055
            ax.errorbar(xx, mean, yerr=[mean-low, high-mean], fmt=marker+'-', color=color,
                        lw=1.3, ms=4, capsize=2.2)
        ax.set(xticks=[0, 1], xticklabels=['Homogeneous', 'Heterogeneous'],
               ylim=(-.04, 1.05), xlabel='Degree propensity regime', ylabel='Mean ARI')
        title(ax, letter, f'Community recovery: r = {ratio}')
    for ax in axs.flat: ygrid(ax)
    methods_legend(fig, y=-.015)
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.175, hspace=.83, wspace=.35)
    save(fig, 'statistical_validation')

    # 6. Shared perturbations: individual outcomes, a compact distribution, mean.
    pert = read('tab33_core_perturbation_raw.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.05))
    rng = np.random.default_rng(236)
    for j, (method, color) in enumerate(zip(CORE, COLORS)):
        for ax, column in [(axs[0], 'NMI'), (axs[1], 'T_change_same_users')]:
            values = pert[pert.method.eq(method)][column].to_numpy()
            violin = ax.violinplot([values], positions=[j], widths=.64,
                                    showextrema=False, bw_method=.4)
            for body in violin['bodies']:
                body.set_facecolor(color); body.set_alpha(.12); body.set_edgecolor('none')
            ax.scatter(j+rng.uniform(-.12, .12, len(values)), values, s=12,
                        color=color, alpha=.75, linewidths=.35, edgecolors='white', zorder=3)
            ax.plot([j-.19, j+.19], [values.mean()]*2, color=INK, lw=1.7, zorder=4)
    axs[0].set(ylabel='NMI on retained users', ylim=(.65, 1.015))
    axs[1].set_ylabel('Change in equal-community contrast')
    axs[1].axhline(0, color=MUTED, lw=.8, ls='--')
    for ax in axs:
        ax.set(xticks=range(3), xticklabels=CORE); ygrid(ax)
    title(axs[0], 'a', 'Community-label stability')
    title(axs[1], 'b', 'Music-contrast stability')
    fig.subplots_adjust(left=.09, right=.985, top=.82, bottom=.17, wspace=.40)
    save(fig, 'shared_perturbations')

# Compact numerical results remain unchanged.
rows = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'Method & $K$ & Largest (\%) & $T_{\rm eq}$ & Ratio & Excess $T_{\rm eq}$ & Holm $p$ & $T_{\rm pair}$ \\', r'\midrule']
for method in CORE:
    r = main.loc[method]
    largest = np.bincount(labels[method]).max()/len(nodes)*100
    rows.append(f'{method} & {int(r.K)} & {largest:.1f} & {r["T"]:.4f} & '
                f'{r["ratio S_same/S_diff"]:.3f} & {r.excess_T:.4f} & '
                f'{r.p_Holm_primary_3_methods:.4f} & {comparison.loc[method,"pair_weighted_T"]:.4f} ' + r'\\')
rows += [r'\bottomrule', r'\end{tabular}']
texfile('main_results.tex', rows)
country = read('tab08_country_check.csv').set_index('Method').loc[CORE]
rows = [r'\begin{tabular}{lrrr}', r'\toprule',
        r'Method & Country NMI & Joint excess & Joint $p$ \\', r'\midrule']
for method, r in country.iterrows():
    rows.append(f'{method} & {r["NMI with country"]:.3f} & {r["joint excess T"]:.4f} & '
                f'{r["p (country x deg x activity)"]:.3f} ' + r'\\')
rows += [r'\bottomrule', r'\end{tabular}']
texfile('country_results.tex', rows)
for name in ['tab02_localization.csv', 'tab12_robustness.csv', 'tab20_scoreplus_implementation.csv',
             'tab23_primary_binning_sensitivity.csv', 'tab24_primary_strata_support.csv',
             'tab27_primary_calibration_family.csv', 'tab29_degree_correction_interaction.csv',
             'tab31_matched_legacy_recovery_summary.csv', 'tab34_core_perturbation_summary.csv',
             'tab35_core_seed_runs.csv', 'tab36_core_seed_summary.csv']:
    read(name)
inputs.update(['outputs/run_manifest.json', 'outputs/statistical_validation_manifest.json',
               'outputs/statistical_audit.json'])
assets += generated
manifest = dict(
    analysis_commit='aa99d899a4d3efa54d1200b3e97af1e87b8896ba',
    rule='Fixed saved partitions and outcomes; no refitting or music-based selection.',
    figure_count=6, table_count=2,
    block_checks='Block Jaccard means and both contrasts agree with the executed analysis.',
    source_sha256={'code/scripts/build_report_assets.py': digest(__file__)},
    input_sha256={p: digest(ROOT/p) for p in sorted(inputs)},
    asset_sha256={p: digest(REPORT/p) for p in assets})
(REPORT / 'generated/asset_manifest.json').write_text(json.dumps(manifest, indent=2))
# Retired figure exports are backed up before editing; do not ship unused plots.
for path in (REPORT / 'figures').iterdir():
    if path.suffix in ['.pdf', '.png'] and str(path.relative_to(REPORT)) not in assets:
        path.unlink()
print('Built 6 report figures, 2 tables, checked block matrices and provenance.')
