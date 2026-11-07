import json
import subprocess
from pathlib import Path
import nbformat

path = Path('STAT236_LastFM_Community_Music.ipynb')
nb = nbformat.reads(subprocess.check_output(['git', 'show', 'HEAD:' + str(path)], text=True), as_version=4)
c = nb.cells
for cell in c:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("'SCORE+'", 'MAIN_METHOD')
        cell.outputs = []
        cell.execution_count = None

def md(i, text):
    c[i] = nbformat.v4.new_markdown_cell(text.strip())

def code(i, text):
    c[i] = nbformat.v4.new_code_cell(text.strip())

def add_md(text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text.strip()))

def add_code(text):
    nb.cells.append(nbformat.v4.new_code_cell(text.strip()))

md(0, r'''
# Is a Social Circle Also a Music Circle?
## Community detection, degree heterogeneity, and conditional music similarity
**STAT 236, Fall 2026 · Faye Fang, Grace Chen**

**Question.** Do communities estimated from mutual follows alone have more similar liked-artist sets than different communities? Does the association persist relative to a null that preserves country and coarse degree/activity groups?

This notebook is the executable analysis and supporting material for the report. It does not constitute the report. Every reported number is recomputed below; original results are not treated as targets.

**Analysis status.** This is an exploratory reanalysis of an already inspected dataset, not a preregistered confirmatory study. Network fitting and tuning use only the graph, not artist identities, similarities, or country. Artist counts and country are used later for evaluation strata. A stabilized, explicitly **truncated SCORE+ variant** is the working main method; the untruncated paper algorithm is a separate comparator. Their difference is assessed, not hidden.

**Route through the analysis:** §1 data and attrition; §2 sparse-network model, algorithms and tuning; §3 Jaccard similarity; §4 degree/activity permutation reference; §5 joint country/degree/activity inference; §6 method comparisons; §7 sensitivity; §8 network illustrations; Appendix A recovery simulations; §9 reproducibility; Appendix B calibration, diagnostics and report evidence.

**Course emphasis:** low-rank signal plus sparse noise, SBM versus degree-corrected block models, eigenvector localization, regularization, nuisance adjustment, dependent dyads, exchangeability, and simulation-based method assessment. Course slide numbers from the original notebook have not been independently verified.
''')

setup = c[2].source
setup = setup.replace('import os, json, time, random, warnings, platform', 'import os, json, time, random, platform, hashlib, importlib.metadata\nfrom pathlib import Path\nfrom datetime import datetime, timezone\nfrom IPython.display import display, Markdown\nfrom threadpoolctl import threadpool_limits\nfrom scipy.stats import t as student_t\nimport lastfm_stats as stats')
setup = setup.replace("warnings.filterwarnings('ignore', category=RuntimeWarning)", 'threadpool_limits(limits=4)\nRUN_STARTED = datetime.now(timezone.utc).isoformat()\nPROFILE = os.environ.get("STAT236_PROFILE", "full")\nassert PROFILE in {"full", "smoke"}')
setup = setup.replace("DATA_DIR, FIG_DIR, TAB_DIR, OUT_DIR = 'data', 'figures', 'tables', 'outputs'", "DATA_DIR = 'data'\nRUN_ROOT = Path('tmp/smoke') if PROFILE == 'smoke' else Path('.')\nFIG_DIR, TAB_DIR, OUT_DIR = [str(RUN_ROOT / d) for d in ('figures', 'tables', 'outputs')]")
setup = setup.replace('B_MAIN   = 1000', 'B_MAIN   = 1999').replace('B_ROBUST = 200', 'B_ROBUST = 199')
setup = setup.replace('K_RANGE  = range(3, 21)', 'K_RANGE  = range(3, 21)\nif PROFILE == "smoke":\n    B_MAIN, B_ROBUST, N_SUB, N_SIM = 9, 5, 2, 1\nMAIN_METHOD = "SCORE+ (truncated)"\nPAPER_METHOD = "SCORE+ (paper)"')
setup = setup.replace("METHOD_ORDER = ['Spectral', 'SCORE', MAIN_METHOD, 'Louvain']", "METHOD_ORDER = ['Spectral', 'SCORE', PAPER_METHOD, MAIN_METHOD, 'Louvain']")
setup = setup.replace("METHOD_COLOR = {MAIN_METHOD: '#2a78d6',", "METHOD_COLOR = {PAPER_METHOD: '#6250d6', MAIN_METHOD: '#2a78d6',")
setup = setup.replace("fig.savefig(os.path.join(FIG_DIR, name + '.png'))", "fig.savefig(os.path.join(FIG_DIR, name + '.png'))\n    fig.savefig(os.path.join(FIG_DIR, name + '.pdf'))")
setup = setup.replace("df.to_csv(os.path.join(TAB_DIR, name + '.csv'), index=index)", "df.to_csv(os.path.join(TAB_DIR, name + '.csv'), index=index, float_format='%.12g')")
setup += '\nprint("Profile:", PROFILE, "| main permutations:", B_MAIN, "| sensitivity permutations:", B_ROBUST)'
code(2, setup)

code(5, c[5].source + r'''

# Fail early on alignment or incidence errors instead of producing plausible output.
assert target.node_id.is_unique
assert set(edges_raw.u) | set(edges_raw.v) <= set(target.node_id)
assert set(likes_raw.node_id) <= set(target.node_id)
assert nx.is_connected(G) and np.all(deg > 0)
assert A.shape == (n, n) and (A != A.T).nnz == 0 and np.all(A.diagonal() == 0)
assert np.all(X.data == 1) and np.all(n_art > 0)
assert np.array_equal(nodes, np.sort(nodes))

inclusion = target[['node_id', 'country']].copy()
inclusion['has_music'] = inclusion.node_id.isin(music_users)
inclusion['in_lcc_analysis'] = inclusion.node_id.isin(nodes)
inclusion['raw_degree'] = inclusion.node_id.map(dict(G_full.degree()))
attrition = inclusion.groupby('country').agg(raw_users=('node_id', 'size'), music_users=('has_music', 'sum'),
    analysis_users=('in_lcc_analysis', 'sum'), mean_raw_degree=('raw_degree', 'mean'))
attrition['retention_fraction'] = attrition.analysis_users / attrition.raw_users
savetab(attrition, 'tab14_attrition_by_country', index=True)
''')
md(6, '''**Sampling population.** The analysis concerns the largest connected component among observed users with at least one recorded artist. The retention table (`tab14_attrition_by_country.csv`) documents exclusions by country. Missing music need not be missing at random, and excluded users cannot be represented by an unqualified population claim. Artist IDs and country labels are anonymous; recorded likes are not listening histories.''')
md(8, '''**EDA interpretation.** Degree heterogeneity and sparsity motivate a degree-corrected network model. A roughly linear log-log tail is descriptive evidence, not a fitted or validated power law. The degree/activity association motivates nuisance adjustment, but coarse stratification will not remove all continuous-covariate differences.''')
md(9, r'''
## 2. Network model and community detection

For $i<j$, the Bernoulli DCBM assumes conditionally independent edges with
$$A_{ij}\mid(c,\theta,B)\sim\operatorname{Bernoulli}(\theta_i\theta_j B_{c_i c_j}),\quad A_{ii}=0.$$
The symmetric nonnegative $B$ and degree propensities $\theta_i>0$ need a scaling constraint (e.g. $\sum_{i:c_i=k}\theta_i=1$ per block); probabilities must be at most one. Up to removal of its diagonal, the signal $\Omega=\Theta ZBZ^\top\Theta$ has rank at most $K$. SBM suppresses within-block degree heterogeneity. In the population idealization, eigenvector rows are scaled by $\theta_i$; taking ratios cancels this nuisance scale. Sparse noise and tiny denominators can disrupt the empirical cancellation.

We do not assume that this real social network exactly follows a DCBM. Hard nonoverlapping communities, conditional edge independence, and model fit are limitations. Simulation tests algorithm behavior under specified generators, not the truth of the real partition.

**Algorithms (all fit without music or country):**
- **Spectral:** top-$K$ adjacency eigenvectors, then k-means; an explicitly unnormalized baseline.
- **SCORE:** adjacency eigenvector ratios, truncated at $\pm\log n$, then k-means.
- **SCORE+ (paper):** $L_\tau=(D+\tau I)^{-1/2}A(D+\tau I)^{-1/2}$, optional extra eigenvector, eigenvalue-weighted ratios, then k-means **without truncation**.
- **SCORE+ (truncated):** same embedding with ratios clipped at $\pm\log n$ before k-means. This is an explicit stabilization of the paper algorithm, and the working main method inherited from the original analysis. Its benefits and costs must be assessed separately.
- **Louvain:** modularity optimization with resolution chosen to approximate the graph-selected $K$.

We use $\tau=\bar d$ and assess the published $0.1d_{\max}$ setting separately. Eigenpairs are ordered by magnitude; the extra-vector rule uses $1-|\hat\lambda_{K+1}|/|\hat\lambda_K|\le0.1$. The magnitude convention is explicit, including when the two eigenvalues have different signs. No claim of a weak-signal theorem follows merely from an empirical small eigengap. See [Jin, Ke & Luo, §3.1](https://arxiv.org/pdf/1811.05927) for the algorithm definition.
''')
code(10, r'''
leading_eigs = stats.leading_eigs
reg_laplacian = stats.reg_laplacian
relabel_by_size = stats.relabel_by_size
score_ratios = stats.score_ratios
scoreplus_M = stats.scoreplus_M

def kmeans(Y, K, seed=SEED):
    return stats.kmeans(Y, K, seed=seed, n_init=N_INIT)

def spectral_labels(vecs, K, seed=SEED):
    return kmeans(vecs[:, :K], K, seed)

def score_labels(vecs, K, seed=SEED):
    return kmeans(score_ratios(vecs, K), K, seed)

def scoreplus_labels(vals, vecs, K, t=0.1, seed=SEED, truncate=True):
    return stats.scoreplus_labels(vals, vecs, K, t=t, seed=seed, truncate=truncate, n_init=N_INIT)

def louvain_labels(Gx, resolution, seed=SEED, nodelist=None):
    comms = nx.community.louvain_communities(Gx, resolution=resolution, seed=seed)
    nl = nodes if nodelist is None else nodelist
    pos = {v: i for i, v in enumerate(nl)}
    lab = np.empty(len(nl), dtype=int)
    for k, community in enumerate(comms):
        lab[[pos[v] for v in community]] = k
    return relabel_by_size(lab)
''')
md(11, r'''### 2.1 Localization and numerical diagnostics
Compare mass in the 50 largest coordinates and $n\sum_i\hat\xi_j(i)^4$ (equal-magnitude coordinates give 1). Large ratios and frequent clipping diagnose embedding sensitivity. Positive power iteration refines the leading denominator entrywise; the numerical residual and all eigenvalues are saved. Localization is an observed diagnosis, not proof that a theorem's assumptions fail.''')
code(12, c[12].source + r'''

eigen_diagnostics = []
for name, matrix, values, vectors in [('A', A, vals_A, vecs_A), ('L_tau', L, vals_L, vecs_L)]:
    for j, value in enumerate(values):
        residual = np.linalg.norm(matrix @ vectors[:, j] - value * vectors[:, j]) / max(abs(value), 1e-15)
        eigen_diagnostics.append(dict(matrix=name, rank=j + 1, eigenvalue=value, relative_residual=residual,
                                      minimum_abs_coordinate=np.abs(vectors[:, j]).min()))
savetab(pd.DataFrame(eigen_diagnostics), 'tab15_eigen_diagnostics')
assert np.all(vecs_A[:, 0] > 0) and np.all(vecs_L[:, 0] > 0)
assert max(row['relative_residual'] for row in eigen_diagnostics) < 1e-7
''')
md(14, '''**Interpretation rule.** Use the computed localization table to compare adjacency and regularized embeddings. Do not equate the fraction of clipped unweighted SCORE ratios with the fraction of clipped eigenvalue-weighted SCORE+ ratios; the latter is reported separately in Appendix B. Neither a lower localization metric nor a higher music lift establishes that a partition is the true one.''')
md(15, r'''### 2.2 Choosing $K$ from the graph
The selection rule is the largest **unrounded** relative eigengap of $L_\tau$ over $K=3,\ldots,20$. Community sizes and stability under independent 10% edge deletions are diagnostics, not additional undisclosed selection rules. For each perturbation, fit the induced largest connected component and compare to the original labels on the retained vertices. Error bars below are SD across graph perturbations, not confidence intervals for the true number of communities. All music similarities and country labels are excluded from tuning.''')
code(16, c[16].source.replace("np.std(stab)", "np.std(stab, ddof=1)").replace('pd.DataFrame(k_rows).round(3)', 'pd.DataFrame(k_rows)'))
md(18, '''**Interpretation rule.** Report the selected gap, community sizes, and perturbation NMI from the tables. Eigengap selection is a heuristic; it does not prove that the graph contains exactly this many communities. Sensitivity across several $K$ values is required. The extra-vector decision concerns the embedding dimension, not a test of signal strength.''')
code(20, c[20].source.replace('pd.DataFrame(gam_rows).round(3)', 'pd.DataFrame(gam_rows)'))
md(22, '''**Comparison caveat.** Louvain is not constrained to produce exactly $K$ groups. We choose its resolution using only the median group count over five seeds, then use the prespecified seed 236 for the analysis, rather than selecting a seed with a favorable music effect. Modularity values above are evaluated at resolution 1 for comparability; they are not likelihoods or proofs of recovery.''')
code(24, c[24].source.replace("'SCORE':    score_labels", "PAPER_METHOD: scoreplus_labels(vals_L, vecs_L, K_STAR, truncate=False),\n    'SCORE':    score_labels").replace('plt.subplots(1, 4, figsize=(14, 3.2)', 'plt.subplots(1, len(METHOD_ORDER), figsize=(18, 3.3)'))
md(25, '''**Interpretation rule.** Contrast size imbalance, hub placement and perturbation behavior across all five methods. In particular, the paper SCORE+ and truncated variant must both appear: any apparent improvement must be attributed to the actual implementation used. No reference labels exist for the real graph.''')
code(27, r'''
jaccard_matrix = stats.jaccard_matrix
t0 = time.time()
S = jaccard_matrix(X)
assert S.dtype == np.float64 and np.all(S.diagonal() == 0)
assert np.allclose(S, S.T, atol=1e-14) and S.min() >= 0 and S.max() <= 1
pair_count = n * (n - 1) // 2
overall_mean = S.sum() / (2 * pair_count)
eu, ev = sp.triu(A, k=1).nonzero()
friend_mean = float(S[eu, ev].mean())
print(f'Float64 Jaccard matrix {S.shape}: {S.nbytes / 1e6:.1f} MB in {time.time()-t0:.1f}s')
print(f'{pair_count:,} dependent pairs; overall mean {overall_mean:.6f}; friend mean {friend_mean:.6f}; ratio {friend_mean / overall_mean:.3f}')
''')
md(28, '''**Interpretation.** The friend/all-pair comparison is descriptive and unadjusted. Millions of dyads are not millions of independent observations; dyads share users. Artist filtering in the released data also limits claims about niche music.''')
md(29, r'''
## 4. Degree/activity reference: the statistical question and estimands

For $W=Z^\top SZ$ and community sizes $n_k$, define
$$T_{\mathrm{equal}}=\frac{1}{K_2}\sum_{k:n_k\ge2}\frac{W_{kk}}{n_k(n_k-1)}-\binom K2^{-1}\sum_{k<l}\frac{W_{kl}}{n_kn_l},$$
where $K_2$ counts nonsingleton communities. The main real-data partitions should have no singletons, and their minimum sizes are checked. Also report $R=\bar S_{same}/\bar S_{diff}$, which is a ratio of mean **Jaccard similarities**, not a percent increase in the count of shared artists.

The pair-weighted sensitivity is
$$T_{\mathrm{pair}}=\frac{\sum_kW_{kk}}{\sum_kn_k(n_k-1)}-\frac{\sum_{k\ne l}W_{kl}}{\sum_{k\ne l}n_kn_l}.$$
These are different estimands. Equal weighting describes a typical community/block pair; pair weighting describes a typical user pair. An imbalanced partition can give opposite signs; this must be discussed rather than selecting a favorable statistic.

**Conditional randomization reference.** Given the graph and fitted partition, the sharp null requires joint invariance of the assignment of entire music records under within-stratum permutations. Degree and activity use quantile bins **without splitting ties by node ID**. Conditioning is on bins, not on exact continuous covariates. The invariant null is stronger than simply $E[T]=0$; its reference distribution need not be centered at zero.

Permuting records is equivalent to inversely permuting community labels, so the full similarity matrix and overlapping dyadic dependence are retained. This avoids an iid-dyad test, but does **not** establish exchangeability of observational users. Unmeasured geography, age, language, and residual network dependence can invalidate that assumption.

Use a one-sided alternative (larger within-community similarity). Report $(1+r)/(B+1)$, exceedances $r$, $B$, a 95% binomial interval for the exhaustive-null tail probability, and $T-E_{perm}T$. The interval describes **Monte Carlo precision**, not uncertainty about a population effect. $z_{perm}$ is only a standardized distance, never a Gaussian tail probability. See [Phipson & Smyth (2010)](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
''')
code(30, r'''
onehot, T_stat, T_pairs = stats.onehot, stats.T_stat, stats.T_pairs
make_strata, perm_test = stats.make_strata, stats.perm_test
STRATA_DA = make_strata(deg, n_art, q=5)
sizes_da = np.unique(STRATA_DA, return_counts=True)[1]
print('Tie-preserving degree x activity strata:', len(sizes_da), '| sizes:', sizes_da.min(), 'to', sizes_da.max())

def inference_row(method, result, scheme, statistic='equal'):
    detail = T_stat(S, partitions[method], full=True) if statistic == 'equal' else T_pairs(S, partitions[method], full=True)
    return {'Method': method, 'strata': scheme, 'statistic': statistic, 'K': len(np.unique(partitions[method])),
            'S_same': detail['S_same'], 'S_diff': detail['S_diff'], 'ratio S_same/S_diff': detail['ratio'],
            **{key:result[key] for key in ['T', 'B', 'exceedances', 'p', 'mc_tail_ci_low', 'mc_tail_ci_high',
               'null_mean', 'null_sd', 'null_mean_mcse', 'null_q025', 'null_q975', 'excess_T', 'z']}}
''')
code(31, r'''
t0 = time.time()
main_res = {m: perm_test(S, partitions[m], STRATA_DA, B_MAIN, seed=SEED) for m in METHOD_ORDER}
print(f'{len(partitions)} x {B_MAIN} permutations (both estimands together) in {time.time()-t0:.0f}s')
main_tab = pd.DataFrame([inference_row(m, main_res[m], 'degree_activity_q5') for m in METHOD_ORDER]).set_index('Method')
main_tab = main_tab.rename(columns={'p': 'p (perm)', 'null_mean': 'null mean', 'null_sd': 'null sd'})
main_tab['p_Holm_5_methods'] = stats.holm_adjust(main_tab['p (perm)'])
savetab(main_tab, 'tab06_main_test', index=True)
main_tab.round(5)
''')
md(32, '''**Inference hierarchy.** This degree/activity reference is a supporting analysis. The stronger country × degree × activity reference in §5 is the primary conditional analysis for report preparation. Both weighting schemes and all five methods are retained. The primary multiplicity family therefore contains 10 joint-strata tests; Holm adjustment is applied to that family. This hierarchy was set during an exploratory reanalysis after prior results were available, so it is not an independent validation dataset.''')
md(35, '''**Community descriptions.** The block similarity matrix and community characteristics are descriptive. They do not supply independent community-level significance tests. Compare the within-community similarity to the appropriate between-community mean and inspect country composition before attaching a substantive interpretation. Anonymous artist IDs do not identify a genre.''')
md(36, r'''
## 5. Country and the primary conditional analysis

Report three distinct permutation references: degree × activity (5 quantile bins per variable), country only, and **country × degree × activity jointly** (3 quantile bins per continuous covariate). Bins are global and then crossed with country. Finer 5- and 7-bin joint references are sensitivity checks; they can generate small or unmixed cells, so their randomization support is diagnosed.

For each reference, report the observed contrast and its excess over the permutation mean. The country-only null mean divided by observed $T$ is a **descriptive null-reproduction ratio**. It is not a fraction causally explained by country, can fall outside $[0,1]$, and does not decompose country and social influence. A residual association does not identify selection versus influence or establish homophily beyond all confounding.
''')
code(37, r'''
STRATA_C = pd.factorize(country)[0]
STRATA_CDA = stats.combine_strata(STRATA_C, make_strata(deg, n_art, q=3))
t0 = time.time()
country_res = {m: perm_test(S, partitions[m], STRATA_C, B_MAIN, seed=SEED + 1) for m in METHOD_ORDER}
country_res2 = {m: perm_test(S, partitions[m], STRATA_CDA, B_MAIN, seed=SEED + 2) for m in METHOD_ORDER}
print(f'Country and joint references in {time.time()-t0:.0f}s')
inference_rows = []
null_arrays = {}
for scheme, results in [('degree_activity_q5', main_res), ('country', country_res), ('country_degree_activity_q3', country_res2)]:
    for m in METHOD_ORDER:
        for statistic, r in [('equal', results[m]), ('pair', results[m]['pair_result'])]:
            inference_rows.append(inference_row(m, r, scheme, statistic))
            null_arrays[f'{scheme}__{m}__{statistic}'] = r['null']
inference_tab = pd.DataFrame(inference_rows)
inference_tab['p_Holm_within_reference_10_tests'] = inference_tab.groupby('strata')['p'].transform(lambda v: stats.holm_adjust(v.values))
inference_tab['primary_family'] = inference_tab.strata.eq('country_degree_activity_q3')
savetab(inference_tab, 'tab16_inference_all_references')
np.savez_compressed(os.path.join(OUT_DIR, 'permutation_nulls.npz'), **null_arrays)
country_tab = pd.DataFrame([{
    'Method': m, 'NMI with country': NMI(country, partitions[m]), 'T': r['T'],
    'null mean (within-country)': r['null_mean'], 'country null reproduction ratio (not causal)': r['null_mean'] / r['T'],
    'z (within-country)': r['z'], 'p (within-country)': r['p'],
    'joint null mean': country_res2[m]['null_mean'], 'joint excess T': country_res2[m]['excess_T'],
    'z (country x deg x activity)': country_res2[m]['z'], 'p (country x deg x activity)': country_res2[m]['p'],
} for m, r in country_res.items()]).set_index('Method').loc[METHOD_ORDER]
savetab(country_tab, 'tab08_country_check', index=True)
display(inference_tab[inference_tab.primary_family][['Method', 'statistic', 'T', 'excess_T', 'p', 'p_Holm_within_reference_10_tests', 'mc_tail_ci_high']].round(6))
country_tab.round(5)
''')
code(38, r'''
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for a, title, results in zip(axs, ['Degree x activity', 'Country', 'Country x degree x activity'], [main_res, country_res, country_res2]):
    for j, m in enumerate(METHOD_ORDER):
        r = results[m]
        a.plot([r['null_q025'], r['null_q975']], [j, j], color=OTHER, lw=5, solid_capstyle='butt')
        a.scatter(r['null_mean'], j, color=INK2, marker='|', s=100, zorder=3)
        a.scatter(r['T'], j, color=METHOD_COLOR[m], s=40, zorder=4)
    a.axvline(0, color=GRID, lw=1)
    a.set(title=title, xlabel='Equal-community Jaccard contrast T')
    a.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
axs[0].set_ylabel('Community detection method')
fig.suptitle('Observed contrast (points) and central 95% permutation reference (grey bars)', fontsize=11)
fig.text(0.5, -0.015, f'LastFM Asia analysis sample; B={B_MAIN} per reference. Bars are null ranges, not effect confidence intervals.', ha='center', fontsize=9)
fig.tight_layout(); savefig(fig, 'fig06_permutation_nulls'); plt.show()
''')
md(39, '''**Interpretation rule.** Lead with the main method's joint-strata excess and adjusted permutation p-value, then show the pair-weighted sensitivity and competing methods. Rejection is relative to the stated invariant null and its exchangeability assumption. The difference between two null means is not a causal attribution. A small p-value cannot substitute for an effect size, a defensible null, or external validation.''')
code(41, c[41].source.replace("'share of T explained by country'", "'country null reproduction ratio (not causal)'"))
code(42, c[42].source.replace("figsize=(13, 4.6)", "figsize=(15, 5.2)").replace('a.set_ylim', 'a.set_ylim'))
md(43, '''**Method comparison.** Higher external music coherence is not a recovery metric, because music was not the ground truth for communities. NMI quantifies agreement, not accuracy. Relate size imbalance and hub placement to the two weighting schemes; use the planted-label simulations for recovery comparisons and report where the truncation adjustment helps or hurts.''')
md(44, r'''## 7. Sensitivity analysis
These checks are exploratory and not a search for the largest effect. Report changes in $T$, Jaccard ratio and null-centered excess; all permutation counts and exceedances are saved. The coarse resolution $1/(B_{robust}+1)$ limits p-values. The separate Appendix B checks country/degree/activity binning, both SCORE+ implementations and community influence. Removing popular artists changes the similarity's denominator and the outcome itself, so an increased ratio is not by itself evidence for a stronger absolute effect or for social causation.''')
code(45, c[45].source.replace("'ratio': full['ratio'], 'z': r['z'], 'p': r['p']", "'ratio': full['ratio'], 'z': r['z'], 'p': r['p'], 'B': r['B'], 'exceedances': r['exceedances'], 'null_mean': r['null_mean'], 'excess_T': r['excess_T']").replace("for K in [5, 8, 11, 15, 20]:", "for K in sorted(set([5, 8, K_STAR, 15, 20])):").replace("add('K', K, MAIN_METHOD, scoreplus_labels(vals_L, vecs_L, K))", "add('K', K, MAIN_METHOD, scoreplus_labels(vals_L, vecs_L, K))\n    add('K', K, PAPER_METHOD, scoreplus_labels(vals_L, vecs_L, K, truncate=False))").replace("add('tau', name, MAIN_METHOD, scoreplus_labels(v, V, K_STAR))", "add('tau', name, MAIN_METHOD, scoreplus_labels(v, V, K_STAR))\n    add('tau', name, PAPER_METHOD, scoreplus_labels(v, V, K_STAR, truncate=False))"))
code(46, c[46].source.replace("for m in METHOD_ORDER:\n    add('pair-weighted T', '-', m, partitions[m], stat=T_pairs)", "# Pair-weighted tests already reuse all B_MAIN main permutations in tab16; no redundant rerun.").replace("savetab(rob_tab.round(5)", "savetab(rob_tab"))
# Different truncation variants share tau names; plot only the working main variant here.
code(47, c[47].source.replace("dt = rob_tab[rob_tab.check == 'tau']", "dt = rob_tab[(rob_tab.check == 'tau') & (rob_tab.method == MAIN_METHOD)]").replace("for a in axs: a.set_ylim(bottom=0.9)", "# Axis limits are left data-driven; negative or small effects must remain visible."))
md(48, '''**Interpretation rule.** Inspect both absolute and relative contrasts. Removing common artists changes the scale of Jaccard similarity. K and regularization choices can alter the partition itself. Do not call the result robust merely because a one-sided test rejects at one setting; inspect sign, magnitude, exchangeability support and sensitivity to weighting.''')
md(49, '''## 8. Network illustrations
These are descriptive illustrations of the **truncated SCORE+ variant**. The fixed layout is reused only when it matches the exact ordered node IDs and graph signature. Layout positions do not enter estimation, tuning, or inference. The community graph uses group size as area, observed edge count as width, and within-group Jaccard mean as color; edge widths are not adjusted for group sizes.''')
code(50, c[50].source.replace("if os.path.exists(lay_path):", "layout_signature = hashlib.sha256(nodes.tobytes() + A.indices.tobytes() + A.indptr.tobytes()).hexdigest()\nlayout_meta_path = os.path.join(OUT_DIR, 'layout_metadata.json')\nlayout_ok = os.path.exists(lay_path) and os.path.exists(layout_meta_path) and json.loads(Path(layout_meta_path).read_text()).get('signature') == layout_signature\nif layout_ok:").replace("pos = np.array(gi.layout_fruchterman_reingold(niter=1500).coords)", "ig.set_random_number_generator(random.Random(SEED))\n    initial = np.random.default_rng(SEED).normal(size=(n, 2)).tolist()\n    pos = np.array(gi.layout_fruchterman_reingold(niter=1500, seed=initial).coords)").replace("np.save(lay_path, pos)", "np.save(lay_path, pos)\n    Path(layout_meta_path).write_text(json.dumps({'signature': layout_signature, 'seed': SEED, 'node_count': n}, indent=2))\nassert pos.shape == (n, 2) and np.isfinite(pos).all()").replace('colored by SCORE+ community', 'colored by truncated SCORE+ community').replace("title='SCORE+ community'", "title='Truncated SCORE+ community'"))
code(51, c[51].source.replace('SCORE+ communities: size', 'Truncated SCORE+ communities: size'))
md(52, r'''
## Appendix A. Recovery under calibrated Poissonized DCBM and SBM generators

The real labels are unknown. We construct three **model-conditional** experiments: degree heterogeneity with truncated SCORE+ planted labels; degree heterogeneity with Louvain planted labels; homogeneous propensities within each block with truncated SCORE+ planted labels.

For block pair $k<l$, draw a Poisson number of edges with mean equal to the observed cross-block count and sample endpoints with probabilities $w_i=d_i/\sum_{j:c_j=k}d_j$. Within a block use its observed undirected edge count. The independent Poisson multiplicity for $i<j$ has mean $\lambda_{ij}=m_{kl}w_iw_j$ across blocks and $2m_{kk}w_iw_j$ within blocks. After merging multiple edges, $P(A_{ij}=1)=1-e^{-\lambda_{ij}}$. Thus this is a **Poissonized degree-corrected generator**, not exactly the linear Bernoulli DCBM above. In the homogeneous setting, expected degree may still differ **between** blocks. Removing self-loops, merging duplicates, and selecting the LCC change expected degrees; retained sample size and recovery group counts are recorded.

Every method sees the same graph within a replicate. Spectral methods receive the planted number of groups (an oracle advantage); Louvain uses the real-data resolution and selects its own group count. NMI and ARI are label-invariant recovery metrics. Report means, SD, Monte Carlo standard errors and t intervals over independent graph replicates, plus paired differences. These intervals concern mean performance under this generator, not uncertainty about real communities. Calibrating to two fitted partitions reduces but does not remove model favoritism.
''')
sim_code = c[53].source.replace("'SCORE': score_labels(V_a, K),", "'SCORE': score_labels(V_a, K), PAPER_METHOD: scoreplus_labels(v_l, V_l, K, truncate=False),")
sim_code = sim_code.replace("r = np.random.default_rng(SEED + 99)\nfor sname, truth, w in settings:", "for setting_index, (sname, truth, w) in enumerate(settings):")
sim_code = sim_code.replace("Am, cc = sample_dcbm(truth, w, r)", "r = np.random.default_rng(SEED + 99 + 1000 * setting_index + rep)\n        Am, cc = sample_dcbm(truth, w, r)")
sim_code = sim_code.replace("'n_lcc': len(cc)", "'n_lcc': len(cc), 'retained_fraction': len(cc) / n, 'estimated_K': len(np.unique(l)), 'planted_K': len(np.unique(truth)), 'planted_K_in_lcc': len(np.unique(tr)), 'm_lcc': Am.nnz // 2")
sim_code = sim_code[:sim_code.index("sim_tab = sim.groupby")]
sim_code += r'''
sim_summary_rows = []
for (setting, method), group in sim.groupby(['setting', 'method'], sort=False):
    row = {'setting': setting, 'method': method, 'replications': len(group),
           'mean_retained_fraction': group.retained_fraction.mean(), 'mean_estimated_K': group.estimated_K.mean()}
    for metric in ['NMI', 'ARI']:
        values = group[metric]
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        mcse = sd / np.sqrt(len(values))
        half = student_t.ppf(.975, len(values)-1) * mcse if len(values) > 1 else np.nan
        row.update({metric + '_mean': values.mean(), metric + '_sd': sd, metric + '_mcse': mcse,
                    metric + '_mean_ci_low': values.mean() - half, metric + '_mean_ci_high': values.mean() + half})
    sim_summary_rows.append(row)
sim_tab = pd.DataFrame(sim_summary_rows)
savetab(sim_tab, 'tabA2_simulation_summary')
paired_rows = []
for setting, group in sim.groupby('setting', sort=False):
    pivot = group.pivot(index='rep', columns='method', values='NMI')
    for other in METHOD_ORDER:
        if other == MAIN_METHOD:
            continue
        difference = pivot[MAIN_METHOD] - pivot[other]
        mcse = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else np.nan
        half = student_t.ppf(.975, len(difference)-1) * mcse if len(difference) > 1 else np.nan
        paired_rows.append(dict(setting=setting, comparison=MAIN_METHOD + ' minus ' + other,
            mean_NMI_difference=difference.mean(), mcse=mcse, ci_low=difference.mean()-half,
            ci_high=difference.mean()+half, replications=len(difference)))
savetab(pd.DataFrame(paired_rows), 'tabA3_simulation_paired_differences')
sim_tab.round(4)
'''
code(53, sim_code)
code(54, c[54].source.replace('plt.subplots(1, 3, figsize=(15, 3.8)', 'plt.subplots(1, 3, figsize=(18, 4.5)').replace("a.set_title(sname, fontsize=10)", "a.set_title(sname, fontsize=9)\n    a.tick_params(axis='x', labelrotation=35)").replace("axs[0].set_ylabel('NMI with planted communities')", "axs[0].set_ylabel('NMI with planted communities')\nfig.suptitle(f'Recovery across {N_SIM} independent simulated graphs per setting (points); mean (line)', fontsize=11)"))
md(55, '''**Interpretation rule.** Use paired recovery differences and their Monte Carlo uncertainty to state which methods work under which generators. Do not turn these calibrated experiments into a universal method ranking. In particular, the homogeneous and heterogeneous generators alter multiple graph properties after simplification and LCC selection, so their difference supports a mechanism without uniquely proving the cause of the real-data failure.''')
code(57, c[57].source.replace("'scipy.sparse.linalg.eigsh, top 30 eigenpairs by |eigenvalue|'", "'scipy.sparse.linalg.eigsh, deterministic v0, tol=1e-11; positive entrywise Perron refinement; top 30 by magnitude'").replace("ratios truncated at ±log n (Jin 2015)", "SCORE: truncated at +/-log n; main SCORE+ variant: truncated; paper comparator: untruncated").replace("τ = d̄ = {deg.mean():.2f} (Lec9 slide 28); published default 0.1·d_max = {0.1*deg.max():.1f} re-introduces localization (tab02), reported as robustness", "tau=mean degree={deg.mean():.2f}, an explicit modification; published 0.1*dmax={0.1*deg.max():.1f} tested for both SCORE+ variants").replace("t = 0.1 (Jin, Ke & Luo 2021 default)", "t=0.1; extra-vector gap uses eigenvalue magnitudes").replace("Jaccard index of liked-artist sets", "Jaccard of binary liked-artist sets, float64, zero diagonal and empty-union convention").replace("5×5 quantile bins of degree × #artists (main); country (confounder check)", "tie-preserving quantile bins: degree/activity q5; country only; primary joint country/degree/activity q3; q5/q7 sensitivity").replace("{B_MAIN} (main and country tests), {B_ROBUST} (robustness)", "{B_MAIN} for each main reference, {B_ROBUST} for sensitivity; both estimands share permutations").replace("DCBM/SBM with parameters estimated from the real data", "Poissonized DCBM and within-block homogeneous SBM, calibrated to real block edge counts"))
md(58, '''### Findings are generated from this run
The evidence summary at the end prints current values. Earlier hardcoded statements about 36% lift, causal country shares, universal recovery rankings, or every setting being significant have been removed. Inspect the primary joint-strata table and all sensitivity results before writing report claims.''')
md(59, '''### Main narrative to assess after execution
1. What sparse-network structure motivates degree correction and regularization?
2. Does a graph-only partition align with music preferences under the stated conditional permutation reference?
3. How much does the answer depend on weighting, country adjustment, implementation choices and the simulated network model?

The most informative result may be disagreement between methods or estimands. Such disagreement is evidence to interpret, not an output to optimize away.''')
md(60, '''### Limits that must carry into the report
- Observational association cannot distinguish selection, peer influence, or shared unmeasured factors.
- Within-stratum joint exchangeability is assumed, not established by preserving dyads in a permutation. Binning only approximately adjusts for continuous covariates.
- Inference conditions on one observed graph and fitted partition; it does not include uncertainty about a population network or true labels. Perturbation stability is a sensitivity diagnostic, not a population confidence interval.
- Likes and country labels are anonymized; the artist release is filtered and omits much of the long tail. A larger Jaccard ratio after artist removal is not a count-of-artists effect or causal evidence.
- Complete-case/LCC selection can bias the target population. Data-source counts alone do not establish byte-for-byte identity with SNAP; local file hashes establish reproducibility of this copy.
- The main method is a truncated SCORE+ modification. The untruncated paper comparator and published regularization must remain visible.
- Network models assume hard communities and conditional edge independence; real graphs may have overlapping memberships and transitivity beyond the model. Calibrated recovery experiments do not verify model fit.
- Analysis choices were revised after viewing earlier results. Sensitivity analyses are exploratory; Holm adjustment addresses the stated primary 10-test family, not all analytical choices.''')

add_md(r'''
## Appendix B. Diagnostics, calibration, and evidence for the report
The following outputs support the grading rubric, especially methodological correctness (25%), results (25%), insight (15%), and reproducibility (5%). They prepare report ingredients without drafting the report.

### B.1 Conditional randomization support and finer bins
Small strata may barely change labels. Record occupied cell sizes, within-cell covariate ranges, and the expected changed-label fraction $1-\sum_k(n_{sk}/n_s)^2$. This does not prove exchangeability, but detects a reference with little randomization support. Larger $q$ is not automatically better.
''')
add_code(r'''
diagnostic_frames, scheme_summaries, bin_sensitivity = [], [], []
for scheme, strata in [('degree_activity_q5', STRATA_DA), ('country', STRATA_C), ('country_degree_activity_q3', STRATA_CDA),
                        ('country_degree_activity_q5', stats.combine_strata(STRATA_C, make_strata(deg, n_art, q=5))),
                        ('country_degree_activity_q7', stats.combine_strata(STRATA_C, make_strata(deg, n_art, q=7)))]:
    for method in METHOD_ORDER:
        frame = stats.strata_diagnostics(strata, partitions[method], degree=deg, artist_count=n_art)
        frame.insert(0, 'method', method); frame.insert(0, 'scheme', scheme)
        diagnostic_frames.append(frame)
        scheme_summaries.append(dict(scheme=scheme, method=method, occupied_strata=len(frame),
            min_size=frame['size'].min(), median_size=frame['size'].median(),
            fraction_users_in_cells_below_10=frame.loc[frame['size'] < 10, 'size'].sum() / n,
            expected_fraction_labels_changed=np.average(frame.expected_changed_fraction, weights=frame['size'])))
    if scheme.endswith(('q5', 'q7')) and scheme.startswith('country_'):
        r = perm_test(S, partitions[MAIN_METHOD], strata, B_ROBUST, seed=SEED + 51)
        for statistic, result in [('equal', r), ('pair', r['pair_result'])]:
            bin_sensitivity.append(inference_row(MAIN_METHOD, result, scheme, statistic))
savetab(pd.concat(diagnostic_frames, ignore_index=True), 'tab17_strata_details')
savetab(pd.DataFrame(scheme_summaries), 'tab18_strata_support')
savetab(pd.DataFrame(bin_sensitivity), 'tab19_joint_binning_sensitivity')
display(pd.DataFrame(scheme_summaries).query('method == @MAIN_METHOD').round(4))
pd.DataFrame(bin_sensitivity).round(5)
''')
add_md('''### B.2 Truncation and community influence
Directly quantify the stabilization actually used and compare graph diagnostics across implementations. Leave-one-community-out contrasts remove that group and all its dyads, then recalculate both estimands from the remaining block sums. These are influence diagnostics, not independent replications or confidence limits.''')
add_code(r'''
embedding_rows = []
for name, values, vectors in [('tau_mean_degree', vals_L, vecs_L), ('tau_0.1_max_degree', *leading_eigs(reg_laplacian(A, .1 * deg.max()), 30))]:
    M = scoreplus_M(values, K_STAR)
    ratios = score_ratios(vectors[:, :M] * values[:M], M, truncate=False)
    for truncated in [False, True]:
        labels = scoreplus_labels(values, vectors, K_STAR, truncate=truncated)
        details = T_stat(S, labels, True)
        embedding_rows.append(dict(regularization=name, truncation=truncated, M=M,
            fraction_weighted_ratios_above_log_n=np.mean(np.abs(ratios) > np.log(n)),
            maximum_abs_raw_ratio=np.abs(ratios).max(), smallest_community=np.bincount(labels).min(),
            largest_fraction=np.bincount(labels).max()/n, T=details['T'], ratio=details['ratio'], T_pair=details['T_pair']))
savetab(pd.DataFrame(embedding_rows), 'tab20_scoreplus_implementation')
z = onehot(partitions[MAIN_METHOD]); w = z.T @ (S @ z); sizes = z.sum(0)
influence_rows = []
for omitted in range(len(sizes)):
    keep = np.arange(len(sizes)) != omitted
    result = stats.block_statistics(w[np.ix_(keep, keep)], sizes[keep])
    influence_rows.append(dict(omitted_community=omitted, omitted_users=int(sizes[omitted]), T=result['T'],
        ratio=result['ratio'], T_pair=result['T_pair'], change_from_full=result['T']-res_sp['T']))
savetab(pd.DataFrame(influence_rows), 'tab21_community_influence')
display(pd.DataFrame(embedding_rows).round(5))
pd.DataFrame(influence_rows).round(5)
''')
add_md(r'''
### B.3 Constructed-null calibration with country confounding
Hold a synthetic music matrix fixed: two countries favor different artist pools. Four equal communities are nested in country, but within each country community assignment is uniformly randomized. This defines an exact conditional no-residual-association null. Across independent outer assignments, compare unrestricted shuffling (an intentionally false null because it ignores country) with country-restricted shuffling. Report rejection proportions at $\alpha=.05$ and binomial Monte Carlo intervals. This checks the procedure on a known null; it does not validate exchangeability in LastFM.
''')
add_code(r'''
cal_rng = np.random.default_rng(SEED + 700)
cal_n, cal_p = 160, 60
cal_country = np.repeat([0, 1], cal_n // 2)
probabilities = np.where(np.arange(cal_p)[None, :] // 30 == cal_country[:, None], .65, .08)
cal_x = sp.csr_matrix((cal_rng.random((cal_n, cal_p)) < probabilities).astype(float))
cal_s = jaccard_matrix(cal_x)
base_labels = np.repeat(np.arange(4), cal_n // 4)
CAL_R, CAL_B = (200, 199) if PROFILE == 'full' else (5, 9)
cal_records = []
for rep in range(CAL_R):
    labels = base_labels.copy()
    for group in [np.flatnonzero(cal_country == k) for k in [0, 1]]:
        labels[group] = labels[cal_rng.permutation(group)]
    for scheme, strata in [('unrestricted_ignores_country', np.zeros(cal_n, dtype=int)), ('country_restricted_correct_null', cal_country)]:
        result = perm_test(cal_s, labels, strata, CAL_B, seed=SEED + 10000 + rep)
        cal_records.append(dict(rep=rep, scheme=scheme, p=result['p'], reject=result['p'] <= .05))
calibration = pd.DataFrame(cal_records)
savetab(calibration, 'tabB1_calibration_raw')
cal_summary = []
for scheme, group in calibration.groupby('scheme', sort=False):
    count = int(group.reject.sum())
    low, high = stats.binomial_interval(count, len(group))
    cal_summary.append(dict(scheme=scheme, repetitions=len(group), inner_permutations=CAL_B, alpha=.05,
        rejection_count=count, rejection_fraction=count/len(group), mc_ci_low=low, mc_ci_high=high))
savetab(pd.DataFrame(cal_summary), 'tabB2_calibration_summary')
pd.DataFrame(cal_summary)
''')
add_md('''### B.4 Report ingredients and reproducibility
The assignment recommends about eight main-text pages (12pt, single spacing), with code and extensive technical tables in supplementary material. The next cells map each rubric category to evidence, save exact configuration and file hashes, and generate current findings. They do not draft the report or guarantee a grade.''')
add_code(r'''
rubric_rows = [
    (10, 'Problem formulation and motivation', 'Question, target population, data appropriateness', 'Notebook opening and section 1; tab01, tab14', 'Define social graph versus external music outcome; qualify selected sample.'),
    (25, 'Methodology and technical understanding', 'Model, assumptions, algorithms, tuning, implementation', 'Sections 2-5; tab02-04, tab13, tab15, tab17-20', 'Explain DCBM geometry, sparse noise, ratio stabilization, graph-only tuning, estimands and invariant null.'),
    (25, 'Results and analysis', 'Complete comparisons, experiments, simulations, valid conclusions', 'tab06, tab08, tab16, tab19-21, tabA1-A3, tabB1-B2; fig03, fig06, fig08, figA1', 'Report excess contrast, both weighting schemes, corrected p, MC precision, conditional recovery and exceptions.'),
    (15, 'Intellectual contribution and insight', 'Interpretation beyond mechanical implementation', 'Localization/truncation comparison, weighting reversal, joint-null comparison, paired simulations', 'Explain when degree correction is insufficient and why effect size is not recovery accuracy.'),
    (10, 'Organization and clarity of the report', 'Logical exposition and understandable statistics', 'Figure/table selection saved below; technical details in notebook', 'Main narrative: question -> model -> diagnostic -> conditional evidence -> sensitivity -> limitations.'),
    (5, 'Reproducibility and supporting materials', 'Data sources, software, choices, code', 'requirements-lock.txt, lastfm_stats.py, scripts/run_notebook.py, tests; outputs/run_manifest.json, permutation_nulls.npz', 'Provide Run All/full command, seeds, hashes, actual versions, raw nulls and raw simulations.'),
    (10, 'Webpage demonstration', 'Clear summary and communication', 'fig06, fig09, fig10; evidence JSON', 'Reuse verified findings later; webpage is outside this code-preparation task.'),
]
rubric = pd.DataFrame(rubric_rows, columns=['weight_percent', 'criterion', 'evidence_needed', 'available_material', 'writing_instruction'])
assert rubric.weight_percent.sum() == 100
savetab(rubric, 'tab00_report_rubric_evidence')

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

manifest = dict(run_started_utc=RUN_STARTED, run_finished_utc=datetime.now(timezone.utc).isoformat(), profile=PROFILE,
    configuration=dict(seed=SEED, B_MAIN=B_MAIN, B_ROBUST=B_ROBUST, N_SUB=N_SUB, N_SIM=N_SIM, N_INIT=N_INIT,
        K_candidates=list(K_RANGE), K_selected=K_STAR, tau=float(TAU), gamma=float(GAMMA), main_method=MAIN_METHOD,
        primary_reference='country_degree_activity_q3', primary_family_size=10, alternative='greater', alpha=.05,
        dtype='float64', analysis_status='exploratory reanalysis after prior inspection',
        python_threads=4, scoreplus_gap='1 - abs(lambda_Kplus1)/abs(lambda_K) <= 0.1',
        permutation_unit='entire user music records (implemented via inverse label permutations)'),
    data=dict(source=SRC, source_url='https://snap.stanford.edu/data/feather-lastfm-social.html',
        distribution_url='https://github.com/benedekrozemberczki/karateclub/tree/master/dataset',
        collection_period='March 2020 (SNAP description)', n=int(n), m=int(G.number_of_edges()), artist_count=int(X.shape[1])),
    versions={name:importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'networkx', 'scikit-learn', 'matplotlib', 'igraph', 'nbformat', 'ipython', 'threadpoolctl']},
    input_sha256={str(p):file_hash(p) for p in sorted(Path(DATA_DIR).glob('*')) if p.is_file()},
    source_sha256={str(p):file_hash(p) for p in [Path('lastfm_stats.py'), Path('scripts/run_notebook.py')]},
    notebook_cell_source_sha256=hashlib.sha256(''.join(cell.source for cell in __import__('nbformat').read('STAT236_LastFM_Community_Music.ipynb', as_version=4).cells).encode()).hexdigest(),
    outputs=dict(figures=sorted(p.name for p in Path(FIG_DIR).glob('*') if p.is_file()),
        tables=sorted(p.name for p in Path(TAB_DIR).glob('*.csv'))))
Path(OUT_DIR, 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
primary = inference_tab[inference_tab.primary_family]
main_primary = primary[primary.Method.eq(MAIN_METHOD)]
evidence = dict(main_method=MAIN_METHOD, n=int(n), m=int(G.number_of_edges()), K=int(K_STAR),
    observed_jaccard=T_stat(S, partitions[MAIN_METHOD], True)['ratio'],
    primary_inference=main_primary.to_dict(orient='records'),
    main_text_figures=['fig02_localization', 'fig03_K_selection', 'fig06_permutation_nulls', 'figA1_simulation'],
    main_text_tables=['tab01_data_summary', 'tab16_inference_all_references', 'tabA2_simulation_summary'],
    required_qualifications=['Exploratory, not preregistered.', 'Conditional exchangeability within coarse bins is an assumption.',
        'No causal decomposition of country and social effects.', 'MC intervals and permutation ranges are not population effect CIs.',
        'Truncated SCORE+ is a modified implementation.', 'Recovery rankings are conditional on the specified generators.'])
Path(OUT_DIR, 'report_evidence.json').write_text(json.dumps(evidence, indent=2))
display(rubric)
''')
add_code(r"""
equal = main_primary[main_primary.statistic.eq('equal')].iloc[0]
pair = main_primary[main_primary.statistic.eq('pair')].iloc[0]
display(Markdown(f'''
**Current run: {PROFILE}.** {n:,} users, {G.number_of_edges():,} edges; graph-selected K={K_STAR}.

**Working main method: {MAIN_METHOD}.** Within/between mean Jaccard ratio = {equal['ratio S_same/S_diff']:.3f};
equal-community contrast T={equal['T']:.5f}. Under joint country/degree/activity randomization,
excess T={equal['excess_T']:.5f}, p={equal['p']:.5g}, Holm p (10-test family)={equal['p_Holm_within_reference_10_tests']:.5g}.
Exceedances: {int(equal['exceedances'])}/{int(equal['B'])}; 95% Monte Carlo tail interval
[{equal['mc_tail_ci_low']:.5g}, {equal['mc_tail_ci_high']:.5g}].

**Pair-weighted sensitivity:** T={pair['T']:.5f}, joint excess={pair['excess_T']:.5f},
Holm p={pair['p_Holm_within_reference_10_tests']:.5g}.

These are conditional associations under a stated exchangeability null, not causal effects or population confidence intervals.
Full method/reference/estimand results: `tab16_inference_all_references.csv`.
Report preparation evidence and exact configuration: `outputs/report_evidence.json` and `outputs/run_manifest.json`.
'''))
""")

nb.metadata['stat236_analysis_version'] = 2
nbformat.validate(nb)
nbformat.write(nb, path)
print('Updated', path, 'with', len(nb.cells), 'cells')
