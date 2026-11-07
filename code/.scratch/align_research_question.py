from pathlib import Path
import nbformat

path = Path('code/STAT236_LastFM_Community_Music.ipynb')
nb = nbformat.read(path, as_version=4)
cells = nb.cells
for cell in cells:
    cell.source = cell.source.replace('MAIN_METHOD', 'DETAIL_METHOD')
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

cells[0].source = '''# Is a Social Circle Also a Music Circle?
## Community Detection in the LastFM Asia Social Network
**STAT 236, Fall 2026 · Faye Fang, Grace Chen**

**Project type:** Application of existing community detection methods to real data, comparing how method choice affects the conclusions.
**Timeline 1:** Topic due October 5; report and webpage due November 8.

**Main research question:** Do users within the same communities, detected from the social network, also share their music taste?

The follow network and music taste are **two separate sources of information**:
1. **Community structure:** Using only the follow network, what communities do we find?
2. **Social circle = music circle?** Are users inside a community more similar in music taste than users in different communities?
3. **Does the method matter?** Compare Spectral, SCORE, and Louvain. Spectral versus SCORE examines degree correction; spectral methods versus Louvain examines agreement across different approaches.

The network supplies adjacency matrix A and the estimated community labels. Music supplies artist sets and Jaccard similarities; it is used only for external evaluation, never for community fitting or tuning. At the pair level the comparison concerns same/different detected-community membership and music similarity. Degree and artist counts are derived nuisance summaries, not additional research outcomes. **Country is an optional supplementary confounding check, not a primary research variable or community label.**

**Method extensions:** SCORE+ is an explicitly justified extension to the original comparison. Both its paper embedding and a truncated stabilization are retained. Detailed community profiles and illustrations use the truncated variant, without replacing the original three-method research question.

**Analysis status:** Exploratory reanalysis of an already inspected dataset, not preregistered confirmation. Network fitting and tuning use only the graph. Main inference uses the degree/activity permutation reference. Pair weighting, SCORE+ implementations, and country references are supporting analyses with clearly identified roles.

**Notebook route:** §1 data; §2 network models and algorithms; §3 music similarity; §4 main community/music test; §5 original-method comparisons; §6 sensitivity; §7 network illustrations; §8 reproducibility and findings. Appendix A contains recovery simulations; Appendix B contains supplementary country checks, diagnostics and report evidence.

**Course emphasis:** sparse-network signal and noise, SBM/DCBM, degree heterogeneity, eigenvector localization, regularization, dependent dyads, exchangeability and simulation-based evaluation. Earlier course-slide numbers have not been independently verified.
'''

setup = cells[2].source
setup = setup.replace('from pathlib import Path', '''from pathlib import Path
import sys
START_DIR = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [START_DIR, *START_DIR.parents]
                     if (p / 'data').is_dir() and (p / 'code/lastfm_stats.py').is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError('Open this notebook within the project directory or code/ folder.')
sys.path.insert(0, str(PROJECT_ROOT / 'code'))''')
setup = setup.replace("DATA_DIR = 'data'", "DATA_DIR = str(PROJECT_ROOT / 'data')")
setup = setup.replace("RUN_ROOT = Path('tmp/smoke') if PROFILE == 'smoke' else Path('.')", "RUN_ROOT = PROJECT_ROOT / 'tmp/smoke' if PROFILE == 'smoke' else PROJECT_ROOT")
setup = setup.replace('DETAIL_METHOD = "SCORE+ (truncated)"', '''B_COUNTRY = B_ROBUST  # supplementary country references; primary test retains B_MAIN
CORE_METHODS = ['Spectral', 'SCORE', 'Louvain']
PRIMARY_REFERENCE = 'degree_activity_q5'
DETAIL_METHOD = "SCORE+ (truncated)"''')
cells[2].source = setup

cells[9].source = cells[9].source.replace('## 2. Network model and community detection', '## 2. Network model and community detection: original comparison and extensions')
cells[9].source = cells[9].source.replace('and the working main method inherited from the original analysis', 'used for detailed illustrations as an extension of the original three-method comparison')
cells[9].source = cells[9].source.replace('**Algorithms (all fit without music or country):**', '**Core comparison:** Spectral, SCORE and Louvain answer the original method-choice question. SCORE+ implementations are additional comparisons motivated by the network diagnostics. **All algorithms fit without music or country:**')
cells[13].source = cells[13].source.replace('(c) Stability of SCORE+', '(c) Stability of truncated SCORE+')
cells[17].source = cells[17].source.replace('(c) Stability of SCORE+', '(c) Stability of truncated SCORE+')
cells[25].source = '''**Interpretation.** Begin with the original comparison: does correcting degree heterogeneity (Spectral versus SCORE) alter the partition, and does Louvain agree with the spectral approaches? SCORE+ rows document the extension. Size imbalance, hub placement and planted-label recovery help interpret these differences; no real-data ground-truth labels are available.'''
cells[29].source = cells[29].source.replace('## 4. Degree/activity reference: the statistical question and estimands', '## 4. Main question: is a social circle also a music circle?')
cells[31].source = cells[31].source.replace("savetab(main_tab, 'tab06_main_test', index=True)", '''main_tab['analysis_role'] = ['core_method' if m in CORE_METHODS else 'SCORE+_extension' for m in main_tab.index]
main_tab['p_Holm_primary_3_methods'] = np.nan
main_tab.loc[CORE_METHODS, 'p_Holm_primary_3_methods'] = stats.holm_adjust(main_tab.loc[CORE_METHODS, 'p (perm)'])
savetab(main_tab, 'tab06_main_test', index=True)''')
cells[32].source = '''**Inference hierarchy.** The main analysis tests music similarity in graph-only communities using degree/activity stratification, following the two-source research design. The primary family contains the equal-community statistic for **Spectral, SCORE and Louvain** (three tests; Holm adjusted). Pair weighting diagnoses sensitivity to community size. Both SCORE+ versions are extensions, and country references belong in Appendix B as supplementary confounding checks. These choices follow an exploratory, previously inspected analysis and are not preregistered.'''
cells[33].source = '### 4.1 Detailed community profiles: truncated SCORE+ extension\nThis illustrates one fitted partition; it does not replace the three-method comparison.'
cells[34].source = cells[34].source.replace('Mean Jaccard similarity between SCORE+ communities', 'Mean Jaccard similarity: truncated SCORE+ communities')

main_plot = nbformat.v4.new_code_cell('''# Main figure: the original community detection comparison and independent music evaluation
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
x = np.arange(len(CORE_METHODS))
same = main_tab.loc[CORE_METHODS, 'S_same'].to_numpy()
different = main_tab.loc[CORE_METHODS, 'S_diff'].to_numpy()
axs[0].bar(x - .18, same, width=.36, color=METHOD_COLOR[DETAIL_METHOD], label='Within detected communities')
axs[0].bar(x + .18, different, width=.36, color=OTHER, label='Between detected communities')
axs[0].set_xticks(x, CORE_METHODS)
axs[0].set(xlabel='Community detection method (network only)', ylabel='Mean Jaccard similarity (music only)',
           title='(a) Do social communities also share music taste?')
axs[0].legend(fontsize=8)
for j, method in enumerate(CORE_METHODS):
    result = main_res[method]
    axs[1].plot([result['null_q025'], result['null_q975']], [j, j], color=OTHER, lw=6, solid_capstyle='butt')
    axs[1].scatter(result['null_mean'], j, marker='|', s=100, color=INK2)
    axs[1].scatter(result['T'], j, color=METHOD_COLOR[method], s=45, zorder=3)
axs[1].axvline(0, color=GRID, lw=1)
axs[1].set_yticks(range(len(CORE_METHODS)), CORE_METHODS)
axs[1].set(xlabel='Equal-community Jaccard contrast T', ylabel='Community detection method',
           title='(b) Observed contrast and degree/activity null')
fig.text(.5, -.01, f'LastFM Asia; B={B_MAIN}. Points: observed T; grey bars: central 95% permutation ranges, not effect confidence intervals.', ha='center', fontsize=8)
fig.tight_layout(); savefig(fig, 'fig06_permutation_nulls'); plt.show()''')

cells[36].source = '''### B.1 Country as a supplementary confounding check
Country is not one of the two primary sources used to pose the research question. These optional checks ask whether geography could partly reproduce the observed association. They do not redefine communities, change graph fitting, or replace the main degree/activity test.

Country-only and country × degree × activity (three-bin) permutations are **supplementary**; finer joint bins are additional sensitivity checks. Their Monte Carlo counts and randomization support are reported. Small or unmixed strata can limit the information supplied by a reference.

The country-only null mean divided by observed T is descriptive, not a causal fraction explained. Rejecting a country-conditioned reference does not identify selection, influence or all unmeasured confounding.'''
country_code = cells[37].source.replace('STRATA_C, B_MAIN', 'STRATA_C, B_COUNTRY').replace('STRATA_CDA, B_MAIN', 'STRATA_CDA, B_COUNTRY')
country_code = country_code.replace("inference_tab['primary_family'] = inference_tab.strata.eq('country_degree_activity_q3')", '''inference_tab['primary_family'] = (inference_tab.strata.eq(PRIMARY_REFERENCE)
    & inference_tab.Method.isin(CORE_METHODS) & inference_tab.statistic.eq('equal'))
inference_tab['analysis_role'] = np.select([
    inference_tab.primary_family,
    inference_tab.strata.str.startswith('country'),
    ~inference_tab.Method.isin(CORE_METHODS),
], ['primary_original_method', 'supplementary_country', 'SCORE+_extension'], default='pair_weighting_sensitivity')
inference_tab['p_Holm_primary_3_methods'] = np.nan
mask = inference_tab.primary_family
inference_tab.loc[mask, 'p_Holm_primary_3_methods'] = stats.holm_adjust(inference_tab.loc[mask, 'p'])''')
country_code = country_code.replace("display(inference_tab[inference_tab.primary_family][['Method', 'statistic', 'T', 'excess_T', 'p', 'p_Holm_within_reference_10_tests', 'mc_tail_ci_high']].round(6))", "display(country_tab.round(5))")
cells[37].source = country_code
country_plot = cells[38].source.replace('plt.subplots(1, 3, figsize=(16, 4.6)', 'plt.subplots(1, 2, figsize=(12.5, 4.6)')
country_plot = country_plot.replace("['Degree x activity', 'Country', 'Country x degree x activity'], [main_res, country_res, country_res2]", "['Country only (supplementary)', 'Country x degree x activity (supplementary)'], [country_res, country_res2]")
country_plot = country_plot.replace('Observed contrast (points) and central 95% permutation reference (grey bars)', 'Supplementary country checks: observed contrast and permutation references')
country_plot = country_plot.replace('B={B_MAIN}', 'B={B_COUNTRY}').replace("'fig06_permutation_nulls'", "'figB1_country_sensitivity'")
cells[38].source = country_plot
cells[39].source = '''**Supplementary interpretation.** Report country checks briefly in sensitivity analyses or the appendix. They address an alternative explanation for community/music association, not a separate main research question. Do not convert null-mean differences into causal contributions. Primary conclusions must remain anchored in the original three-method community/music comparison.'''
cells[40].source = '## 5. Does the method choice change the answer?\nSpectral versus SCORE examines degree correction. Spectral methods versus Louvain examine agreement across different approaches. SCORE+ is a documented extension.'
compare = cells[41].source.replace(", 'Country': country", '')
begin = compare.index("summary_tab =")
compare = compare[:begin] + '''summary_tab = main_tab[['analysis_role', 'K', 'S_same', 'S_diff', 'T', 'ratio S_same/S_diff', 'p (perm)', 'p_Holm_primary_3_methods']].copy()
summary_tab['pair_weighted_T'] = [main_res[m]['pair_result']['T'] for m in summary_tab.index]
summary_tab['NMI with SCORE'] = [NMI(partitions[m], partitions['SCORE']) for m in summary_tab.index]
savetab(summary_tab, 'tab11_method_comparison', index=True)
display(summary_tab.loc[CORE_METHODS].round(5))
display(hub_tab.round(3))'''
cells[41].source = compare
plot = cells[42].source
plot = "core_nmi = nmi_mat.loc[CORE_METHODS, CORE_METHODS]\n" + plot
plot = plot.replace('nmi_mat.values', 'core_nmi.values').replace('len(names)', 'len(CORE_METHODS)').replace('set_xticklabels(names,', 'set_xticklabels(CORE_METHODS,').replace('set_yticklabels(names)', 'set_yticklabels(CORE_METHODS)')
plot = plot.replace('for m in METHOD_ORDER:', 'for m in CORE_METHODS:').replace('(a) NMI between partitions', '(a) Agreement among the original three methods')
cells[42].source = plot
cells[43].source = '''**Main interpretation.** Compare community sizes, hub placement, music contrasts and weighting sensitivity for Spectral, SCORE and Louvain. These directly answer whether degree correction or a different optimization approach changes the conclusions. Agreement is not accuracy: music coherence is external evidence and NMI measures agreement; only planted-label simulations provide a recovery metric. Keep SCORE+ findings clearly labeled as extensions.'''
cells[44].source = cells[44].source.replace('## 7.', '## 6.')
cells[49].source = cells[49].source.replace('## 8.', '## 7.')
cells[56].source = '## 8. Reproducibility and findings'
cells[57].source = cells[57].source.replace('primary joint country/degree/activity q3', 'supplementary joint country/degree/activity q3').replace("{B_MAIN} for each main reference, {B_ROBUST} for sensitivity; both estimands share permutations", "{B_MAIN} for the degree/activity main test; {B_COUNTRY} for country checks; {B_ROBUST} for other sensitivities; paired estimands share draws")
cells[58].source = '### Findings follow the original research question\nThe final summary compares Spectral, SCORE and Louvain on network-derived communities and independently measured music similarity. SCORE+ provides a documented method extension. Country checks are supplementary.'
cells[59].source = '''### Report narrative
1. What communities are detected from the follow network alone?
2. Is music taste more similar within those communities than between them?
3. Does the conclusion change when correcting degree heterogeneity (Spectral versus SCORE) or changing method family (spectral methods versus Louvain)?

Explain SCORE+ separately as a motivated extension. Country belongs in a short confounding sensitivity discussion or appendix. Interpret association, estimator weighting and method disagreement without causal claims.'''
cells[60].source = cells[60].source.replace('The main method is a truncated SCORE+ modification.', 'The detailed community illustration uses a truncated SCORE+ modification.').replace('the stated primary 10-test family', 'the stated primary three-method family')
cells[63].source = '## Appendix B. Supplementary country checks, diagnostics, and report evidence\nThese materials support methodological assessment and robustness without replacing the two-source research question.'
strata_heading = nbformat.v4.new_markdown_cell('### B.2 Randomization support and finer country bins\nOccupied cell sizes, covariate ranges and the expected changed-label fraction diagnose whether the supplementary reference has useful mixing. These are checks on randomization support, not proof of exchangeability.')
cells[65].source = cells[65].source.replace('B.2', 'B.3')
cells[67].source = cells[67].source.replace('B.3', 'B.4')
cells[69].source = cells[69].source.replace('B.4', 'B.5')
meta = cells[70].source
meta = meta.replace("'Sections 2-5;", "'Sections 2-4;")
meta = meta.replace('tab06, tab08, tab16, tab19-21, tabA1-A3, tabB1-B2; fig03, fig06, fig08, figA1', 'Main: tab05, tab06, tab11 and fig04, fig06, fig07; extensions/sensitivity: tab08, tab16, tab19-21, tabA1-A3, tabB1-B2')
meta = meta.replace('Localization/truncation comparison, weighting reversal, joint-null comparison, paired simulations', 'Spectral versus SCORE degree correction, comparison with Louvain, weighting reversal; SCORE+ and country as supporting analyses')
meta = meta.replace('question -> model -> diagnostic -> conditional evidence -> sensitivity -> limitations', 'follow-network communities -> independent music validation -> method-choice comparison -> extensions and sensitivity -> limitations')
meta = meta.replace('requirements-lock.txt, lastfm_stats.py, scripts/run_notebook.py, tests;', 'code/requirements-lock.txt, code/lastfm_stats.py, code/scripts/run_notebook.py, code/tests;')
meta = meta.replace('main_method=DETAIL_METHOD,', "core_methods=CORE_METHODS, illustration_method=DETAIL_METHOD, B_COUNTRY=B_COUNTRY,")
meta = meta.replace("primary_reference='country_degree_activity_q3', primary_family_size=10", "primary_reference=PRIMARY_REFERENCE, primary_family_size=3, primary_statistic='equal'")
meta = meta.replace('input_sha256={str(p):', 'input_sha256={str(p.relative_to(PROJECT_ROOT)):')
meta = meta.replace("[Path('lastfm_stats.py'), Path('scripts/run_notebook.py'), Path('requirements.txt'), Path('requirements-lock.txt')]", "[Path('code/lastfm_stats.py'), Path('code/scripts/run_notebook.py'), Path('code/requirements.txt'), Path('code/requirements-lock.txt')]")
meta = meta.replace('str(p):file_hash(p) for p in [Path', 'str(p):file_hash(PROJECT_ROOT / p) for p in [Path')
meta = meta.replace(".read('STAT236_LastFM_Community_Music.ipynb',", ".read(PROJECT_ROOT / 'code/STAT236_LastFM_Community_Music.ipynb',")
meta = meta.replace("main_primary = primary[primary.Method.eq(DETAIL_METHOD)]", "extension_results = inference_tab[inference_tab.strata.eq(PRIMARY_REFERENCE) & ~inference_tab.Method.isin(CORE_METHODS)]")
meta = meta.replace('evidence = dict(main_method=DETAIL_METHOD,', 'evidence = dict(core_methods=CORE_METHODS, illustration_method=DETAIL_METHOD,')
meta = meta.replace("observed_jaccard=T_stat(S, partitions[DETAIL_METHOD], True)['ratio'],", "research_question='Are users inside communities detected from the follow network more similar in music taste than users in different communities?',\n    separate_information_sources=['follow network for community detection', 'music taste for external evaluation'],\n    country_role='supplementary confounding check',")
meta = meta.replace("primary_inference=main_primary.to_dict(orient='records'),", "primary_inference=primary.to_dict(orient='records'),\n    method_extensions=extension_results.to_dict(orient='records'),")
meta = meta.replace("'fig02_localization', 'fig03_K_selection', 'fig06_permutation_nulls', 'figA1_simulation'", "'fig03_K_selection', 'fig04_community_sizes', 'fig06_permutation_nulls', 'fig07_method_comparison'")
meta = meta.replace("'tab01_data_summary', 'tab16_inference_all_references', 'tabA2_simulation_summary'", "'tab01_data_summary', 'tab06_main_test', 'tab11_method_comparison'")
cells[70].source = meta
cells[71].source = '''display(Markdown(f"""**Current run: {PROFILE}.** {n:,} users, {G.number_of_edges():,} edges; graph-selected K={K_STAR}.

**Research question:** communities are estimated from the follow network alone, then tested using independent music taste. The primary comparison is Spectral, SCORE and Louvain; country is supplementary.
"""))
for method in CORE_METHODS:
    result = primary[primary.Method.eq(method)].iloc[0]
    pair = main_res[method]['pair_result']
    display(Markdown(f"""**{method}:** within/between mean Jaccard ratio={result['ratio S_same/S_diff']:.3f};
equal-community T={result['T']:.5f}, degree/activity-null excess={result['excess_T']:.5f};
p={result['p']:.5g}, Holm p (original three-method family)={result['p_Holm_primary_3_methods']:.5g}.
Pair-weighted sensitivity T={pair['T']:.5f}.
"""))
display(Markdown("""SCORE+ implementations are documented extensions. Country-only and joint-country references are supplementary checks in Appendix B.
These are associations under stated permutation assumptions, not causal effects.
Main comparison: `tab06_main_test.csv` and `tab11_method_comparison.csv`; all references: `tab16_inference_all_references.csv`.
"""))'''

# Place the country analysis physically in the appendix, not among main results.
ordered = []
for i in range(len(cells)):
    if i in [36, 37, 38, 39]:
        continue
    ordered.append(cells[i])
    if i == 32:
        ordered.append(main_plot)
    if i == 63:
        ordered.extend([cells[36], cells[37], cells[38], cells[39], strata_heading])
nb.cells = ordered
nb.metadata['stat236_analysis_version'] = 3
nbformat.validate(nb)
nbformat.write(nb, path)
print('Aligned original research questions; country moved to appendix;', len(nb.cells), 'cells')
