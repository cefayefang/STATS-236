from pathlib import Path
import nbformat
p=Path('code/STAT236_LastFM_Community_Music.ipynb')
n=nbformat.read(p,4)
assert not any('Appendix D.' in c.source for c in n.cells)
md=nbformat.v4.new_markdown_cell
cc=nbformat.v4.new_code_cell
report_intro=n.cells.pop(71)
report_intro.source=report_intro.source.replace('### B.5 Report ingredients and reproducibility','## Appendix E. Report ingredients and reproducibility')
index=next(i for i,c in enumerate(n.cells) if c.source.startswith('rubric_rows ='))
add=[md('''## Appendix D. Targeted validation of the original three-method comparison
These additions address inference calibration, nuisance adjustment, recovery fairness, and algorithmic stability. They were specified after the earlier analysis was inspected and remain exploratory. They preserve the original primary reference and hypothesis family; no reference is selected by its p-value.

### D.1 Primary-reference sensitivity
Compare prespecified degree/activity quantile bin counts 3, 5, 7, and 10, plus exact degree crossed with activity quintiles. The original q5 draws are reused. Report reference-centered contrasts, Monte Carlo uncertainty, randomization support, and residual covariate association. Exact degree still only approximately adjusts for activity. Residual association alone neither proves nor disproves music exchangeability.'''),
cc('''import statistical_validation as validation
from dataclasses import asdict
validation_config = validation.ValidationConfig.for_profile(PROFILE)
bin_validation, bin_support = validation.primary_binning_sensitivity(
    S, deg, n_art, partitions, TAB_DIR, OUT_DIR, validation_config, main_res)
display(bin_validation[['scheme','method','statistic','T','excess_T','p','p_Holm_3_methods_within_scheme_statistic']])
display(bin_support)'''),
md('''### D.2 Calibration of the actual primary procedure
Hold one independently generated network and its graph-only partitions fixed, then draw 200 independent music datasets per scenario. Each dataset uses 199 inner permutations, all three methods, and both weighting schemes. The invariant binned null has degree-bin-specific music preferences but no within-bin community effect; both references are invariant. A continuous-degree-only stress scenario has smooth degree-dependent music preferences and no additional community term: exact-degree permutations are invariant, whereas coarse q5 bins need not be. A planted community/music association is a positive control for power.

Use exact binomial Monte Carlo intervals, and a simultaneous lower bound over the 18 invariant-null cases to flag clear excess size. Also report family rejection after Holm correction over the three equal-community tests. These experiments condition on one generated graph and cannot establish exchangeability in LastFM. Stress rejection under approximate bins is not exact-null type I error.'''),
cc('''primary_calibration, primary_calibration_family, calibration_design = validation.calibrate_primary_test(
    TAB_DIR, OUT_DIR, validation_config)
display(primary_calibration)
display(primary_calibration_family)'''),
md('''### D.3 Independent planted networks with matched K information
Use an independently specified balanced Bernoulli DCBM with three groups, 600 vertices, expected full-graph mean degree 18, degree heterogeneity sd in {0, 1.2}, and within/between intensity ratio in {3, 8}. Degree weights have block mean one and bounds [0.35, 2.5]. Edge probabilities are validated and never clipped. Thirty independent replications share truth and uniform edge draws across regimes within a replication. All four regimes use the same iteratively selected connected evaluation population; no graph is redrawn. Record retention to expose conditioning on this population.

Spectral and SCORE receive the planted K. Louvain receives the same K information through a prespecified resolution grid, choosing the first exact cluster-count match, otherwise the closest count. It never receives planted memberships. Fixed-resolution Louvain remains a separately marked reference. Report paired recovery differences and the paired heterogeneity interaction (SCORE minus Spectral in heterogeneous minus homogeneous networks). Intervals quantify Monte Carlo precision under this design, not population-network uncertainty or universal rankings.'''),
cc('''independent_summary, independent_paired, degree_interaction = validation.independent_recovery(
    TAB_DIR, validation_config)
display(independent_summary)
display(independent_paired)
display(degree_interaction)'''),
md('''### D.4 Equal K information on the original calibrated generators
Reproduce the original Poissonized graphs exactly and verify retained sample sizes, edge counts, and original-method recovery against Appendix A. Add K-assisted Louvain on those identical graphs, reporting actual K-match rates. A finite grid need not attain K in every graph. Keep fixed-resolution Louvain separately visible. Generator calibration to fitted partitions and component selection remain limitations even after information is made comparable.'''),
cc('''matched_legacy_summary, matched_legacy_paired = validation.legacy_matched_recovery(
    A, partitions, deg, TAB_DIR, validation_config, resolution=GAMMA)
display(matched_legacy_summary)
display(matched_legacy_paired)'''),
md('''### D.5 Shared graph perturbations and random seeds
Use 20 common edge-retention masks (independent retention probability 0.9) for Spectral, SCORE, and Louvain. In every replicate, all methods fit and evaluate the same largest connected component, with original K/resolution and the same fitting seed. Compare partitions by NMI/ARI and recompute both music contrasts. Each contrast change subtracts the original partition's contrast restricted to these same retained users, separating population removal from reassignment. This is graph-perturbation sensitivity, not a resampling confidence interval for a population effect.

Separately vary 10 fitting seeds on the original graph. Report seed-to-original NMI, music contrasts, and pairwise seed similarity. Pairwise similarities are dependent and receive no iid-pair confidence interval.'''),
cc('''core_perturbations, core_perturbation_summary, core_seed_runs, core_seed_summary = validation.core_stability(
    A, S, G, nodes, partitions, K_STAR, GAMMA, TAB_DIR, OUT_DIR, validation_config)
display(core_perturbation_summary)
display(core_seed_summary)'''),
md('''### D.6 Validation figures and provenance
Five figures show primary calibration, nuisance-stratification sensitivity, independent recovery, calibrated-generator K fairness, and core-method stability. Keep these diagnostics and their limitations alongside the main research findings when choosing report claims.'''),
cc('''from validation_visualizations import generate_validation_figures, write_validation_manifest
validation_figures = generate_validation_figures(RUN_ROOT, show=True)
plt.show()
validation_manifest = write_validation_manifest(RUN_ROOT, validation_config, calibration_design, PROFILE)
print('Five-page validation review: figures/statistical_validation_review.pdf')
print('Validation interpretation warnings:')
for warning in validation_manifest['warnings']:
    print('-', warning)
if not validation_manifest['warnings']:
    print('No automatic diagnostic flag; observational exchangeability and model adequacy remain assumptions.')'''),report_intro]
n.cells[index:index]=add
meta=next(c for c in n.cells if c.source.startswith('rubric_rows ='))
meta.source=meta.source.replace('tab17-20', 'tab17-20, tab23-36; Appendix D').replace('tabB1-B2\',', 'tabB1-B2; tab23-37 and fig14-18\',')
meta.source=meta.source.replace("primary_reference=PRIMARY_REFERENCE, primary_family_size=3", "statistical_validation=asdict(validation_config), primary_reference=PRIMARY_REFERENCE, primary_family_size=3")
meta.source=meta.source.replace("Path('code/visualizations.py'),", "Path('code/visualizations.py'), Path('code/statistical_validation.py'), Path('code/validation_visualizations.py'), Path('code/scripts/run_validation.py'),")
meta.source=meta.source.replace("calibration_summary=cal_summary,", "calibration_summary=cal_summary,\n    primary_calibration_summary=json.loads(primary_calibration.to_json(orient='records')),\n    primary_calibration_family=json.loads(primary_calibration_family.to_json(orient='records')),\n    independent_recovery_summary=json.loads(independent_summary.to_json(orient='records')),\n    degree_correction_interaction=json.loads(degree_interaction.to_json(orient='records')),\n    matched_legacy_summary=json.loads(matched_legacy_summary.to_json(orient='records')),\n    core_perturbation_summary=json.loads(core_perturbation_summary.to_json(orient='records')),\n    core_seed_summary=json.loads(core_seed_summary.to_json(orient='records')),\n    validation_warnings=validation_manifest['warnings'],\n    validation_review='figures/statistical_validation_review.pdf',\n    validation_manifest='outputs/statistical_validation_manifest.json',")
last=n.cells[-1]
last.source+='''\ndisplay(Markdown("**Targeted validation:** primary calibration and stratification sensitivity, matched-information recovery, and shared core-method perturbations are in Appendix D. Review these before choosing report claims."))
for warning in validation_manifest['warnings']:
    display(Markdown('**Qualification:** '+warning))
'''
n.cells[0].source=n.cells[0].source.replace('Appendix B contains supplementary country checks, diagnostics and report evidence.', 'Appendix B contains supplementary country checks and diagnostics; Appendix C contains core-method visualizations; Appendix D contains primary calibration, matched-information recovery and shared perturbations; Appendix E collects report evidence.')
nbformat.validate(n)
nbformat.write(n,p)
print('Integrated',len(add)-1,'validation cells;',sum(c.cell_type=='code' for c in n.cells),'total code cells')
