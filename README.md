# STAT 236: Is a Social Circle Also a Music Circle?

Faye Fang and Grace Chen · Fall 2026

**Project type:** Application of existing community detection methods to real data,
with a comparison of how method choice affects the conclusions.
**Timeline 1:** Topic due October 5; report and webpage due November 8.

**Main question:** Do users within the same communities, detected from a social
network, also share their music taste?

The follow network and music taste are **two separate sources of information**.
We use the network to find communities and music taste only to test them:

1. **Community structure:** Using only the follow network, what communities do we find?
2. **Social circle = music circle?** Are users inside a community more similar in
   music taste than users in different communities?
3. **Does the method matter?** Compare Spectral, SCORE and Louvain. Spectral versus
   SCORE examines correcting degree heterogeneity; spectral methods versus Louvain
   examine whether a different approach gives the same conclusion.

Adjacency A supplies graph-only community labels; artist incidence X supplies
Jaccard similarities for external evaluation. Music never enters detection or
parameter tuning. Degree and artist counts are derived nuisance summaries.
**Country is a supplementary confounding check, not a main research variable.**
SCORE+ is a documented methodological extension, with its untruncated and truncated
implementations evaluated separately from the original three-method comparison.

## Run

The audited environment uses Python 3.12.3. Install the exact environment with
`code/requirements-lock.txt`, or use `code/requirements.txt` for compatible package ranges.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements-lock.txt
.venv/bin/python -m unittest discover -s code/tests -v
.venv/bin/python code/scripts/run_notebook.py
.venv/bin/python code/scripts/verify_results.py
```

The runner executes every code cell and saves the executed notebook under `code/`,
tables, figures, raw permutations, simulations, metadata and per-cell timings.
No Jupyter server is needed. Alternatively, open the notebook in Jupyter from
the project directory or from `code/` and use **Run All**.

```bash
.venv/bin/python code/scripts/run_notebook.py --smoke
```

Smoke output is isolated under `tmp/smoke/`. It uses the real sample and graph
methods, but very few permutations, perturbations and simulations; its inference
and recovery summaries must not be used as report results. Network layout is
cached with the ordered node IDs and graph signature. Recomputing layout adds
time but does not affect statistical results.

## Organization

All project source code, the notebook, tests, runners and dependency files live
under `code/`. Data, generated results and project instructions remain at the root.

- `code/STAT236_LastFM_Community_Music.ipynb`: executable methods, analysis, findings,
  limitations, diagnostics and report preparation evidence.
- `code/lastfm_stats.py`: reusable embeddings, Jaccard, estimands, stratification,
  permutation inference, Monte Carlo intervals and Holm adjustment.
- `code/scripts/run_notebook.py`: in-process executor with rich outputs/checkpoints.
- `code/tests/test_statistics.py`: small independent reference problems for inference.
- `data/`: local LastFM files.
- `figures/`: PNG (200 dpi) and PDF figures for the report.
- `tables/`: data, inference, sensitivity, simulation and rubric tables.
- `outputs/partitions.csv`: labels indexed by original user ID.
- `outputs/permutation_nulls.npz`: three principal references, five methods and
  both estimands; keys identify each combination.
- `outputs/run_manifest.json`: configuration, versions, input/code hashes and
  output inventory.
- `outputs/execution_log.json`: cell success flags and measured run times.
- `outputs/report_evidence.json`: current primary results, implementation,
  simulation/calibration summaries and pointers for report writing.

The existing Chinese `.docx` intuition guide predates this audit and has not
been updated. Take numbers and qualifications from the executed notebook and
current evidence files.

## Report visualizations

All figure labels, annotations, captions and the figure guide are in English.
The six-page `figures/visualization_review.pdf` collects selected visuals for
report planning. Individual research figures are exported as 300 dpi PNG and PDF:

- `fig11_core_networks`: the same network and layout colored by Spectral, SCORE
  and Louvain; colors are matched to SCORE for display only.
- `fig12_core_music_heatmaps`: shared-scale within/between community music
  similarity for the original methods, with community sizes underneath.
- `fig13_weighting_sensitivity`: equal-community versus equal-user-pair contrasts,
  including the Spectral sign change and appropriate permutation ranges.

`tables/tab22_visualization_guide.csv` explains each selected figure's research
question, interpretation limits and suggested report placement. The existing
main test, method comparison and K-selection plots are included in the review.

```bash
.venv/bin/python code/visualizations.py
```

This command regenerates the new figures from saved partitions and current data,
checks layout alignment, and verifies plotted similarity statistics against the
executed analysis. It does not refit communities using music. The notebook also
generates and embeds these figures during Run All. Source and artifact hashes
are saved in `outputs/visualization_manifest.json`.

## Statistical design

We compare unnormalized spectral clustering, SCORE, untruncated SCORE+,
**SCORE+ with ratio truncation**, and Louvain. The original research comparison remains Spectral, SCORE and Louvain. The
truncated variant is used for detailed community profiles and network illustrations
as a methodological extension. Truncation is
explicitly identified as a stabilization, and the paper algorithm is retained
as a separate comparator. Mean-degree regularization and the published
`0.1 * maximum degree` setting are both assessed.

K is selected by the largest unrounded magnitude eigengap over 3 through 20.
Sizes and edge-deletion stability are diagnostics, not undisclosed selection
rules. The SCORE+ extra-vector rule separately uses the signed eigenvalue ratio
in the paper. Louvain resolution approximates the selected K using group counts
across fixed seeds. Eigenvectors use deterministic initial vectors and a positive,
entrywise-refined leading denominator. Graph fitting never sees music or country.

Jaccard and matrix calculations use float64. Equal-community and pair-weighted
contrasts are different estimands; both are calculated from each permutation
using one product. Quantile strata retain ties rather than splitting by node ID;
occupied cells, covariate ranges, sizes and label-mixing support are exported.

The primary reference stratifies by **degree x artist count**, using five
quantile bins per quantity without splitting ties (the actual occupied cell
count is reported). These nuisance summaries come from the same two data sources.
The main contrast compares within-community and between-community mean Jaccard
similarity. The primary Holm family is the equal-community statistic for
**Spectral, SCORE and Louvain** (three tests). Pair weighting diagnoses sensitivity
to community-size imbalance; SCORE+ results document an extension.

The main reference uses 1,999 permutations. Country-only and country/degree/activity
references are supplementary appendix checks and use 199 permutations, as do
other exploratory sensitivity checks. Neither country labels nor music taste
enter detection or tuning. Multiplicity adjustment does not account for every
exploratory analytical choice.

Tables give `(1 + exceedances) / (B + 1)`, observed contrasts, permutation means,
null-centered excesses, central null ranges, and binomial intervals for Monte
Carlo tail precision. These ranges/intervals are not population effect confidence
intervals. Permutation z values are descriptive and are not extrapolated to
Gaussian tail probabilities.

Recovery experiments use three calibrated Poissonized block-model settings,
20 graphs each, with methods paired on the same graph in each replicate. They
report NMI, ARI, retained sample sizes, fitted group counts, Monte Carlo SEs,
intervals for mean simulated performance, and paired differences. Simplified
graphs have edge probability `1 - exp(-lambda)`, not exactly a linear Bernoulli
DCBM. The homogeneous comparison removes within-block propensity variation;
between-block expected degrees may still differ. Spectral methods receive the
planted K; Louvain chooses its own K at the real-data resolution.

A constructed-null calibration has 200 outer random assignments and 199 inner
permutations. It compares country-restricted and unrestricted inference when
music and community composition share a country factor. This checks behavior
on a known null, not exchangeability in the actual data.

## Report preparation

The assignment gives 90% of the grade to the report and 10% to the webpage.
Report criteria are problem formulation (10%), methodology (25%), results (25%),
intellectual contribution (15%), clarity (10%) and reproducibility (5%).
`tables/tab00_report_rubric_evidence.csv` maps these criteria to available
materials. The guidelines recommend about eight main-text pages, 12pt and single
spacing, with code and extensive technical details in supporting material.

- Methods: notebook sections 2–4; `tab02`–`tab04`, `tab13`, `tab15`, `tab20`.
- Main comparison: `tab06_main_test.csv`, `tab11_method_comparison.csv` and
  `fig06_permutation_nulls`; original-method rows are marked `primary_family` in
  `tab16_inference_all_references.csv`.
- Supplementary country results: `tab08`, country rows in `tab16`, and
  `figB1_country_sensitivity`; these do not lead the main narrative.
- Sensitivity: `tab12`, `tab18`–`tab21`; `fig08_robustness` displays both absolute
  and relative contrasts after popular-artist removal.
- Recovery: `tabA1`–`tabA3` and `figA1_simulation`.
- Constructed-null calibration: `tabB1` and `tabB2`.
- Sample selection: `tab01_data_summary.csv`, `tab14_attrition_by_country.csv`.

Organize the main narrative around the three research questions above. Explain
SCORE+ as an extension and keep country checks in sensitivity analysis or the
appendix. This work prepares evidence; it does not draft the report or webpage.

## Qualifications that must carry into the report

This is an exploratory reanalysis after previous results were inspected, not a
preregistered confirmatory study. Within-stratum joint exchangeability of whole
music-record assignments is an assumption. Keeping overlapping dyads in the
permutation does not establish exchangeability; coarse bins only approximate
continuous-covariate adjustment. Inference conditions on one graph and fitted
partition, without population-network uncertainty. Perturbation stability is a
sensitivity diagnostic, not a population confidence interval.

LCC/complete-case selection, unmeasured geography/language/age, overlapping
memberships and edge dependence can matter. The country-null/observed-T ratio
is descriptive, not a causal fraction explained. Residual association cannot
distinguish selection from influence. Likes are not listening histories, and
anonymous artist IDs cannot identify genres.

A Jaccard ratio is not a ratio of artist counts. Removing popular artists changes
the denominator: a larger ratio can coexist with a smaller absolute contrast.
Higher music coherence does not imply more accurate community recovery.
Recovery rankings are conditional on the specified generators.

## Data and primary sources

The local files come from the dataset authors' karateclub distribution. Raw
counts match the [SNAP LastFM Asia description](https://snap.stanford.edu/data/feather-lastfm-social.html)
(7,624 users, 27,806 mutual-follow edges, March 2020 collection). Matching counts
do not establish byte-for-byte identity with another download. Local input hashes
identify the exact copy used. The loader also accepts original SNAP CSV/JSON names.

Primary references: [SCORE](https://arxiv.org/abs/1211.5803),
[SCORE+](https://arxiv.org/abs/1811.05927),
[degree-corrected block models](https://arxiv.org/abs/1008.3926), and
[permutation p-values](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
The notebook has the reference list. Course-slide numbers retained from the
earlier notebook have not been independently verified.

## Targeted statistical validation (Appendix D)

The original three-method question and primary test remain fixed. These
exploratory additions address the statistical audit rather than choosing an
analysis by its result. Source: `code/statistical_validation.py`; plots:
`code/validation_visualizations.py`; independent reference tests:
`code/tests/test_validation.py`. Full Run All includes every addition.

- **Primary-reference sensitivity:** degree/activity quantile bins 3, 5, 7, 10,
  and exact degree crossed with activity quintiles. The original q5 reference
  reuses its 1,999 draws; other references use 499 shared permutations for the
  three core methods and both estimands. Tables 23-24 preserve tail precision,
  residual covariate diagnostics and label-mixing support. Exact degree does not
  imply exact activity adjustment. Comparisons are exploratory.
- **Actual-procedure calibration:** one independently generated, fixed graph;
  200 independent music datasets in each of three scenarios; 199 inner
  permutations per dataset/reference. The scenarios separate an invariant
  binned null, continuous-degree-only stress, and a planted music positive
  control. Both references are invariant in the first scenario. Only exact
  degree is invariant under the continuous-degree stress design; q5 rejection
  there diagnoses approximate adjustment rather than exact-null type I error.
  Tables 25-27 report pointwise binomial Monte Carlo intervals, a simultaneous
  lower bound across the 18 invariant-null cases, and Holm family rejection.
  Conditional calibration does not establish LastFM exchangeability.
- **Independent recovery with matched information:** balanced Bernoulli DCBM,
  n=600, K=3, expected full-graph mean degree 18; degree-weight sd 0 or 1.2 and
  within/between intensity ratio 3 or 8; 30 paired replications per setting.
  Weights have block mean one and bounds [0.35, 2.5]. Probabilities are validated,
  never clipped. Replications share truth and edge uniforms across regimes and
  use an identical iteratively selected connected core across all four regimes.
  No graphs are redrawn. Tables 28-30 give retention, paired differences,
  pointwise Monte Carlo intervals, and the degree-correction interaction.
- **K fairness on calibrated generators:** reproduce the 20 original graphs per
  setting exactly, then give Louvain planted K information through a fixed
  resolution grid. Selection uses graph cluster counts only, never planted
  memberships or music. Spectral and SCORE receive the same K. Exact matches
  are not guaranteed; tables 31-32 retain actual match rates and search traces.
  Fixed-resolution Louvain is a separate reference in both recovery experiments.
- **Core-method stability:** 20 common 90%-edge-retention masks, the same retained
  users and fitting seed for all methods, and 10 seeds on the original graph.
  Tables 33-36 retain partition agreement and both music contrasts. Each
  perturbation contrast change subtracts the original contrast on the same
  retained users. Perturbation/seed ranges are sensitivity diagnostics, not
  population confidence intervals. Pairwise seed similarities are dependent.

`figures/statistical_validation_review.pdf` collects five English figures
(fig14-18); `tables/tab37_validation_figure_guide.csv` explains their roles and
limits. Raw draws and perturbation/seed partitions are saved in `outputs/`.
`outputs/statistical_validation_manifest.json` records configuration, source and
artifact hashes, design details and diagnostic qualifications. Scientific
outcomes are allowed to disagree with the initial conclusions; verification
checks correctness and provenance, not that null rejection must be below 5%.

To rerun only these additions from saved full-analysis partitions and results:

```bash
.venv/bin/python code/scripts/run_validation.py
```

Use full notebook execution for the final, internally synchronized report bundle.
The standalone command's `--smoke` writes to `tmp/validation_smoke/` and is not
report evidence. In a notebook smoke run all new replication counts are reduced
and the original recovery graphs are limited to those generated by that run.

### Current validation findings

On LastFM, exact degree crossed with activity quintiles retains positive
equal-community contrasts for Spectral, SCORE and Louvain (499 permutations;
Holm p=0.006 within this exploratory three-method family). Spectral's user-pair
contrast remains negative. This is robustness to these references, not proof
that all confounding has been removed.

On the constructed invariant binned null, rejection rates across methods,
references and weighting schemes range from 2% to 6.5% in 200 replications;
no case triggers the simultaneous excess-size flag. Under continuous-degree-only
stress, Spectral's equal-community rejection rate is 33% with coarse q5 bins
and 6.5% with exact degree. The primary p-values support a stated binned
reference comparison; they must not be described as exact continuous degree
and activity adjustment. Calibration on one graph cannot establish LastFM
exchangeability.

In the independent recovery design, the heterogeneous-minus-homogeneous
interaction in SCORE's ARI advantage over Spectral is 0.501 (pointwise Monte
Carlo 95% interval 0.485-0.517) at intensity ratio 3 and 0.809 (0.802-0.817)
at ratio 8. K-assisted Louvain attains K=3 in 86.7%-100% of runs across settings;
fixed-resolution Louvain returns a single community in this design. These
results illustrate why degree correction and K information matter under the
specified generator; they do not establish a universally best method.

Across the 20 common edge-deletion runs, mean partition NMI is 0.877 for
Spectral, 0.759 for SCORE and 0.882 for Louvain. All three retain positive
equal-community music contrasts; Spectral retains a negative user-pair contrast
in every run. Label changes and stability of the substantive music comparison
are distinct findings. Retained users range from 96.7% to 97.4%; contrast changes
always compare the same retained users. These ranges do not quantify
population-network uncertainty.

The completed full run executes 42 code cells in approximately 11 minutes and
embeds 20 figures. Fourteen implementation reference tests and 830 saved-result
integrity checks pass. Passing these checks establishes reproducibility and
internal consistency, not the substantive exchangeability assumption.
