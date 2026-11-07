# Do Social Circles Share Music Taste?

**Comparing four community detection methods on LastFM Asia**

STAT 236 · Fall 2026 · Faye Fang and Grace Chen

[Analysis notebook](STAT236_LastFM_Community_Music.ipynb) · [Interactive walkthrough](same_circle_same_playlist.html) · [Figures](output/figures/) · [Result tables](output/tables/)

We build social circles from mutual-follow relationships, then use liked-artist lists to ask whether those circles share music preferences. Spectral clustering, SCORE, SCORE+, and Louvain provide four different ways to group the same users.

![Four methods applied to the same LastFM Asia network](output/figures/network_comparison.png)

*The same network, 11 circles per method, and different partitions. Colors identify circles separately within each method.*

**7,431 users · 27,045 social ties · 7,842 artists · 18 country labels**

## Two goals

1. **Answer the research question:** are music preferences more similar within social circles than between circles?
2. **Compare the methods:** how do degree adjustment and grouping objectives change circle sizes, music contrasts, and their geographic interpretation?

Only the network is used for fitting and tuning. Music preferences and country labels are reserved for assessment.

## Main findings

- **Shared taste appears under all four methods when circles receive equal weight.** Within-circle Jaccard similarity is 25% to 54% higher than between-circle similarity.
- **Weighting changes the interpretation.** SCORE, SCORE+, and Louvain remain positive when user pairs receive equal weight. Spectral changes from `T_eq = 0.0468` to `T_pair = -0.0137`; its largest circle supplies 99.92% of within-circle pairs.
- **Country is an important comparison.** Within-country shuffling reproduces 37% to 72% of the observed circle-weighted contrast. A positive excess remains for all four methods after jointly matching country, degree, and artist-list length.

All four primary Holm-adjusted p-values are **0.002**, the minimum attainable here with 1,999 shuffles and four tests. Equal p-values do not imply equal effect sizes. Country shuffle shares describe a comparison baseline, not a causal contribution.

The analysis is exploratory. Music association and agreement with country describe the resulting circles; they do not establish grouping accuracy without known true memberships. Full values are in [the main results](output/tables/main_test.csv), [country checks](output/tables/country_check.csv), and [partition agreement](output/tables/nmi_matrix.csv).

## Start here

- **Read the analysis:** open [the notebook](STAT236_LastFM_Community_Music.ipynb), which includes saved outputs.
- **Explore visually:** download [the interactive walkthrough](same_circle_same_playlist.html) and open it locally in a browser.
- **Inspect the evidence:** browse [figures](output/figures/) and [CSV tables](output/tables/), or use [the saved circle assignments](output/partitions.csv).

### Reproduce the results

The saved results use **Python 3.12.3**. Package versions are pinned in [requirements.txt](requirements.txt).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/jupyter notebook STAT236_LastFM_Community_Music.ipynb
```

Run the cells in order, or choose **Run All**. A full run writes figures and tables to `output/`. The default seed is 236; Louvain resolution tuning uses seeds 236 through 240.

<details>
<summary><strong>Command-line execution and quick checks</strong></summary>

To execute the notebook and refresh its saved outputs:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  STAT236_LastFM_Community_Music.ipynb
```

For a quick code check, set `STAT236_QUICK=1` before execution. Quick mode uses fewer permutations and writes to `tmp/quick_output/`. The reported analysis uses normal mode with the full permutation counts.

</details>

## Methods and evaluation

<details>
<summary><strong>Four methods, one network, a common circle count</strong></summary>

- **Spectral clustering:** k-means on the leading eigenvectors of the adjacency matrix, without degree adjustment.
- **SCORE:** k-means on eigenvector ratios, with clipping at `±log(n)`.
- **SCORE+:** ratios from a regularized matrix, with eigenvalue weights and an extra eigenvector for weak signals. We use mean-degree regularization, `τ = 7.28`, `t = 0.1`, and 12 eigenvectors. Added clipping uses the same `±log(n)` bound as SCORE.
- **Louvain:** modularity optimization at resolution `γ = 0.25`.

The main comparison uses **11 circles**. We select this working scale from the largest relative eigengap over candidates 3 through 20. Louvain's resolution is selected by the median circle count across five seeds. These choices use network information only.

Clipping SCORE+ is an implementation choice in this project. [Alternative versions](output/tables/scoreplus_versions.csv) report its behavior without clipping and with the published default regularization of `0.1 × maximum degree`.

</details>

<details>
<summary><strong>Music similarity, weighting, and permutation tests</strong></summary>

Music similarity is Jaccard similarity: the number of shared artists divided by the number liked by at least one user.

- **`T_eq`** averages within-circle means equally and subtracts the equally weighted mean between circle pairs. This is the primary contrast.
- **`T_pair`** pools within-circle pairs and between-circle pairs separately, with equal weight for user pairs in each mean.
- **Excess** is the observed contrast minus its permutation mean.

We keep the circles fixed and shuffle whole artist lists, preserving the dependence among user pairs.

- **Main analysis:** 1,999 shuffles within degree and artist-list-length groups, formed using up to five quantile bins per variable; Holm correction covers the four primary tests.
- **Country comparisons:** 199 shuffles within country alone, then within country crossed with three quantile groups each of degree and list length.
- **Supplementary checks:** 199 shuffles for alternative settings and artist-removal checks.

The tests assume whole lists are exchangeable within their shuffle groups. The bins provide approximate matching, and the findings describe association rather than causation.

</details>

## Repository guide

```text
.
├── STAT236_LastFM_Community_Music.ipynb   # analysis and saved outputs
├── same_circle_same_playlist.html        # interactive walkthrough
├── requirements.txt                     # pinned dependencies
├── data/                                # network, artists, country labels
└── output/
    ├── figures/                         # PDF and PNG figures
    ├── tables/                          # CSV results
    ├── partitions.csv                   # circle labels for each method
    ├── layout.npy                       # saved network coordinates
    └── layout_metadata.json             # layout compatibility checks
```

## Data and references

The data come from the authors' karateclub repository and match the [SNAP LastFM Asia description](https://snap.stanford.edu/data/feather-lastfm-social.html): 7,624 users and 27,806 mutual-follow edges. We remove 173 users without music records, then retain the largest connected component. Artist identities and country codes are anonymized. Likes record preferences rather than listening frequency.

<details>
<summary><strong>Dataset and methodological references</strong></summary>

- Rozemberczki and Sarkar (2020), [Characteristic Functions on Graphs](https://doi.org/10.1145/3340531.3411866).
- Jin (2015), [Fast Community Detection by SCORE](https://arxiv.org/abs/1211.5803).
- Jin, Ke and Luo (2022), [Improvements on SCORE, Especially for Weak Signals](https://arxiv.org/abs/1811.05927).
- Karrer and Newman (2011), [Stochastic Blockmodels and Community Structure in Networks](https://arxiv.org/abs/1008.3926).
- Blondel et al. (2008), [Fast Unfolding of Communities in Large Networks](https://arxiv.org/abs/0803.0476).
- Phipson and Smyth (2010), [Permutation P-values Should Never Be Zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).

</details>
