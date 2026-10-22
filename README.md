# STAT 236 Project: Is a Social Circle Also a Music Circle?

Community detection on the LastFM Asia social network (Faye Fang, Grace Chen).

## Contents

| Path | What it is |
|---|---|
| `STAT236_LastFM_Community_Music.ipynb` | Main notebook (executed, with outputs and interpretations) |
| `requirements.txt` | Python packages |
| `data/` | LastFM Asia data (edges, liked artists, country) |
| `figures/` | All figures as PNG (200 dpi), ready for the report and webpage |
| `tables/` | All result tables as CSV |
| `outputs/partitions.csv` | Community label of every user for each method |
| `outputs/layout.npy` | Cached network layout (delete to recompute) |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook STAT236_LastFM_Community_Music.ipynb   # Run All; about 20 min on a laptop
```

For a quick run, lower `B_MAIN`, `B_ROBUST` and `N_SIM` in the setup cell.

## Data

The data are the LastFM Asia network (Rozemberczki & Sarkar, 2020) from https://snap.stanford.edu/data/feather-lastfm-social.html. The files in `data/` come from the authors' `karateclub` repository (same network: 7,624 users and 27,806 edges). The notebook also reads the SNAP files directly (`lastfm_asia_edges.csv`, `lastfm_asia_features.json`, `lastfm_asia_target.csv`) if you put them in `data/`.

## Deviations from the proposal

The research question, data, similarity measure, test statistic and permutation test are as proposed. The changes below are all additions or justified adjustments; nothing in the proposal was dropped. Explain them briefly in the report's Methodology section.

| Proposal | What we did | Why |
|---|---|---|
| Three methods: Spectral, SCORE (main), Louvain | Four methods: **SCORE+ is the main method**; Spectral, SCORE and Louvain are kept | The leading eigenvectors of A are localized on a few hubs (top 50 users hold ~80% of the mass; ~39% of SCORE ratios hit the ±log n truncation), so SCORE's degree correction breaks down on this network. SCORE+ (Lec9 slides 35–43) adds pre-PCA Laplacian regularization and fixes this (notebook §2.1, fig02, tab02). |
| (not specified) | SCORE+ regularization τ = mean degree d̄ (Lec9 slide 28) instead of the published default 0.1·d_max | 0.1·d_max = 21.4 is ~3× the mean degree here (d_max/d̄ = 29, vs. 3–13 in the Lec9 benchmark networks) and partly re-introduces localization. Both are reported (robustness, fig08c). |
| Choose K from eigengap, sizes and stability | Same criteria, applied to the main method's matrix L_τ; **K = 11** used for all spectral methods | Common K keeps the method comparison fair; K ∈ {5, 8, 11, 15, 20} reported as robustness. |
| Louvain resolution tuned so K is close to the spectral methods | γ = 0.25 (K = 11); default γ = 1 (K = 28) also reported | As proposed. |
| Stratified permutation test (degree × #artists) | As proposed, plus a **country-stratified** test | Country could drive both friendships and taste. It explains about half of the effect (notebook §5). |
| Test statistic T (community-averaged) | As proposed, plus a **pair-weighted T** as a robustness check | The community-averaged T is inflated by Spectral's many tiny hub clusters; under pair weighting Spectral's effect turns negative. |
| Optional DCBM simulation | Done (Appendix A), with three settings: DCBM with SCORE+ truth, DCBM with Louvain truth, and SBM without degree heterogeneity | Shows that Spectral fails because of degree heterogeneity and that SCORE+ is reliable. |
| Data from SNAP | Same LastFM Asia dataset obtained from the authors' `karateclub` GitHub repository | The SNAP download was blocked in our environment. Node and edge counts match SNAP exactly, and the loader also reads the SNAP files. |

## Figure guide

| Figure | Content | Suggested use |
|---|---|---|
| fig01 | Degree distribution, artists per user | Report: Data |
| fig02 | Eigenvector localization (A vs. regularized Laplacian) | Report: Methods (why SCORE+) |
| fig03 | Choosing K (scree, eigengap, stability) | Report: Methods |
| fig04 | Community sizes per method | Report: Results |
| fig05 | Similarity matrix between SCORE+ communities | Report: Results / Appendix |
| fig06 | Observed T vs. permutation nulls (headline) | Report + webpage |
| fig07 | NMI between methods; hub placement | Report: Method comparison |
| fig08 | Robustness (K, popular artists, τ) | Report: Robustness |
| fig09 | Network colored by community | Webpage |
| fig10 | Community graph | Webpage |
| figA1 | DCBM simulation | Appendix |
