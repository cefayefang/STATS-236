# Is a Social Circle Also a Music Circle?

STAT 236, Fall 2026. Faye Fang and Grace Chen.

If we find social circles using only who follows whom, are they also music circles? We study this question on the LastFM Asia network with 7,431 users and four community detection methods: Spectral clustering, SCORE, SCORE+ and Louvain.

The network is used to form communities. Music and country are used only to test them.

- **[Analysis notebook](STAT236_LastFM_Community_Music.ipynb):** all code, with saved outputs.
- **[Interactive webpage](same_circle_same_playlist.html):** download the file and open it in a browser to explore the results.
- **[Figures](output/figures/) and [tables](output/tables/):** everything the notebook creates.

## Questions

1. What communities can we find from the follow network alone?
2. Are users in the same community more similar in music taste than users in different communities?
3. Is this only because people from the same country like the same music?
4. Does the answer change across methods?

## Main results

| Method | Largest community | T_eq | Within / between ratio | Holm p | T_pair | Reproduced by within-country shuffling | Excess after controlling country, degree and activity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spectral | 91.9% | 0.0468 | 1.54 | 0.002 | −0.0137 | 37% | 0.0191 |
| SCORE | 18.7% | 0.0154 | 1.25 | 0.002 | 0.0163 | 65% | 0.0047 |
| SCORE+ | 25.5% | 0.0212 | 1.36 | 0.002 | 0.0178 | 52% | 0.0094 |
| Louvain | 23.8% | 0.0198 | 1.33 | 0.002 | 0.0177 | 72% | 0.0036 |

- All four methods show higher music similarity within communities than between them.
- Country does not explain all of the pattern. A positive excess remains after we control for country, degree and activity.
- The method changes how we read the result. Spectral has the largest T_eq, but its T_pair is negative, because one group holds 99.92% of its within-community pairs.

These are associations, not causal effects. The analysis is exploratory, because some choices were made after looking at the data. We do not rank the methods, because the data have no true community labels.

## Files

```
STAT236_LastFM_Community_Music.ipynb   all analysis code, with saved outputs
same_circle_same_playlist.html         interactive webpage
requirements.txt                       exact package versions
data/                                  LastFM Asia data files
output/figures/                        all figures (PDF and PNG)
output/tables/                         all result tables (CSV)
output/partitions.csv                  community label of each user, for each method
output/layout.npy                      saved network positions, used only for plotting
output/layout_metadata.json            checks that the saved positions match the network
```

## Run the analysis

The saved results use Python 3.12.3.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 STAT236_LastFM_Community_Music.ipynb
```

You can also open the notebook in Jupyter and choose **Run All**. All random steps use seed 236.

To test that the code runs, set `STAT236_QUICK=1` before running. This uses very few permutations and writes to `tmp/quick_output/`. Do not use those results.

## Methods and settings

| Method | What it does | Settings |
| --- | --- | --- |
| Spectral | k-means on the first K eigenvectors of A | K = 11 |
| SCORE | k-means on eigenvector ratios, clipped to ±log n | K = 11 |
| SCORE+ | The same idea on the regularized matrix L_τ, with eigenvalue weights and one extra eigenvector when the signal looks weak | K = 11, τ = mean degree = 7.28, t = 0.1, M = 12, clipped to ±log n |
| Louvain | Modularity optimization | γ = 0.25 |

All settings use the network only.

- **K** is the largest relative eigengap of L_τ for K from 3 to 20. All methods use the same K so the comparison is fair.
- **γ** is the value whose median community count over five seeds is closest to K.
- **Clipping SCORE+** is our change to the published algorithm. SCORE uses the same limit. The notebook also reports SCORE+ without clipping and with τ = 0.1 × max degree.

## Testing

For each method, T is the mean Jaccard similarity within communities minus the mean between communities.

- `T_eq` gives each community equal weight. This is the main statistic.
- `T_pair` gives each pair of users equal weight.

User pairs are not independent, so we use permutation tests. Communities stay fixed and whole music records are shuffled among users.

| Test | Shuffle among | Permutations |
| --- | --- | --- |
| Main test | Users in the same degree × activity bin (5 bins each) | 1,999 |
| Country check | Users in the same country | 199 |
| Joint check | Users in the same country × degree × activity bin (3 bins each) | 199 |
| Sensitivity checks | Users in the same degree × activity bin | 199 |

The Holm correction covers the four methods in the main test. The tests assume that music records can be swapped within a bin, and the bins only roughly control degree and activity.

## Data

The files come from the authors' karateclub repository. The raw counts match the [SNAP LastFM Asia description](https://snap.stanford.edu/data/feather-lastfm-social.html): 7,624 users and 27,806 mutual-follow edges. We remove 173 users with no music data and keep the largest connected component. The final sample has 7,431 users, 27,045 edges, 7,842 artists and 18 countries. Artists and countries are anonymous.

## References

- Rozemberczki and Sarkar (2020), [Characteristic functions on graphs](https://doi.org/10.1145/3340531.3411866). Source of the data.
- Jin (2015), [Fast community detection by SCORE](https://arxiv.org/abs/1211.5803).
- Jin, Ke and Luo (2022), [Improvements on SCORE, especially for weak signals](https://arxiv.org/abs/1811.05927).
- Karrer and Newman (2011), [Stochastic blockmodels and community structure in networks](https://arxiv.org/abs/1008.3926).
- Blondel et al. (2008), [Fast unfolding of communities in large networks](https://arxiv.org/abs/0803.0476).
- Phipson and Smyth (2010), [Permutation p-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
