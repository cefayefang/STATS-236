import nbformat

path = 'STAT236_LastFM_Community_Music.ipynb'
nb = nbformat.read(path, as_version=4)
nb.cells[47].source = r'''
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
rk = rob_tab[rob_tab.check == 'K']
for method in METHOD_ORDER[:-1]:
    group = rk[rk.method == method]
    axs[0, 0].plot(group.setting.astype(int), group.ratio, marker='o', ms=4, color=METHOD_COLOR[method], label=method)
group = rob_tab[rob_tab.check == 'gamma']
axs[0, 0].plot(group.K, group.ratio, marker='s', color=METHOD_COLOR['Louvain'], label='Louvain (resolution varied)')
axs[0, 0].axhline(1, color=INK2, lw=1, ls='--')
axs[0, 0].set(xlabel='Number of communities K', ylabel='Within / between mean Jaccard', title='(a) Sensitivity to community granularity')
axs[0, 0].legend(fontsize=8)

dp = rob_tab[rob_tab.check == 'drop popular artists']
xs = ['none', 'top 1%', 'top 5%', 'top 10%']
for method in METHOD_ORDER:
    group = dp[dp.method == method]
    ratio_values = [main_tab.loc[method, 'ratio S_same/S_diff']] + list(group.ratio)
    contrast_values = [main_tab.loc[method, 'T']] + list(group.T)
    axs[0, 1].plot(xs, ratio_values, marker='o', ms=4, color=METHOD_COLOR[method], label=method)
    axs[1, 0].plot(xs, contrast_values, marker='o', ms=4, color=METHOD_COLOR[method], label=method)
axs[0, 1].axhline(1, color=INK2, lw=1, ls='--')
axs[1, 0].axhline(0, color=INK2, lw=1, ls='--')
axs[0, 1].set(xlabel='Most popular artists removed', ylabel='Within / between mean Jaccard', title='(b) Relative contrast after changing artist set')
axs[1, 0].set(xlabel='Most popular artists removed', ylabel='Equal-community Jaccard contrast T', title='(c) Absolute contrast after changing artist set')

dt = rob_tab[rob_tab.check == 'tau']
for method in [PAPER_METHOD, MAIN_METHOD]:
    group = dt[dt.method == method]
    axs[1, 1].plot(group.setting.astype(str), group.T, color=METHOD_COLOR[method], marker='o', label=method)
axs[1, 1].set(xlabel='Regularization tau', ylabel='Equal-community Jaccard contrast T', title='(d) Regularization and ratio truncation')
axs[1, 1].legend(fontsize=8)
fig.text(.5, -.01, 'LastFM Asia; fixed graph-only partitions for artist removal. Changing the artist set changes the Jaccard denominator.', ha='center', fontsize=9)
fig.tight_layout(); savefig(fig, 'fig08_robustness'); plt.show()
'''.strip()
# Avoid matrix-transpose attribute ambiguity for the column named T.
nb.cells[47].source = nb.cells[47].source.replace('list(group.T)', "list(group['T'])").replace('group.setting.astype(str), group.T,', "group.setting.astype(str), group['T'],")
nb.cells[54].source = nb.cells[54].source.replace("a.set_title(sname, fontsize=9)", "a.set_title(sname.replace(': ', ':\\n'), fontsize=9)")
nb.cells[62].source += '\n- Phipson, B. and Smyth, G. K. (2010). [Permutation P-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf). *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39.\n- Hemerik, J. and Goeman, J. (2018). [Exact testing with random permutations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/). *TEST* 27, 811-825.\n\nPrimary algorithm/data sources: [SCORE](https://arxiv.org/abs/1211.5803), [SCORE+](https://arxiv.org/abs/1811.05927), [degree-corrected block models](https://arxiv.org/abs/1008.3926), and [SNAP LastFM Asia](https://snap.stanford.edu/data/feather-lastfm-social.html).'
nb.cells[70].source = nb.cells[70].source.replace("Path('scripts/run_notebook.py')]", "Path('scripts/run_notebook.py'), Path('requirements.txt'), Path('requirements-lock.txt')]")
nb.cells[70].source = nb.cells[70].source.replace("primary_inference=main_primary.to_dict(orient='records'),", "primary_inference=main_primary.to_dict(orient='records'),\n    all_primary_comparisons=primary.to_dict(orient='records'),\n    simulation_summary=sim_tab.to_dict(orient='records'),\n    calibration_summary=cal_summary,\n    implementation_diagnostics=embedding_rows,\n    sensitivity_artifacts=['tab12_robustness.csv', 'tab19_joint_binning_sensitivity.csv', 'tab21_community_influence.csv'],")
nbformat.validate(nb)
nbformat.write(nb, path)
