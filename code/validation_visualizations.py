"""English figures for the targeted statistical validation experiments."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from statistical_validation import CORE, RESOLUTIONS

COLORS = {'Spectral':'#eda100', 'SCORE':'#1baf7a', 'Louvain':'#eb6834', 'Louvain (fixed resolution)':'#a5adb5'}
SCHEMES = ['degree_activity_q3', 'degree_activity_q5', 'degree_activity_q7', 'degree_activity_q10', 'degree_exact_activity_q5']
SCHEME_LABELS = ['q=3', 'q=5\nprimary', 'q=7', 'q=10', 'Exact degree\nactivity q=5']
INK, MUTED = '#17212b', '#586473'


def _calibration(table):
    scenarios = ['exact_binned_null', 'smooth_degree_only_stress', 'planted_music_positive_control']
    figure, axes = plt.subplots(2, 3, figsize=(14.8, 8.1))
    for col, (scenario, title) in enumerate(zip(scenarios, ['Invariant binned null', 'Continuous-degree-only stress', 'Planted music positive control'])):
        for row, statistic in enumerate(['equal', 'pair']):
            ax = axes[row, col]
            subset = table[table.scenario.eq(scenario) & table.statistic.eq(statistic)]
            for j, method in enumerate(CORE):
                for scheme, offset, marker in [('degree_activity_q5', -.10, 'o'), ('degree_exact_activity_q5', .10, '^')]:
                    value = subset[subset.method.eq(method) & subset.scheme.eq(scheme)].iloc[0]
                    estimate = value.rejection_fraction
                    ax.errorbar(j+offset, estimate, yerr=[[estimate-value.pointwise_MC_low], [value.pointwise_MC_high-estimate]],
                        fmt=marker, color=COLORS[method], ms=6, capsize=3, lw=1.1,
                        markerfacecolor=COLORS[method] if marker=='o' else 'white')
            ax.axhline(.05, color=INK, ls='--', lw=1)
            limit = max(.15, subset.pointwise_MC_high.max()*1.08) if col==0 else 1.04
            ax.set(ylim=(0, min(limit, 1.04)), xlabel='Graph-only detection method')
            ax.set_xticks(range(3), CORE, fontsize=9)
            if col==0:
                ax.set_ylabel(('Equal-community T' if row==0 else 'User-pair T')+'\nRejection fraction', fontsize=10)
            if row==0:
                ax.set_title(title, fontsize=11, pad=10)
    handles = [Line2D([],[],marker='o',ls='',color=INK,label='Primary degree/activity q5 bins'),
               Line2D([],[],marker='^',ls='',color=INK,markerfacecolor='white',label='Exact degree / activity q5 bins')]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.065), ncol=2, fontsize=9)
    reps, draws = int(table.replications.iloc[0]), int(table.inner_permutations.iloc[0])
    figure.suptitle('Does the primary permutation procedure behave as intended?', fontsize=15, weight='bold', y=.985)
    figure.text(.5,.012, f'{reps} independent music replications; {draws} permutations each; bars: pointwise binomial Monte Carlo 95% intervals. Dashed line: alpha=.05.\n'
        'Middle column: no community term beyond exact degree; q5 is approximate, so its rejection rate is a stress diagnostic, not exact-null size.\n'
        'Left: both references are invariant nulls. Right: rejection measures power. These experiments do not establish exchangeability in LastFM.',
        ha='center', fontsize=8, color=MUTED, linespacing=1.4)
    figure.subplots_adjust(left=.07,right=.985,bottom=.21,top=.90,hspace=.30,wspace=.24)
    return figure


def _binning(table, support):
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 8))
    for method in CORE:
        for ax, statistic, title in [(axes[0,0],'equal','(a) Equal-community excess contrast'),
                                     (axes[0,1],'pair','(b) User-pair excess contrast')]:
            group = table[table.method.eq(method) & table.statistic.eq(statistic)].set_index('scheme').loc[SCHEMES]
            ax.plot(range(5), group.excess_T, marker='o', color=COLORS[method], label=method)
            ax.axhline(0,color=INK,ls='--',lw=.8)
            ax.set(title=title,ylabel='Observed T minus\npermutation mean')
        values = support[support.method.eq(method)].set_index('scheme').loc[SCHEMES]
        axes[1,0].plot(range(5), values.residual_log_degree_fraction,marker='o',color=COLORS[method],label=method)
        axes[1,1].plot(range(5), values.expected_labels_changed,marker='o',color=COLORS[method],label=method)
    axes[1,0].set(title='(c) Residual degree/partition association',ylabel='Between-community fraction of\nresidual log-degree variance')
    axes[1,1].set(title='(d) Randomization support',ylabel='Expected fraction of labels changed',ylim=(0,1))
    for ax in axes.flat:
        ax.set_xticks(range(5), SCHEME_LABELS,fontsize=8)
        ax.set_xlabel('Prespecified stratification scheme')
        ax.axvspan(.86,1.14,color='#e9edf0',alpha=.45,zorder=0)
    handles, labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.075), ncol=3, fontsize=9)
    figure.suptitle('How sensitive is community/music inference to nuisance stratification?',fontsize=15,weight='bold',y=.98)
    counts = table.groupby('scheme').B.first()
    figure.text(.5,.015,f'LastFM Asia, March 2020 snapshot. Primary q5: B={int(counts["degree_activity_q5"])}; other references: B={int(counts["degree_activity_q3"])}.\n'
        'Comparisons are exploratory. Exact degree does not mean exact activity adjustment. Residual covariate association is diagnostic, not a test of music exchangeability.',
        ha='center',fontsize=8,color=MUTED,linespacing=1.4)
    figure.subplots_adjust(left=.085,right=.985,bottom=.21,top=.90,hspace=.46,wspace=.24)
    return figure


def _recovery(raw, summary, independent):
    if independent:
        figure, axes = plt.subplots(2,2,figsize=(13.8,8.1),sharey=True)
        cases = [(sd, ratio) for sd in [0.,1.2] for ratio in [3.,8.]]
        groups = [(ax, raw[raw.degree_sd.eq(sd) & raw.within_between_ratio.eq(ratio)],
                   f'degree sd={sd:g}; within/between={ratio:g}') for ax,(sd,ratio) in zip(axes.flat,cases)]
    else:
        figure, axes = plt.subplots(1,3,figsize=(14.8,5.8),sharey=True)
        groups = [(ax, group, setting.replace(': ', ':\n')) for ax,(setting,group) in zip(axes,raw.groupby('setting',sort=False))]
    methods = CORE+['Louvain (fixed resolution)']
    metric = 'ARI' if independent else 'NMI'
    rng = np.random.default_rng(236)
    for ax, group, title in groups:
        for j, method in enumerate(methods):
            values = group[group.method.eq(method)][metric]
            selected = summary[summary.setting.eq(group.setting.iloc[0]) & summary.method.eq(method)].iloc[0]
            mean, lower, upper = selected[metric+'_mean'], selected[metric+'_ci_low'], selected[metric+'_ci_high']
            ax.scatter(j+rng.uniform(-.09,.09,len(values)),values,s=9,alpha=.35,color=COLORS[method])
            if np.isfinite(lower) and np.isfinite(upper):
                ax.errorbar(j,mean,yerr=[[mean-lower],[upper-mean]],fmt='D',color=COLORS[method],capsize=4,ms=6,lw=1.5)
            else:
                ax.scatter(j,mean,color=COLORS[method],marker='D')
        ax.set_xticks(range(4), ['Spectral','SCORE','Louvain\nK-assisted','Louvain\nfixed gamma'],fontsize=9)
        ax.set(title=title,ylim=(-.08,1.08),xlabel='Detection method / information regime')
        gamma_group = group[group.method.eq('Louvain')]
        match = gamma_group.estimated_K.eq(gamma_group.target_K).mean()
        ax.text(.5,.035,f'K-matched Louvain runs: {match:.0%}',transform=ax.transAxes,ha='center',fontsize=8,color=MUTED)
    (axes.flat[0] if independent else axes[0]).set_ylabel(metric+' with planted labels')
    reps = int(summary.replications.iloc[0])
    if independent:
        figure.suptitle('Independently specified Bernoulli DCBM: matched K information',fontsize=15,weight='bold',y=.985)
        figure.text(.5,.012,f'{reps} paired graph replications per setting; three planted groups; expected full-graph mean degree 18.\n'
            'Same connected evaluation population across degree/signal regimes within each replication; probabilities are never clipped and graphs are not redrawn.\n'
            'Diamonds/bars: mean and pointwise Monte Carlo 95% interval. Gray fixed-resolution Louvain is a reference, not part of the matched-information comparison.',
            ha='center',fontsize=8,color=MUTED,linespacing=1.4)
        figure.subplots_adjust(left=.07,right=.985,bottom=.18,top=.90,hspace=.43,wspace=.20)
    else:
        figure.suptitle('Does unequal K information change the calibrated recovery comparison?',fontsize=15,weight='bold',y=.985)
        figure.text(.5,.02,f'{reps} reproduced legacy graphs per setting; all methods evaluated on identical retained users in each graph.\n'
            'Louvain K assistance selects resolution using graph cluster counts only, never membership truth. Actual K matches are recorded.\n'
            'Points: individual graphs; diamonds/bars: mean and pointwise Monte Carlo 95% interval. These are generator-conditional recovery results.',
            ha='center',fontsize=8,color=MUTED,linespacing=1.4)
        figure.subplots_adjust(left=.055,right=.985,bottom=.25,top=.84,wspace=.19)
    return figure


def _stability(raw, seeds):
    figure, axes = plt.subplots(2,2,figsize=(13.3,7.9))
    rng = np.random.default_rng(236)
    metrics = [('NMI','(a) Partition stability under shared edge deletion','NMI on the same retained users'),
               ('T_change_same_users','(b) Equal-community music contrast change','Perturbed T minus restricted original T'),
               ('pair_T_change_same_users','(c) User-pair music contrast change','Perturbed T minus restricted original T')]
    for ax,(metric,title,ylabel) in zip(axes.flat[:3],metrics):
        for j,method in enumerate(CORE):
            values = raw[raw.method.eq(method)][metric]
            ax.scatter(j+rng.uniform(-.10,.10,len(values)),values,color=COLORS[method],s=20,alpha=.65)
            ax.plot([j-.16,j+.16],[values.mean()]*2,color=INK,lw=2)
        ax.set(title=title,ylabel=ylabel)
        if metric!='NMI':
            ax.axhline(0,color=INK,lw=.8,ls='--')
        else:
            ax.set_ylim(0,1.03)
    ax = axes[1,1]
    for j,method in enumerate(CORE):
        values = seeds[seeds.method.eq(method)].NMI_to_original
        ax.scatter(j+rng.uniform(-.10,.10,len(values)),values,color=COLORS[method],s=25,alpha=.7)
        ax.plot([j-.16,j+.16],[values.mean()]*2,color=INK,lw=2)
    ax.set(title='(d) Seed sensitivity on the unperturbed graph',ylabel='NMI to original partition',ylim=(0,1.03))
    for ax in axes.flat:
        ax.set_xticks(range(3),CORE)
        ax.set_xlabel('Original community detection method')
    reps, seed_count = raw.rep.nunique(), seeds.seed.nunique()
    figure.suptitle('Are the community assignments and music contrasts stable?',fontsize=15,weight='bold',y=.98)
    figure.text(.5,.015,f'LastFM Asia, March 2020 snapshot: {reps} common 10% edge-deletion masks; {seed_count} random seeds per method.\n'
        f'Retained users: {raw.retained_fraction.min():.1%} to {raw.retained_fraction.max():.1%}. Each T change compares identical retained users.\n'
        'Points are perturbations/seeds; black bars are means. These are algorithmic sensitivity diagnostics, not population confidence intervals.',
        ha='center',fontsize=8,color=MUTED,linespacing=1.4)
    figure.subplots_adjust(left=.085,right=.985,bottom=.18,top=.90,hspace=.38,wspace=.25)
    return figure


def generate_validation_figures(run_root, show=False):
    run = Path(run_root)
    tables, outputs, figures = run/'tables', run/'outputs', run/'figures'
    calibration = pd.read_csv(tables/'tab26_primary_calibration_summary.csv')
    binning = pd.read_csv(tables/'tab23_primary_binning_sensitivity.csv')
    support = pd.read_csv(tables/'tab24_primary_strata_support.csv')
    independent = pd.read_csv(tables/'tab28_independent_recovery_raw.csv')
    independent_summary = pd.read_csv(tables/'tab28_independent_recovery_summary.csv')
    legacy = pd.read_csv(tables/'tab31_matched_legacy_recovery_raw.csv')
    legacy_summary = pd.read_csv(tables/'tab31_matched_legacy_recovery_summary.csv')
    perturbations = pd.read_csv(tables/'tab33_core_perturbation_raw.csv')
    seeds = pd.read_csv(tables/'tab35_core_seed_runs.csv')
    style = {'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,
             'axes.spines.right':False,'axes.grid':True,'axes.axisbelow':True,'grid.color':'#e3e7eb',
             'grid.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'}
    with mpl.rc_context(style):
        rendered = [
            ('fig14_primary_calibration',_calibration(calibration)),
            ('fig15_primary_binning_sensitivity',_binning(binning,support)),
            ('fig16_independent_matched_recovery',_recovery(independent,independent_summary,True)),
            ('fig17_legacy_K_fairness',_recovery(legacy,legacy_summary,False)),
            ('fig18_core_stability',_stability(perturbations,seeds)),
        ]
        with PdfPages(figures/'statistical_validation_review.pdf',metadata={'Title':'STAT 236 - Targeted Statistical Validation','Author':'Faye Fang and Grace Chen'}) as pdf:
            for stem,figure in rendered:
                figure.savefig(figures/(stem+'.png'),dpi=300,bbox_inches='tight')
                figure.savefig(figures/(stem+'.pdf'),bbox_inches='tight')
                pdf.savefig(figure,bbox_inches='tight')
        if not show:
            for _,figure in rendered:
                plt.close(figure)
    guide = pd.DataFrame([
        ('fig14_primary_calibration','Primary inference','Invariant-null size, continuous-degree stress and planted music power',
         'Stress rejection is not exact-binned-null type I error; intervals describe Monte Carlo precision.'),
        ('fig15_primary_binning_sensitivity','Primary inference','Reference-centered contrasts, residual covariates and label mixing',
         'Binning comparisons are exploratory; exact degree is not exact activity adjustment.'),
        ('fig16_independent_matched_recovery','Does the method matter?','Independent Bernoulli design with common users and equal K information',
         'Recovery is conditional on the specified generator and common-core selection.'),
        ('fig17_legacy_K_fairness','Does the method matter?','Known-K versus fixed-resolution Louvain on reproduced legacy graphs',
         'Matching uses cluster counts only; actual K-match rates and retained populations are reported.'),
        ('fig18_core_stability','Community structure and music validation','Shared edge perturbations and random-seed sensitivity for all core methods',
         'Perturbations and seeds are algorithmic sensitivity diagnostics, not population sampling uncertainty.'),
    ],columns=['figure','research_question','what_it_shows','interpretation_limit'])
    guide['suggested_report_placement'] = [
        'Appendix; summarize continuous-degree stress qualification in main methods/limitations.',
        'Appendix; state whether the main conclusion persists under exact-degree sensitivity.',
        'Main results, selecting panels if space is limited; generator and pointwise MC intervals in caption.',
        'Appendix; discuss K information when interpreting Appendix A rankings.',
        'Main robustness subsection or appendix; distinguish label instability from music-contrast stability.',
    ]
    guide.to_csv(tables/'tab37_validation_figure_guide.csv',index=False)
    return [stem for stem,_ in rendered]


def write_validation_manifest(run_root, config, calibration_design, profile):
    run = Path(run_root)
    digest = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    calibration = pd.read_csv(run/'tables/tab26_primary_calibration_summary.csv')
    summary = pd.read_csv(run/'tables/tab28_independent_recovery_summary.csv')
    legacy = pd.read_csv(run/'tables/tab31_matched_legacy_recovery_summary.csv')
    warnings = []
    if calibration.evidence_excess_size.any():
        warnings.append('Simultaneous Monte Carlo lower bounds indicate excess rejection in at least one constructed invariant-null case.')
    stress = calibration[calibration.scenario.eq('smooth_degree_only_stress') & calibration.scheme.eq('degree_activity_q5')]
    if stress.rejection_fraction.max()>.10:
        worst = stress.loc[stress.rejection_fraction.idxmax()]
        exact = calibration[calibration.scenario.eq(worst.scenario) & calibration.scheme.eq('degree_exact_activity_q5')
                            & calibration.method.eq(worst.method) & calibration.statistic.eq(worst.statistic)].iloc[0]
        warnings.append(f'Continuous-degree-only stress: {worst.method}/{worst.statistic} rejects in {worst.rejection_fraction:.1%} of q5 runs versus {exact.rejection_fraction:.1%} with exact degree. '
            'Interpret the primary result as a binned reference comparison, not exact continuous-covariate adjustment.')
    match_limits = []
    for name, frame in [('independent',summary),('calibrated',legacy)]:
        minimum = frame[frame.method.eq('Louvain') & frame.information_regime.eq('known_K_assisted')].K_match_fraction.min()
        if minimum < 1:
            match_limits.append(f'{name} minimum setting-level match rate {minimum:.1%}')
    if match_limits:
        warnings.append('The finite Louvain resolution grid did not attain target K in every graph ('+'; '.join(match_limits)+
                        '). Equal K information does not ensure equal achieved K; retain actual match rates.')
    files = [p for p in (run/'tables').glob('tab*.csv') if (match:=re.match(r'tab(\d+)_',p.name)) and 23<=int(match.group(1))<=37]
    files += list((run/'figures').glob('fig1[4-8]_*.png'))+list((run/'figures').glob('fig1[4-8]_*.pdf'))
    files += [run/'figures/statistical_validation_review.pdf']
    files += [run/'outputs'/name for name in ['primary_binning_nulls.npz','primary_calibration_nulls.npz',
              'core_perturbation_partitions.npz','core_seed_partitions.npz']]
    sources = [Path(__file__),Path(__file__).with_name('statistical_validation.py')]
    result = dict(profile=profile, configuration=asdict(config), core_methods=CORE, figure_review_pages=5,
        calibration_design=calibration_design,
        independent_design=dict(n=config.recovery_n,K=3,expected_mean_degree=18,degree_sd=[0.,1.2],within_between_ratio=[3.,8.],
            theta_bounds=[.35,2.5],probability_model='independent Bernoulli edges before common-core selection; no clipping',
            population='same iteratively selected connected core across all four regimes within each replicate',
            pairing='common planted labels and uniform edge draws across regimes; independent replications',
            minimum_mean_retained_fraction=float(summary.mean_retained_fraction.min())),
        louvain_K_assistance=dict(resolution_grid=list(RESOLUTIONS),
            rule='First exact count match in prespecified grid order; otherwise closest count, breaking ties by distance of log resolution from zero.',
            information='Graph and K only. No planted memberships, artist identities or music similarities are used to select resolution.'),
        interpretation_limits=['Exploratory additions after prior inspection.',
            'Invariant-null calibration is conditional on one fixed generated graph; it cannot establish observational exchangeability.',
            'Continuous-degree stress is not the same null as coarse-bin invariance.',
            'Exact-degree stratification still bins artist activity.',
            'Recovery intervals are pointwise Monte Carlo intervals under specified generators.',
            'Graph perturbation and seed ranges are not population effect confidence intervals.'],
        warnings=warnings, artifact_sha256={str(p.relative_to(run)):digest(p) for p in files},
        source_sha256={'code/'+p.name:digest(p) for p in sources})
    (run/'outputs/statistical_validation_manifest.json').write_text(json.dumps(result,indent=2))
    return result
