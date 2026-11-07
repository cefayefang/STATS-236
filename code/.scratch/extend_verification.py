from pathlib import Path
p=Path('code/scripts/verify_results.py')
s=p.read_text().replace("== 15, 'Fifteen PNG figures exported, including original-method visual comparisons'", "== 20, 'Twenty PNG figures exported, including statistical validation'")
s=s.replace("== 16, 'Fifteen individual PDF figures and one figure review exported'", "== 22, 'Twenty individual PDF figures and two review booklets exported'")
s=s.replace("embedded == 15, 'All fifteen figures embedded in executed notebook'", "embedded == 20, 'All twenty figures embedded in executed notebook'")
insert='''# Targeted statistical additions: verify evidence, not favorable scientific outcomes.
import lastfm_stats as stats
from statistical_validation import CORE, monte_carlo_mean
validation = json.loads((ROOT/'outputs/statistical_validation_manifest.json').read_text())
vc = validation['configuration']
require(validation['profile']=='full', 'Statistical validation uses full configuration')
require(validation['core_methods']==CORE and validation['figure_review_pages']==5,
        'Validation retains three original methods and five review pages')
for path, expected in {**validation['artifact_sha256'], **validation['source_sha256']}.items():
    require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected, f'Validation SHA256 matches: {path}')
bins = pd.read_csv(ROOT/'tables/tab23_primary_binning_sensitivity.csv')
bin_nulls = np.load(ROOT/'outputs/primary_binning_nulls.npz')
require(len(bins)==30 and bins.groupby(['scheme','statistic']).size().eq(3).all(),
        'Five prespecified nuisance references cover three methods and both estimands')
for row in bins.itertuples():
    key = f'{row.scheme}__{row.method}__{row.statistic}'
    values = bin_nulls[key]
    expected_B = config['B_MAIN'] if row.scheme=='degree_activity_q5' else vc['bin_B']
    require(len(values)==row.B==expected_B, f'Binning raw draw count: {key}')
    result = stats.summarize_null(row.T,values)
    require(np.isclose(result['p'],row.p) and np.isclose(result['null_mean'],row.null_mean,atol=1e-12),
            f'Binning tail and null mean agree: {key}')
    if row.scheme=='degree_activity_q5':
        require(np.array_equal(values,nulls[key]),f'Primary original draws reused: {key}')
for _,group in bins.groupby(['scheme','statistic']):
    require(np.allclose(stats.holm_adjust(group.p.values),group.p_Holm_3_methods_within_scheme_statistic),
            f'Exploratory Holm family agrees: {group.scheme.iloc[0]}/{group.statistic.iloc[0]}')
support = pd.read_csv(ROOT/'tables/tab24_primary_strata_support.csv')
require(len(support)==15 and support.expected_labels_changed.between(0,1).all(), 'Binning support recorded for every partition/reference')
require(support.loc[support.scheme.eq('degree_exact_activity_q5'),'residual_log_degree_fraction'].eq(0).all(),
        'Exact-degree strata remove within-cell degree variation')
calraw = pd.read_csv(ROOT/'tables/tab25_primary_calibration_raw.csv')
calsummary = pd.read_csv(ROOT/'tables/tab26_primary_calibration_summary.csv')
calfamily = pd.read_csv(ROOT/'tables/tab27_primary_calibration_family.csv')
calnulls = np.load(ROOT/'outputs/primary_calibration_nulls.npz')
require(len(calraw)==3*vc['calibration_R']*2*3*2 and len(calsummary)==36 and len(calfamily)==6,
        'All calibration scenarios, methods, references and statistics present')
require(calraw.groupby(['scenario','scheme','rep','statistic']).size().eq(3).all(), 'Calibration methods share music replications')
for scenario in calraw.scenario.unique():
    values = calnulls[scenario]
    require(values.shape==(vc['calibration_R'],2,vc['calibration_B'],3,2),f'Calibration raw null dimensions: {scenario}')
    for j,scheme in enumerate(['degree_activity_q5','degree_exact_activity_q5']):
        for m,method in enumerate(CORE):
            for t,statistic in enumerate(['equal','pair']):
                group=calraw[calraw.scenario.eq(scenario)&calraw.scheme.eq(scheme)&calraw.method.eq(method)&calraw.statistic.eq(statistic)].sort_values('rep')
                observed=group['T'].values
                tolerance=100*np.finfo(float).eps*np.maximum(abs(observed),1)
                # CSV precision can slightly shift comparisons; nulls equal to the observation
                # are still checked with the established numerical tolerance.
                count=(values[:,j,:,m,t]>=observed[:,None]-tolerance[:,None]).sum(1)
                require(np.allclose(group.p,(count+1)/(vc['calibration_B']+1)),f'Calibration p-values match stored nulls: {scenario}/{scheme}/{method}/{statistic}')
                summaryrow=calsummary[calsummary.scenario.eq(scenario)&calsummary.scheme.eq(scheme)&calsummary.method.eq(method)&calsummary.statistic.eq(statistic)].iloc[0]
                require(np.isclose(group.reject.mean(),summaryrow.rejection_fraction),f'Calibration rejection summary agrees: {scenario}/{scheme}/{method}/{statistic}')
expected_invariant=calraw.scenario.eq('exact_binned_null') | (calraw.scenario.eq('smooth_degree_only_stress') & calraw.scheme.eq('degree_exact_activity_q5'))
require(np.array_equal(calraw.assumed_invariant_null,expected_invariant), 'Continuous-degree stress is not mislabeled as an invariant coarse-bin null')
require(np.array_equal(calsummary.evidence_excess_size,calsummary.assumed_invariant_null & calsummary.simultaneous_MC_lower.gt(.05)),
        'Calibration flags follow simultaneous Monte Carlo bounds; outcomes are not forced to pass')
for row in calfamily.itertuples():
    group=calraw[calraw.scenario.eq(row.scenario)&calraw.scheme.eq(row.scheme)&calraw.statistic.eq('equal')]
    pivot=group.pivot(index='rep',columns='method',values='p')[CORE]
    rejects=np.array([np.any(stats.holm_adjust(p)<=.05) for p in pivot.values])
    require(np.isclose(rejects.mean(),row.Holm_family_rejection_fraction),f'Calibration primary-family rejection agrees: {row.scenario}/{row.scheme}')
indraw=pd.read_csv(ROOT/'tables/tab28_independent_recovery_raw.csv')
require(len(indraw)==4*vc['recovery_R']*4,'Four independently specified recovery settings and four information/method variants saved')
require(np.allclose(indraw.expected_mean_degree,18) and indraw.maximum_probability.between(0,1).all(),
        'Bernoulli full-graph expected degree is 18 and all probabilities valid')
require(indraw.theta_min.ge(.35-1e-10).all() and indraw.theta_max.le(2.5+1e-10).all(), 'Independent propensity bounds preserved')
require(indraw.groupby('rep').n_common.nunique().eq(1).all(), 'Common recovery evaluation population size across all settings and methods within each replicate')
require(indraw.target_K.eq(3).all() and indraw.groupby(['setting','rep']).size().eq(4).all(), 'Planted three-group design and complete paired comparisons')
require(indraw.NMI.between(0,1).all() and indraw.ARI.between(-1,1).all(), 'Independent recovery scores in valid ranges')
legacy=pd.read_csv(ROOT/'tables/tab31_matched_legacy_recovery_raw.csv')
require(len(legacy)==3*vc['legacy_R']*4,'All original recovery graphs have matched-information and fixed-resolution comparisons')
for method, oldmethod in [('Spectral','Spectral'),('SCORE','SCORE'),('Louvain (fixed resolution)','Louvain')]:
    merged=legacy[legacy.method.eq(method)].merge(simulation[simulation.method.eq(oldmethod)],on=['setting','rep'],suffixes=('_new','_old'))
    require(len(merged)==3*vc['legacy_R'] and np.allclose(merged.NMI_new,merged.NMI_old,atol=1e-9) and np.allclose(merged.ARI_new,merged.ARI_old,atol=1e-9),f'Legacy graph recovery reproduced: {method}')
for prefix,raw in [('tab28_independent_recovery',indraw),('tab31_matched_legacy_recovery',legacy)]:
    summary=pd.read_csv(ROOT/'tables'/f'{prefix}_summary.csv')
    paired=pd.read_csv(ROOT/'tables'/f'{prefix}_paired.csv')
    for row in summary.itertuples():
        group=raw[raw.setting.eq(row.setting)&raw.method.eq(row.method)]
        require(np.isclose(group.estimated_K.eq(group.target_K).mean(),row.K_match_fraction),f'Actual K-match fraction retained: {prefix}/{row.setting}/{row.method}')
    for row in paired.itertuples():
        group=raw[raw.setting.eq(row.setting)&raw.information_regime.eq('known_K_assisted')]
        pivot=group.pivot(index='rep',columns='method',values=row.metric)
        first,second=row.comparison.split(' minus ')
        calculated=monte_carlo_mean(pivot[first]-pivot[second])
        require(all(np.isclose(getattr(row,k),v) for k,v in calculated.items()),f'Paired recovery uncertainty agrees: {prefix}/{row.setting}/{row.metric}/{row.comparison}')
interaction=pd.read_csv(ROOT/'tables/tab29_degree_correction_interaction.csv')
for row in interaction.itertuples():
    group=indraw[indraw.within_between_ratio.eq(row.within_between_ratio)&indraw.information_regime.eq('known_K_assisted')]
    pivot=group.pivot(index=['rep','degree_sd'],columns='method',values=row.metric)
    gain=(pivot.SCORE-pivot.Spectral).unstack('degree_sd')
    calculated=monte_carlo_mean(gain[1.2]-gain[0.])
    require(all(np.isclose(getattr(row,k),v) for k,v in calculated.items()),f'Degree-correction interaction agrees: ratio={row.within_between_ratio}/{row.metric}')
pert=pd.read_csv(ROOT/'tables/tab33_core_perturbation_raw.csv')
seeds=pd.read_csv(ROOT/'tables/tab35_core_seed_runs.csv')
require(len(pert)==vc['perturbation_R']*3 and pert.groupby('rep').method.nunique().eq(3).all(),'Shared perturbations cover every core method')
require(pert.groupby('rep').n_lcc.nunique().eq(1).all() and pert.groupby('rep').edge_retention_fraction.nunique().eq(1).all(), 'Shared edge masks and retained users across methods')
require(np.allclose(pert.T_change_same_users,pert['T']-pert.restricted_reference_T) and np.allclose(pert.pair_T_change_same_users,pert.T_pair-pert.restricted_reference_T_pair),
        'Perturbation contrast changes compare exactly the same retained users')
require(pert.NMI.between(0,1).all() and pert.ARI.between(-1,1).all(), 'Perturbation similarity scores in valid ranges')
require(len(seeds)==vc['seed_R']*3 and seeds.groupby('seed').method.nunique().eq(3).all(), 'Seed sensitivity covers every core method on identical seeds')
require(len(evidence['primary_calibration_summary'])==36 and len(evidence['independent_recovery_summary'])==16,
        'Report evidence includes primary calibration and independent recovery')
require(evidence['validation_warnings']==validation['warnings'], 'Report qualifications preserve diagnostic warnings')

'''
s=s.replace('report = dict(success=True',insert+'report = dict(success=True')
p.write_text(s)
