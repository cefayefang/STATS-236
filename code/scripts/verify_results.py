"""Audit that saved full-run outputs agree and retain their Monte Carlo evidence."""
from pathlib import Path
import hashlib
import json
import sys

import nbformat
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))


def require(condition, description):
    if not condition:
        raise AssertionError(description)
    checks.append(description)


checks = []
manifest = json.loads((ROOT / 'outputs/run_manifest.json').read_text())
config = manifest['configuration']
require(manifest['profile'] == 'full', 'Manifest identifies the full analysis')
require(config['B_MAIN'] == 1999 and config['N_SIM'] == 20, 'Full permutation and simulation counts recorded')
for path, expected in {**manifest['input_sha256'], **manifest['source_sha256']}.items():
    actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    require(actual == expected, f'Input/source SHA256 matches: {path}')

notebook = nbformat.read(CODE_DIR / 'STAT236_LastFM_Community_Music.ipynb', as_version=4)
nbformat.validate(notebook)
codes = [cell for cell in notebook.cells if cell.cell_type == 'code']
require([cell.execution_count for cell in codes] == list(range(1, len(codes) + 1)), 'Every notebook code cell executed in order')
require(not any(output.output_type == 'error' for cell in codes for output in cell.outputs), 'Executed notebook has no error outputs')
source_hash = hashlib.sha256(''.join(cell.source for cell in notebook.cells).encode()).hexdigest()
require(source_hash == manifest['notebook_cell_source_sha256'], 'Executed notebook source matches run manifest')
log = json.loads((ROOT / 'outputs/execution_log.json').read_text())
require(len(log['cells']) == len(codes) and all(c['success'] for c in log['cells']), 'Execution log records success for every cell')

partitions = pd.read_csv(ROOT / 'outputs/partitions.csv', index_col='node_id')
require(partitions.index.is_unique and len(partitions) == manifest['data']['n'], 'Partitions align to unique retained user IDs')
require(partitions.shape[1] == 5, 'Both SCORE+ implementations and all baselines retained')
require(all(partitions[col].min() == 0 for col in partitions), 'Community encoding starts at zero')

inference = pd.read_csv(ROOT / 'tables/tab16_inference_all_references.csv')
require(len(inference) == 30 and len(inference.drop_duplicates(['Method', 'strata', 'statistic'])) == 30,
        'Thirty distinct reference/method/estimand combinations saved')
primary = inference[inference.primary_family]
require(len(primary) == 3 and set(primary.Method) == {'Spectral', 'SCORE', 'Louvain'},
        'Primary family retains the three original community detection methods')
require(primary.strata.eq('degree_activity_q5').all() and primary.statistic.eq('equal').all(),
        'Main research question uses music similarity with degree/activity stratification')
require(not inference[inference.strata.str.startswith('country')].primary_family.any(),
        'Country references are supplementary sensitivity analyses')
require(inference.loc[~inference.primary_family, 'p_Holm_primary_3_methods'].isna().all(),
        'Primary correction is not assigned to extensions or country checks')
ordered_primary = primary.sort_values('p')
expected_primary = np.minimum(1, np.maximum.accumulate(ordered_primary.p.to_numpy() * np.arange(3, 0, -1)))
require(np.allclose(ordered_primary.p_Holm_primary_3_methods, expected_primary),
        'Primary Holm correction applies to exactly the original three methods')
require(set(inference.Method) == set(partitions), 'Inference includes every fitted method')
require(np.isfinite(inference.select_dtypes(include='number').drop(columns=['p_Holm_primary_3_methods'])).all().all(),
        'Inference table contains finite numeric values outside inapplicable primary corrections')
require(np.allclose(inference.p, (inference.exceedances + 1) / (inference.B + 1), atol=1e-13),
        'Saved p-values equal the plus-one Monte Carlo formula')
require(np.allclose(inference.excess_T, inference['T'] - inference.null_mean, atol=1e-12),
        'Saved excess contrasts equal observed minus null mean')

nulls = np.load(ROOT / 'outputs/permutation_nulls.npz')
require(len(nulls.files) == len(inference), 'Every principal inference row has raw null draws')
for row in inference.itertuples():
    values = nulls[f'{row.strata}__{row.Method}__{row.statistic}']
    require(len(values) == row.B, f'Raw permutation count: {row.strata}/{row.Method}/{row.statistic}')
    count = np.count_nonzero(values >= row.T - 100 * np.finfo(float).eps * max(abs(row.T), 1))
    require(int(count) == row.exceedances, f'Raw tail exceedances: {row.strata}/{row.Method}/{row.statistic}')
    require(abs(values.mean() - row.null_mean) < 1e-12, f'Raw null mean: {row.strata}/{row.Method}/{row.statistic}')
    require(abs(values.std(ddof=1) - row.null_sd) < 1e-12, f'Raw null SD: {row.strata}/{row.Method}/{row.statistic}')

for scheme, group in inference.groupby('strata'):
    order = np.argsort(group.p.to_numpy(), kind='stable')
    ordered = group.p.to_numpy()[order]
    expected = []
    previous = 0
    for j, value in enumerate(ordered):
        previous = max(previous, min(1, (len(group) - j) * value))
        expected.append(previous)
    require(np.allclose(group.p_Holm_within_reference_10_tests.to_numpy()[order], expected), f'Holm correction agrees: {scheme}')

simulation = pd.read_csv(ROOT / 'tables/tabA1_simulation_raw.csv')
require(len(simulation) == 3 * config['N_SIM'] * 5, 'All graph simulation replicates and methods saved')
require(simulation.groupby(['setting', 'rep']).size().eq(5).all(), 'Methods are paired on every simulated graph')
require(simulation.groupby(['setting', 'rep']).n_lcc.nunique().eq(1).all(), 'Paired methods use identical retained vertex counts')
require(simulation.NMI.between(0, 1).all() and simulation.ARI.between(-1, 1).all(), 'Recovery scores lie in valid ranges')
summary = pd.read_csv(ROOT / 'tables/tabA2_simulation_summary.csv')
require(np.isfinite(summary.select_dtypes(include='number')).all().all(), 'Full-run simulation uncertainty summaries are finite')
calibration = pd.read_csv(ROOT / 'tables/tabB1_calibration_raw.csv')
require(len(calibration) == 400, 'Both constructed-null procedures evaluated for 200 outer assignments')

rubric = pd.read_csv(ROOT / 'tables/tab00_report_rubric_evidence.csv')
require(rubric.weight_percent.sum() == 100 and len(rubric) == 7, 'Rubric covers all seven criteria and 100 percent')
evidence = json.loads((ROOT / 'outputs/report_evidence.json').read_text())
require(len(evidence['all_primary_comparisons']) == 3, 'Report evidence retains every original-method primary comparison')
require(len(evidence['simulation_summary']) == 15, 'Report evidence retains every simulated setting/method')
require(len(list((ROOT / 'figures').glob('*.png'))) == 20, 'Twenty PNG figures exported, including statistical validation')
require(len(list((ROOT / 'figures').glob('*.pdf'))) == 22, 'Twenty individual PDF figures and two review booklets exported')
embedded = sum('image/png' in output.get('data', {}) for cell in codes for output in cell.outputs)
require(embedded == 20, 'All twenty figures embedded in executed notebook')
visual = json.loads((ROOT / 'outputs/visualization_manifest.json').read_text())
require(visual['core_methods'] == ['Spectral', 'SCORE', 'Louvain'], 'New visual comparisons retain the original methods')
require(visual['figure_review_pages'] == 6, 'Figure review contains six selected research figures')
require(hashlib.sha256((ROOT / visual['source_file']).read_bytes()).hexdigest() == visual['source_sha256'],
        'Visualizations match the recorded plotting source')
for name, expected in visual['artifact_sha256'].items():
    require(hashlib.sha256((ROOT / 'figures' / name).read_bytes()).hexdigest() == expected,
            f'Visual artifact SHA256 matches: {name}')
require(hashlib.sha256((ROOT / visual['guide']).read_bytes()).hexdigest() == visual['guide_sha256'],
        'English figure guide matches visualization manifest')

# Targeted statistical additions: verify evidence, not favorable scientific outcomes.
import lastfm_stats as stats
from statistical_validation import CORE, RESOLUTIONS, monte_carlo_mean
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
for tracefile, raw in [('tab30_independent_resolution_search.csv',indraw),('tab32_legacy_resolution_search.csv',legacy)]:
    trace=pd.read_csv(ROOT/'tables'/tracefile)
    for (setting,rep),group in trace.groupby(['setting','rep'],sort=False):
        chosen=raw[raw.setting.eq(setting)&raw.rep.eq(rep)&raw.method.eq('Louvain')].iloc[0]
        require(np.allclose(group.resolution,list(RESOLUTIONS)[:len(group)]),f'Prespecified Louvain search order: {tracefile}/{setting}/{rep}')
        exact=group[group.estimated_K.eq(chosen.target_K)]
        if len(exact):
            expected=exact.resolution.iloc[0]
            valid=len(exact)==1 and exact.index[-1]==group.index[-1]
        else:
            candidate=min(group.itertuples(),key=lambda r:(abs(r.estimated_K-chosen.target_K),abs(np.log(r.resolution))))
            expected=candidate.resolution
            valid=len(group)==len(RESOLUTIONS)
        require(valid and np.isclose(chosen.resolution,expected),f'Louvain selection follows cluster-count-only rule: {tracefile}/{setting}/{rep}')
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

report = dict(success=True, check_count=len(checks), checks=checks,
              execution_seconds=log['total_seconds'], profile=manifest['profile'])
(ROOT / 'outputs/verification.json').write_text(json.dumps(report, indent=2))
print(f"PASS: {len(checks)} result-integrity checks; full run {log['total_seconds']:.1f} seconds; {embedded} embedded figures")
