from pathlib import Path
p=Path('code/scripts/audit_statistics.py')
s=p.read_text()
s=s.replace("scope='Statistical review only; existing production analysis unchanged.'", "scope='Statistical review and follow-up on targeted validation; fitted primary partitions and hypotheses unchanged.'")
s=s.replace("'The existing constructed-null calibration tests country-only randomization, not the primary degree/activity reference.',", "'Primary-reference calibration is conditional on one generated graph and cannot establish observational exchangeability.',")
s=s.replace("'Edge-deletion stability currently assesses truncated SCORE+; the three core methods do not receive matched graph perturbations.',", "'Shared perturbations and seeds assess algorithmic sensitivity, not uncertainty over a population network.',")
s=s.replace("destination = ROOT / 'outputs/statistical_audit.json'", """validation_path = ROOT/'outputs/statistical_validation_manifest.json'
if validation_path.exists():
    followup=json.loads(validation_path.read_text())
    result['validation_followup']=dict(
        manifest='outputs/statistical_validation_manifest.json',
        configuration=followup['configuration'], warnings=followup['warnings'],
        completed_actions=['Primary degree/activity calibration and five nuisance references for core methods.',
            'Matched-K recovery on reproduced legacy graphs and independent Bernoulli designs.',
            'Shared graph perturbations and seed checks for Spectral, SCORE and Louvain.'],
        interpretation='Original fairness diagnostics above describe Appendix A before K assistance; Appendix D addresses this comparison. Remaining limits and flags must accompany report claims.')
    for action in result['prioritized_actions']:
        action['status']='addressed in Appendix D' if action['priority']!='Interpretation' else 'ongoing reporting qualification'
    result['assessment']='Targeted statistical additions completed; inference remains conditional and exploratory.'
destination = ROOT / 'outputs/statistical_audit.json'""")
p.write_text(s)
