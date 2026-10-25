"""Task-scoped conclusions; descriptive quality flags never imply universal usability."""
import math

VERSION = '2.0'


def interval_direction(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return 'unavailable'
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value) or value[0] > value[1]:
        return 'unavailable'
    return 'positive' if value[0] > 0 else 'negative' if value[1] < 0 else 'inconclusive'


def summarize(result):
    q, u = result.get('quality', {}), result.get('utility', {})
    flags = [key for key, value in q.get('fidelity', {}).items() if value.get('screen_passed') is False]
    for key in ('diversity', 'overlap'):
        if q.get(key, {}).get('screen_passed') is False:
            flags.append(key)
    flags += [f'class_{key}' for key, value in q.get('per_class', {}).items() if value.get('screen_passed') is False]
    direction = interval_direction(u.get('paired_bootstrap_95_ci')) if u.get('status') == 'computed' else 'not_evaluated'
    control = interval_direction(u.get('synthetic_vs_resampling_control', {}).get('paired_bootstrap_95_ci'))
    return {'version': VERSION, 'quality_concerns': flags, 'utility_direction': direction,
            'resampling_control_direction': control,
            'scope': result.get('output_domain', 'recorded output domain'),
            'interpretation': 'Quality flags are exploratory descriptions. Utility intervals condition on one training run; they do not include training-seed variation.'}


def decision(result, pilot, mode, coverage):
    evidence = summarize(result)
    def answer(label, reason):
        return {'decision': label, 'reason': reason, 'evidence_summary': evidence}
    if result.get('status') == 'failed':
        return answer('insufficient_evidence', 'Required evaluation did not complete.')
    if not result.get('validity', {}).get('passed'):
        return answer('do_not_use', 'Outputs violate the recorded validity requirements.')
    if pilot:
        return answer('insufficient_evidence', 'Pilot evidence checks implementation; it does not establish downstream benefit.')
    u = result.get('utility', {})
    if mode != 'test':
        direction = evidence['utility_direction']
        finding = {'positive': 'The validation comparison favors the alternative.',
                   'negative': 'The validation comparison favors the baseline.',
                   'inconclusive': 'The validation interval does not establish an improvement.',
                   'not_evaluated': 'No downstream benefit was measured.',
                   'unavailable': 'The downstream comparison lacks a usable uncertainty interval.'}[direction]
        return answer('insufficient_evidence', finding + ' These are development findings; a frozen final assessment is pending. Quality flags do not prove universal unusability.')
    if not coverage:
        return answer('insufficient_evidence', 'The original plan predates this evaluation contract; current evidence is reported without retroactively approving its criteria.')
    no_generation = result.get('task') == 'representation_learning' and result.get('counts', {}).get('generated') == 0
    if result.get('quality', {}).get('status') != 'computed' and not no_generation:
        return answer('insufficient_evidence', 'Sample-quality evaluation is incomplete; no conclusion is inferred from missing checks.')
    if u.get('status') != 'computed' or not u.get('converged', False):
        return answer('insufficient_evidence', 'No complete, converged downstream comparison establishes usefulness. Distributional similarity alone is insufficient, including for PBMC biological use.')
    if evidence['utility_direction'] == 'negative':
        return answer('do_not_use', 'The frozen comparison supports worse performance for the evaluated downstream task. This conclusion is restricted to that task and setup.')
    if evidence['utility_direction'] != 'positive':
        return answer('insufficient_evidence', 'The frozen downstream interval does not establish improvement; no absolute F1 cutoff is imposed.')
    if result.get('task') == 'class_balancing' and evidence['resampling_control_direction'] != 'positive':
        return answer('insufficient_evidence', 'Improvement over real-only training does not establish a gain over the matched real-resampling control.')
    concern = ' Distributional concerns remain and must be reported; this is not a sample-realism endorsement.' if evidence['quality_concerns'] else ''
    return answer('limited_use', 'The frozen task comparison supports improvement in this specific setup. Restrict the claim to this task, population, classifier and one training run; inspect per-class tradeoffs.' + concern)
