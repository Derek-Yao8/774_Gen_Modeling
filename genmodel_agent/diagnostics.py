"""Descriptive training evidence; hypotheses are not sample-quality verdicts."""
import math

VERSION = '1.0'


def summarize(rows, model):
    keys = ('train_negative_elbo', 'validation_negative_elbo',
            'train_reconstruction_nll', 'validation_reconstruction_nll', 'train_kl', 'validation_kl',
            'train_nll', 'validation_nll', 'generator_loss', 'discriminator_loss',
            'validation_feature_wasserstein')
    series = {}
    invalid = []
    for key in keys:
        points = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                invalid.append(key)
                continue
            points.append((row['epoch'], float(value)))
        if points:
            tail = points[-min(5, len(points)):]
            series[key] = {'first': points[0][1], 'last': points[-1][1],
                           'minimum': min(v for _, v in points),
                           'minimum_epoch': min(points, key=lambda p:p[1])[0],
                           'recent_start_epoch': tail[0][0], 'recent_end_epoch':tail[-1][0],
                           'recent_change':tail[-1][1]-tail[0][1], 'observations':len(points)}
    notes = []
    hypotheses = []
    if model == 'image_gan':
        notes.append('Adversarial losses describe optimization, not sample quality. Their relative magnitudes do not diagnose discriminator dominance or mode collapse.')
    else:
        train, val = ('train_nll','validation_nll') if model.endswith('_flow') else ('train_negative_elbo','validation_negative_elbo')
        if train in series and val in series and len(rows) >= 5 and not invalid:
            t, v = series[train], series[val]
            if t['recent_change'] < 0 and v['recent_change'] > 0:
                hypotheses.append({'signal':'possible_overfitting', 'evidence':[train,val],
                    'interpretation':'Training objective decreased while validation objective increased over the recent window. This is compatible with overfitting, not proof of its cause.',
                    'candidate_response':'Consider a controlled capacity or optimization change; do not simply extend identical training.'})
            elif v['recent_change'] < 0 and v['minimum_epoch'] == rows[-1]['epoch']:
                hypotheses.append({'signal':'validation_still_improving', 'evidence':[val],
                    'interpretation':'The last epoch is the best recorded validation objective and the recent change is negative; convergence is not established.',
                    'candidate_response':'Consider additional epochs within budget only if sample-quality evidence supports it.'})
        if model.endswith('_flow'):
            notes.append('NLL is for standardized continuous log-dequantized expression. Compare only identical features and preprocessing; it is not count likelihood or ELBO.')
        else:
            notes.append('ELBO, reconstruction NLL and KL describe model fit. Low KL can motivate posterior-collapse investigation but no universal KL cutoff is used.')
    if len(rows) < 5:
        notes.append('Fewer than five epochs: no automatic trend hypothesis is assigned.')
    if invalid:
        notes.append('Nonfinite or nonnumeric history values require investigation; trend hypotheses are suppressed.')
    notes.append('Recent changes are descriptive, not significance tests. Improved model fit does not establish improved samples or downstream benefit.')
    return {'version':VERSION,'status':'available' if series else 'unavailable',
            'epochs_observed':len(rows),'series':series,'hypotheses':hypotheses,'limitations':notes,
            'invalid_metrics':sorted(set(invalid))}


def assessment(model):
    q, u = model.get('quality',{}), model.get('utility',{})
    return {'validity':model.get('validity',{}),
            'sample_quality':{'status':q.get('status','unavailable'),
                'heuristic_screen_passed':q.get('quality_screen_passed'),
                'interpretation':'Exploratory flags, not a calibrated scientific definition of usability.'},
            'downstream_utility':{'status':u.get('status','unavailable'),
                'macro_f1_delta':u.get('macro_f1_delta'),
                'paired_bootstrap_95_ci':u.get('paired_bootstrap_95_ci'),
                'output_domain':model.get('output_domain', 'recorded output domain'),
                'interpretation':'A task-specific comparison is required to demonstrate benefit; statistical similarity alone does not demonstrate it.'}}


def markdown(diagnostic):
    lines=['#### Model-fit diagnostics','',f"History: {diagnostic['status']}; epochs observed: {diagnostic['epochs_observed']}.",'']
    if diagnostic['series']:
        lines += ['| Diagnostic | First | Last | Minimum (epoch) | Recent change |', '|---|---:|---:|---:|---:|']
        for key,v in diagnostic['series'].items():
            lines.append(f"| {key} | {v['first']:.5g} | {v['last']:.5g} | {v['minimum']:.5g} ({v['minimum_epoch']}) | {v['recent_change']:.5g} |")
        lines += ['', 'Recent change compares the last two to five available observations; it is not a significance test. Minima of GAN losses are descriptive, not optimal-quality checkpoints.', '']
    for h in diagnostic['hypotheses']:
        lines += [f"- **{h['signal']}**: {h['interpretation']} {h['candidate_response']}"]
    lines += ['']+[f'- {note}' for note in diagnostic['limitations']]+['']
    return lines
