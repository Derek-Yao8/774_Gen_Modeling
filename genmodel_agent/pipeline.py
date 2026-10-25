"""One-command validation workflow with bounded, evidence-dependent replanning."""
import argparse
import json
from pathlib import Path

from . import cli, planning, execution, evaluation
from .planning import require, save
from . import diagnostics as training_diagnostics
from . import report_visuals


def screening(result):
    """Inspect actual screens, not the intentionally conservative validation label."""
    require(result.get('mode') == 'validation' and not result.get('test_accessed'),
            'Pipeline accepts validation evidence only')
    if result.get('status') != 'completed' or not result.get('models'):
        return 'incomplete'
    passed = True
    positive_utility = True
    has_utility = False
    for model in result['models']:
        if model.get('status') != 'computed' or not model.get('validity', {}).get('passed'):
            return 'incomplete'
        quality, utility = model['quality'], model['utility']
        representation_only = model.get('task') == 'representation_learning' and model['counts']['generated'] == 0
        if not representation_only:
            if quality.get('status') != 'computed':
                return 'incomplete'
            passed = passed and quality.get('quality_screen_passed') is True
        if utility.get('status') == 'computed':
            from .evidence_policy import summarize
            ev = summarize(model)
            has_utility = True
            positive_utility = positive_utility and utility.get('converged', False) and ev['utility_direction'] == 'positive'
            if model.get('task') == 'class_balancing':
                positive_utility = positive_utility and ev['resampling_control_direction'] == 'positive'
            passed = passed and utility.get('utility_screen_passed') is True
        elif utility.get('status') != 'not_applicable' or model.get('task') != 'data_generation':
            return 'incomplete'
    # This is a bounded revision signal, not an adoption verdict.
    return 'passed' if (positive_utility if has_utility else passed) else 'failed'


def feedback_record(plan, result, training=None, execution_dir=None):
    # Numerical evidence only. Do not send arrays, generated samples or AI prose.
    models = []
    for model in result['models']:
        fields = ('model', 'status', 'task', 'output_domain', 'validity', 'counts', 'quality', 'utility', 'recommendation', 'image_assessment')
        compact = {k: model[k] for k in fields if k in model}
        if 'utility' in compact:
            def trim(value):
                if isinstance(value, dict):
                    return {k:trim(v) for k,v in value.items() if 'confusion_matrix' not in k}
                return value
            compact['utility'] = trim(compact['utility'])
        compact['assessment'] = training_diagnostics.assessment(model)
        models.append(compact)
    diagnostics = []
    for model in (training or {}).get('models', []):
        keys = ('model', 'status', 'device', 'epochs_completed', 'best_epoch',
                'best_validation_negative_elbo', 'best_validation_nll', 'best_validation_feature_wasserstein', 'stop_reason', 'elapsed_seconds',
                'train_observations', 'validation_observations', 'generated_samples')
        measured = {k:model[k] for k in keys if k in model}
        measured['fit_diagnostics'] = training_diagnostics.summarize([], model['model'])
        if execution_dir is not None:
            path = Path(execution_dir)/model['model']/'history.json'
            if path.exists():
                rows = json.loads(path.read_text())
                measured['fit_diagnostics'] = training_diagnostics.summarize(rows, model['model'])
                # First, best and last five epochs summarize convergence without huge prompts.
                epochs = {1, model.get('best_epoch')} | {r['epoch'] for r in rows[-5:]}
                measured['selected_epoch_history'] = [r for r in rows if r['epoch'] in epochs]
        diagnostics.append(measured)
    return {'plan': plan, 'mode': 'validation', 'models': models, 'training_diagnostics': diagnostics}


def render(state, report_dir=None):
    lines = ['# Automated generative-modeling analysis', '',
             f"Workflow status: **{state['status']}**", '', state.get('reason', ''), '',
             f"Objective: {state['objective']}", '',
             'This workflow uses validation only. Final-test data remain reserved. '
             'Passing development screens is not approval for downstream use.', '',
             f"Training allowances committed: {state['committed_training_minutes']} of "
             f"{state['total_training_minutes']} minutes. Planning, evaluation and process "
             'startup/termination overhead are outside this training budget.', '',
             '[Dataset profile](inspection/profile.md)', '']
    for attempt in state['attempts']:
        n = attempt['round']
        lines += [f'## Attempt {n}', '', f"Planning: {attempt['planning_status']}", '',
                  f'[AI decisions and hyperparameters](round_{n}/planning/modeling_plan.md)', '']
        if attempt.get('explanation'):
            lines += [attempt['explanation'], '']
        if attempt.get('questions'):
            lines += ['Clarification required:'] + [f'- {q}' for q in attempt['questions']] + ['']
        if 'execution_status' in attempt:
            lines += [f"Training: {attempt['execution_status']}", '',
                      f'[Training artifacts](round_{n}/execution/execution_report.md)', '']
        if 'screening' in attempt:
            lines += [f"Exploratory validation screens: **{attempt['screening']}**", '',
                      f'[Measured results and interpretation](round_{n}/evaluation/evaluation_report.md)', '']
            for model in attempt.get('evidence', {}).get('models', []):
                lines += [f"### {model['model']}", '',
                          f"Recommendation: {model.get('recommendation', {}).get('decision', 'unavailable')}", '']
                lines += [f"Output validity: {model.get('validity', {}).get('passed', 'unavailable')}", '']
                fit = next((d.get('fit_diagnostics') for d in attempt['evidence'].get('training_diagnostics', []) if d['model'] == model['model']), None)
                if fit: lines += training_diagnostics.markdown(fit)
                lines += ['#### Sample-quality evidence', '', 'These screens are exploratory heuristics; a failed screen is not proof that data are unusable.', '']
                for name, metric in model.get('quality', {}).get('fidelity', {}).items():
                    lines.append(f"- {name}: discrepancy/reference ratio {metric.get('ratio')}; screen passed: {metric.get('screen_passed')}")
                utility = model.get('utility', {})
                lines += ['', '#### Downstream-use evidence', '', f"Evaluation status: {utility.get('status', 'unavailable')}", '']
                if utility.get('status') == 'computed':
                    lines.append(f"- Macro-F1 difference: {utility['macro_f1_delta']}; paired interval: {utility['paired_bootstrap_95_ci']}")
                lines += ['']
        if report_dir is not None:
            lines += report_visuals.markdown(report_dir, Path(report_dir)/f'round_{n}')
    lines += ['## Downstream recommendation', '',
              'Do not adopt generated outputs on the basis of this development run alone. '
              'Exploratory concerns can warrant model revision or stopping; incomplete evaluation '
              'cannot establish usefulness. Any final assessment requires a frozen protocol and '
              'the separate guarded test-evaluation command.', '',
              'See pipeline.json for the complete attempt ledger, budgets and stopping reason. '
              'Each attempt retains its own plans, checkpoints, generated data and evaluation records.', '']
    return '\n'.join(lines)


def run(dataset, objective, out, *, split=None, orientation=None, label_column=None,
        id_column=None, total_minutes=60, max_rounds=2, max_epochs=50, max_samples=5000,
        device='auto', max_memory_gib=6, pilot=False, ai=True, planner=None):
    require(planning.text(objective), 'Objective must not be blank')
    require(split in (None, 'train'), 'Workflow inspection must use training data; validation/test input splits are reserved')
    require(planning.number(total_minutes, 1, 1440, True), 'Training budget must be 1..1440 minutes')
    require(planning.number(max_rounds, 1, 3, True), 'Choose 1..3 rounds')
    require(planning.number(max_epochs, 1, 1000, True), 'Invalid epoch ceiling')
    require(planning.number(max_samples, 1, 100000, True), 'Invalid sample ceiling')
    require(device in ('auto','cpu','mps','cuda'), 'Invalid device')
    require(planning.number(max_memory_gib, 0.5, 64), 'Invalid memory ceiling')
    out = Path(out).resolve()
    require(not out.exists(), 'Choose a new pipeline directory; existing attempts are never overwritten')
    out.mkdir(parents=True)
    state = dict(schema_version='1.0', status='running', objective=objective,
                 dataset=str(Path(dataset).resolve()), test_accessed=False, pilot=pilot,
                 total_training_minutes=total_minutes, committed_training_minutes=0,
                 max_rounds=max_rounds, attempts=[])
    def checkpoint(status=None, reason=None):
        if status: state['status'] = status
        if reason: state['reason'] = reason
        save(out/'pipeline.json', state)
        (out/'generated_report.md').write_text(render(state, out), encoding='utf-8')
    checkpoint()
    history = []
    try:
        args = [str(dataset), '--objective', objective, '--out', str(out/'inspection')]
        for key, value in (('--split',split),('--orientation',orientation),('--label-column',label_column),('--id-column',id_column)):
            if value is not None: args += [key, value]
        if cli.main(args) != 0:
            checkpoint('input_error', 'Dataset inspection failed; see inspection/validation_error.json.')
            return state
        for n in range(1, max_rounds+1):
            remaining = total_minutes - state['committed_training_minutes']
            if remaining < 1:
                checkpoint('budget_exhausted', 'No training allowance remains. No downstream benefit established.')
                break
            # Reserve room for later rounds; charge selected hard ceilings, not optimistic runtime.
            allowance = max(1, int(remaining // (max_rounds-n+1)))
            folder = out/f'round_{n}'
            feedback = {'attempts':history, 'remaining_training_minutes':remaining,
                        'round_allowance_minutes':allowance} if history else None
            proposed = planning.run(out/'inspection/profile.json', folder/'planning', planner=planner,
                                   total_minutes=allowance, max_epochs=max_epochs, max_samples=max_samples,
                                   experiment_feedback=feedback)
            attempt = dict(round=n, allowance_minutes=allowance, planning_status=proposed['status'],
                           explanation=proposed.get('explanation', proposed.get('error','')),
                           questions=proposed.get('questions', []))
            state['attempts'].append(attempt)
            checkpoint()
            if proposed['status'] != 'ready':
                checkpoint(proposed['status'], proposed.get('explanation', proposed.get('error','Planning stopped.')))
                break
            plan = proposed['plan']
            committed = sum(m['max_minutes'] for m in plan['models'])
            require(committed <= remaining, 'Training budget exceeded')
            state['committed_training_minutes'] += committed
            attempt['committed_minutes'] = committed
            checkpoint()
            trained = execution.run(folder/'planning/modeling_plan.json', folder/'execution',
                                    device=device, pilot=pilot, max_memory_gib=max_memory_gib)
            attempt['execution_status'] = trained['status']
            checkpoint()
            if trained['status'] != 'completed':
                checkpoint('execution_failed', 'Training did not complete. No automatic scientific retry uses incomplete evidence.')
                break
            evaluated = evaluation.run(folder/'execution', folder/'evaluation', mode='validation', ai=ai)
            screen = screening(evaluated)
            attempt['screening'] = screen
            attempt['evidence'] = feedback_record(plan, evaluated, trained, folder/'execution')
            history.append(attempt['evidence'])
            checkpoint()
            if screen == 'incomplete':
                checkpoint('evaluation_incomplete', 'Required evaluation failed or is missing; inspect artifacts before another experiment.')
                break
            if pilot:
                checkpoint('pilot_completed', 'Pilot verified the workflow only. No scientific replanning or downstream recommendation is justified.')
                break
            if screen == 'passed':
                checkpoint('validation_passed', 'The revision stopping criterion was reached; this is not an adoption verdict. Review the evidence before freezing a separate final-test protocol.')
                break
            if n == max_rounds:
                checkpoint('round_limit_reached', 'Exploratory screens remained flagged at the attempt limit. Downstream usefulness has not been demonstrated; this is not proof of universal unusability.')
                break
    except Exception as exc:
        checkpoint('failed', f'{type(exc).__name__}: {exc}')
    except KeyboardInterrupt:
        checkpoint('interrupted', 'Interrupted; completed attempt artifacts are retained. Use a new directory for another run.')
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('train',))
    parser.add_argument('--orientation', choices=('rows','columns'))
    parser.add_argument('--label-column')
    parser.add_argument('--id-column')
    parser.add_argument('--total-minutes', type=int, default=60, help='Total committed training ceilings across rounds; excludes planning/evaluation')
    parser.add_argument('--max-rounds', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--max-samples', type=int, default=5000)
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    parser.add_argument('--max-memory-gib', type=float, default=6)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--no-ai-interpretation', action='store_true', help='Planner remains AI-driven; omit only AI evaluation narration')
    args = vars(parser.parse_args())
    args['ai'] = not args.pop('no_ai_interpretation')
    try:
        result = run(**args)
    except (ValueError,OSError) as exc:
        parser.exit(2, f'Pipeline refused: {exc}\n')
    print(f"{result['status']}: {args['out'].resolve() / 'generated_report.md'}")
    return 0 if result['status'] in ('validation_passed','pilot_completed','round_limit_reached','budget_exhausted','not_recommended') else 2


if __name__ == '__main__':
    raise SystemExit(main())
