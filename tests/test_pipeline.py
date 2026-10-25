from copy import deepcopy
import json

import pytest

from genmodel_agent import pipeline, planning
from genmodel_agent.planning import save
from test_execution import make_image_plan


class Planner:
    backend = 'test'
    def __init__(self, plans):
        self.plans = plans
        self.calls = 0
        self.prompts = []
    def propose(self, prompt, directory):
        self.prompts.append(prompt)
        plan = self.plans[min(self.calls, len(self.plans)-1)]
        self.calls += 1
        if plan is None:
            return dict(status='not_recommended', explanation='Measured failures do not justify another supported experiment.', questions=[], plan_json='')
        return dict(status='ready', explanation='Use measured evidence to select this candidate.', questions=[], plan_json=json.dumps(plan))


def evidence(passed=False, status='completed'):
    return dict(status=status, mode='validation', test_accessed=False, models=[dict(
        model='image_vae', status='computed', task='data_generation', validity={'passed': True},
        counts={'generated':100,'holdout':100}, quality={'status':'computed','quality_screen_passed':passed},
        utility={'status':'not_applicable'}, recommendation={'decision':'insufficient_evidence'})])


def fake_stages(monkeypatch, evaluations):
    calls=[]
    def train(path, out, **kwargs):
        calls.append(json.loads(path.read_text())['plan'])
        out.mkdir(parents=True)
        return {'status':'completed'}
    def evaluate(path, out, **kwargs):
        assert kwargs['mode']=='validation'
        out.mkdir(parents=True)
        return deepcopy(evaluations[min(len(calls)-1,len(evaluations)-1)])
    monkeypatch.setattr(pipeline.execution, 'run', train)
    monkeypatch.setattr(pipeline.evaluation, 'run', evaluate)
    return calls


def test_real_single_command_pipeline_preserves_test_sentinel(tmp_path):
    data, _, _, plan=make_image_plan(tmp_path)
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',
                        total_minutes=1,max_rounds=1,max_epochs=2,max_samples=6,
                        device='cpu',ai=False,planner=Planner([plan]))
    assert result['status']=='round_limit_reached'
    assert result['attempts'][0]['execution_status']=='completed'
    assert result['attempts'][0]['screening']=='failed'
    diagnostics=result['attempts'][0]['evidence']['training_diagnostics'][0]
    assert diagnostics['epochs_completed']==2
    assert diagnostics['selected_epoch_history'][0]['epoch']==1
    assert diagnostics['fit_diagnostics']['status']=='available'
    assert diagnostics['fit_diagnostics']['hypotheses']==[]
    assert 'Model-fit diagnostics' in (tmp_path/'run/generated_report.md').read_text()
    out=tmp_path/'run'
    evaluation=json.loads((out/'round_1/evaluation/evaluation_results.json').read_text())
    assert not evaluation['test_accessed']
    assert evaluation['models'][0]['fit_diagnostics']['status']=='available'
    assert 'assessment' in evaluation['models'][0]
    assert (out/'round_1/execution/image_vae/generated.npz').exists()
    assert (out/'generated_report.md').exists()
    with pytest.raises(ValueError):
        pipeline.run(data,'Generate images',out)


def test_failed_screens_trigger_changed_plan_with_same_split_and_budget(tmp_path,monkeypatch):
    data,_,_,plan=make_image_plan(tmp_path)
    revised=deepcopy(plan)
    revised['models'][0]['learning_rate']=0.0005
    ai=Planner([plan,revised])
    calls=fake_stages(monkeypatch,[evidence(),evidence(True)])
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',total_minutes=2,
                        max_epochs=2,max_samples=6,planner=ai,ai=False)
    assert result['status']=='validation_passed' and len(calls)==2
    assert result['committed_training_minutes']==2
    assert calls[0]['split']==calls[1]['split']
    assert 'quality_screen_passed' in ai.prompts[1]
    assert 'Prefer changing one or two scientific settings' in ai.prompts[1]
    assert 'assessment' in ai.prompts[1]
    assert (tmp_path/'run/round_2/planning/experiment_feedback.json').exists()


@pytest.mark.parametrize('change',['none','split','task'])
def test_invalid_revisions_never_train(tmp_path,monkeypatch,change):
    data,_,_,plan=make_image_plan(tmp_path)
    revised=deepcopy(plan)
    if change=='split': revised['split']['seed']=99
    if change=='task':
        revised['task']='class_balancing'
        revised['evaluation']['metrics'] += ['real_test_macro_f1','per_class_recall']
        revised['evaluation']['baseline']='real_only_classifier'
    ai=Planner([plan,revised])
    calls=fake_stages(monkeypatch,[evidence()])
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',total_minutes=2,
                        max_epochs=2,max_samples=6,planner=ai,ai=False)
    assert result['status']=='validation_failed'
    assert len(calls)==1 and ai.calls==3  # one bounded schema/scientific repair


@pytest.mark.parametrize('terminal',['not_recommended','incomplete','passed','pilot','budget'])
def test_terminal_conditions_do_not_launch_unneeded_training(tmp_path,monkeypatch,terminal):
    data,_,_,plan=make_image_plan(tmp_path)
    ai=Planner([None if terminal=='not_recommended' else plan])
    ev=evidence(passed=terminal=='passed',status='incomplete' if terminal=='incomplete' else 'completed')
    calls=fake_stages(monkeypatch,[ev])
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',total_minutes=1 if terminal=='budget' else 2,
                        max_epochs=2,max_samples=6,planner=ai,ai=False,pilot=terminal=='pilot')
    expected={'not_recommended':'not_recommended','incomplete':'evaluation_incomplete',
              'passed':'validation_passed','pilot':'pilot_completed','budget':'budget_exhausted'}
    assert result['status']==expected[terminal]
    assert len(calls)==(0 if terminal=='not_recommended' else 1)
    assert ai.calls==1


def test_planner_can_stop_after_measured_failure(tmp_path,monkeypatch):
    data,_,_,plan=make_image_plan(tmp_path)
    calls=fake_stages(monkeypatch,[evidence()])
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',total_minutes=2,
                        max_epochs=2,max_samples=6,planner=Planner([plan,None]),ai=False)
    assert result['status']=='not_recommended' and len(calls)==1
    assert 'Measured failures' in (tmp_path/'run/generated_report.md').read_text()


def test_failure_writes_report_and_stops(tmp_path,monkeypatch):
    data,_,_,plan=make_image_plan(tmp_path)
    def fail(*args,**kwargs): raise RuntimeError('worker failed')
    monkeypatch.setattr(pipeline.execution,'run',fail)
    result=pipeline.run(data,'Generate images',tmp_path/'run',split='train',
                        planner=Planner([plan]),max_epochs=2,max_samples=6)
    assert result['status']=='failed'
    assert 'worker failed' in (tmp_path/'run/generated_report.md').read_text()


@pytest.mark.parametrize('split',['val','test'])
def test_reserved_input_rejected_before_inspection(tmp_path,monkeypatch,split):
    def forbidden(*args,**kwargs):
        pytest.fail('Reserved input must not be inspected')
    monkeypatch.setattr(pipeline.cli,'main',forbidden)
    with pytest.raises(ValueError,match='training data'):
        pipeline.run(tmp_path/'input.npz','Generate images',tmp_path/'run',split=split)
    assert not (tmp_path/'run').exists()
