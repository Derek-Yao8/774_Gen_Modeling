from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from genmodel_agent.data import InputError, load_dataset
from genmodel_agent.execution import run as train
from genmodel_agent.profile import profile_dataset
from genmodel_agent.planning import save
from genmodel_agent.evaluation import run, decision, interpret
from genmodel_agent.evaluation_contract import CONTRACT, DEFAULTS
from genmodel_agent.evaluation_metrics import validity, quality, image_features, paired_scores
from test_execution import make_image_plan

OPTS={**DEFAULTS,'bootstrap':20,'reference_repeats':3,'max_train':40,'max_holdout':12,'metric_rows':12}


@pytest.fixture(scope='module')
def saved_execution(tmp_path_factory):
    root=tmp_path_factory.mktemp('evaluation_integration')
    data,path,_,plan=make_image_plan(root)
    # Replace test sentinel with genuine test fixtures before profiling/training.
    with np.load(data,allow_pickle=True) as archive: values=dict(archive)
    rng=np.random.default_rng(99)
    values.update(test_images=rng.integers(0,256,(12,8,8,3),dtype=np.uint8),test_labels=np.arange(12)%2)
    np.savez_compressed(data,**values)
    profile=profile_dataset(load_dataset(data,split='train'),'Augment labeled images')
    save(path.parent/'input_profile.json',profile)
    envelope=json.loads(path.read_text())
    envelope['evaluation_contract']=CONTRACT
    plan['task']='class_balancing'
    plan['evaluation']['metrics'] += ['real_test_macro_f1','per_class_recall']
    plan['evaluation']['baseline']='real_only_classifier'
    envelope['plan']=plan
    save(path,envelope)
    result=train(path,root/'training',device='cpu')
    assert result['status']=='completed'
    return root/'training'


def test_validity_rejects_nonfinite_counts_and_wrong_gene_order():
    r=validity(np.array([[1.,np.nan],[-1.,2.]]),[2],'counts',2,feature_ids=['b','a'],expected_ids=['a','b'])
    assert not r['passed'] and len(r['errors'])>=2


def test_copying_detected_and_distribution_shift_flagged():
    rng=np.random.default_rng(4)
    train=rng.integers(80,150,(40,8,8,3),dtype=np.uint8)
    held=rng.integers(80,150,(20,8,8,3),dtype=np.uint8)
    scaler=StandardScaler().fit(image_features(train))
    copied,_=quality(train,held,train[:12],'image',scaler,OPTS)
    assert copied['overlap']['synthetic_exact_duplicate_fraction']==1
    shifted,_=quality(train,held,np.full((12,8,8,3),250,dtype=np.uint8),'image',scaler,OPTS)
    assert not shifted['quality_screen_passed']
    assert shifted['fidelity']['feature_wasserstein']['ratio']>2


def test_paired_comparison_uses_same_observations():
    y=np.tile([0,1],50)
    perfect=y.copy()
    poor=1-y
    r=paired_scores(y,poor,perfect,np.array([0,1]),20,42)
    assert r['macro_f1_delta']==1 and r['paired_bootstrap_95_ci']==[1.,1.]
    same=paired_scores(y,perfect,perfect,np.array([0,1]),20,42)
    assert same['macro_f1_delta']==0 and not same['utility_screen_passed']


def test_validation_never_loads_test_values(saved_execution,tmp_path,monkeypatch):
    import genmodel_agent.evaluation_data as module
    loader=module.load_dataset
    calls=[]
    def guard(*args,**kwargs):
        calls.append(kwargs.get('split'))
        assert kwargs.get('split')!='test'
        return loader(*args,**kwargs)
    monkeypatch.setattr(module,'load_dataset',guard)
    r=run(saved_execution,tmp_path/'validation',options=OPTS,ai=False)
    assert r['status']=='completed' and not r['test_accessed']
    assert 'val' in calls
    assert r['models'][0]['utility']['metric_name']=='validation_macro_f1_proxy'
    assert r['models'][0]['recommendation']['decision']=='insufficient_evidence'


def test_freeze_final_and_duplicate_test_access(saved_execution,tmp_path):
    frozen=tmp_path/'frozen.json'
    run(saved_execution,tmp_path/'validation',options=OPTS,freeze=frozen,ai=False)
    bad={**OPTS,'seed':13}
    with pytest.raises(InputError,match='Frozen protocol mismatch'):
        run(saved_execution,tmp_path/'mismatch',options=bad,mode='test',protocol=frozen,ai=False)
    result=run(saved_execution,tmp_path/'final',options=OPTS,mode='test',protocol=frozen,ai=False)
    assert result['status']=='completed' and result['test_accessed']
    assert result['models'][0]['utility']['metric_name']=='real_test_macro_f1'
    with pytest.raises(FileExistsError):
        run(saved_execution,tmp_path/'repeat',options=OPTS,mode='test',protocol=frozen,ai=False)


def test_ai_cannot_upgrade_pilot_decision(tmp_path):
    r={'model':'image_vae','status':'computed','recommendation':{'decision':'insufficient_evidence'}}
    class Fake:
        def propose(self,prompt,directory):
            return {'status':'ready','questions':[], 'plan_json':json.dumps({'image_vae':{
                'decision':'limited_use','interpretation':'Looks good','evidence_keys':['status'],'limitations':['Pilot']}})}
    ai=interpret([r],tmp_path,Fake())
    assert ai['status']=='unavailable_or_rejected'
    assert 'override' in ai['error']


def test_empty_generation_is_allowed_for_representation():
    a=np.empty((0,8,8,3),dtype=np.uint8)
    assert validity(a,[8,8,3],'image',0)['passed']
    train=np.ones((10,8,8,3),dtype=np.uint8)
    q,_=quality(train,train,a,'image',None,OPTS)
    assert q['status']=='not_applicable_no_generation'


def test_new_plan_records_executable_evaluation_contract():
    from genmodel_agent.planning import prompt_for
    from test_planning import fixture, BUDGET
    profile,_=fixture()
    prompt=prompt_for(profile,BUDGET,None)
    assert 'evaluation_contract' in prompt and 'Wasserstein' in prompt


def test_missing_artifact_yields_incomplete_report(saved_execution,tmp_path):
    import shutil
    copy=tmp_path/'copy'
    shutil.copytree(saved_execution,copy)
    (copy/'image_vae/generated.npz').rename(copy/'image_vae/generated.missing')
    result=run(copy,tmp_path/'evaluation',options=OPTS,ai=False)
    assert result['status']=='incomplete'
    assert result['models'][0]['recommendation']['decision']=='insufficient_evidence'
    assert (tmp_path/'evaluation/evaluation_report.md').exists()


def test_pilot_final_test_is_refused_before_data_access(saved_execution,tmp_path):
    import shutil
    copy=tmp_path/'copy'
    shutil.copytree(saved_execution,copy)
    status=json.loads((copy/'execution.json').read_text())
    status['pilot']=True
    save(copy/'execution.json',status)
    with pytest.raises(InputError,match='Pilot runs cannot'):
        run(copy,tmp_path/'final',mode='test',options=OPTS,ai=False)
    assert not (tmp_path/'final').exists()


def test_ai_evidence_paths_must_exist(tmp_path):
    from genmodel_agent.evaluation import evidence_exists
    assert evidence_exists({'utility':{'macro_f1_delta':-.1}},'utility.macro_f1_delta')
    assert not evidence_exists({'utility':{'macro_f1_delta':-.1}},'utility.test_accuracy')
    assert not evidence_exists({'quality':{}},'__class__')
    result={'model':'image_vae','status':'computed','recommendation':{'decision':'insufficient_evidence'}}
    class Fake:
        def propose(self,prompt,directory):
            return {'status':'ready','questions':[], 'plan_json':json.dumps({'image_vae':{
                'decision':'insufficient_evidence','interpretation':'Pilot evidence only',
                'evidence_keys':['recommendation.decision'],'limitations':['Pilot']}})}
    assert interpret([result],tmp_path,Fake())['status']=='completed'
