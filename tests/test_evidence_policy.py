from copy import deepcopy
import numpy as np
from scipy.stats import energy_distance
from genmodel_agent import evidence_policy as p
from genmodel_agent.image_assessment import energy, texture_features, fit_features, transform, visual_packet


def evidence():
    return {'status':'computed','task':'class_balancing','validity':{'passed':True},
            'quality':{'status':'computed','quality_screen_passed':False,'fidelity':{'x':{'screen_passed':False}}},
            'utility':{'status':'computed','converged':True,'paired_bootstrap_95_ci':[.001,.005],
                       'utility_screen_passed':False,'synthetic_vs_resampling_control':{'paired_bootstrap_95_ci':[.0001,.004]}}}


def test_task_benefit_not_universal_quality_gate():
    r=evidence()
    assert p.decision(r,False,'test',True)['decision']=='limited_use'
    assert p.decision(r,False,'validation',True)['decision']=='insufficient_evidence'
    assert p.decision(r,True,'test',True)['decision']=='insufficient_evidence'
    r['utility']['paired_bootstrap_95_ci']=[-.03,-.01]
    assert p.decision(r,False,'test',True)['decision']=='do_not_use'
    r['utility']['paired_bootstrap_95_ci']=[-.01,.02]
    assert p.decision(r,False,'test',True)['decision']=='insufficient_evidence'


def test_similarity_does_not_establish_pbmc_utility():
    r=evidence();r.update(task='data_generation',output_domain='log_dequantized_expression')
    r['quality']['quality_screen_passed']=True;r['utility']={'status':'not_applicable'}
    assert p.decision(r,False,'test',True)['decision']=='insufficient_evidence'
    r['validity']['passed']=False
    assert p.decision(r,False,'test',True)['decision']=='do_not_use'


def test_incomplete_control_and_invalid_interval_never_pass():
    r=evidence();r['utility']['synthetic_vs_resampling_control']={}
    assert p.decision(r,False,'test',True)['decision']=='insufficient_evidence'
    assert p.interval_direction([float('nan'),1])=='unavailable'
    r=evidence();r['quality']['status']='failed'
    assert p.decision(r,False,'test',True)['decision']=='insufficient_evidence'


def test_energy_formula_and_joint_dependence():
    x=np.array([[0.],[1.],[2.]]);y=np.array([[1.],[4.]])
    assert np.isclose(energy(x,y),energy_distance(x[:,0],y[:,0])**2)
    a=np.array([[-1.,-1.],[1.,1.]]);b=np.array([[-1.,1.],[1.,-1.]])
    assert energy(a,a)==0 and energy(a,b)>0 # equal marginals, different joint structure


def test_texture_preserves_detail_and_training_fit_immutable():
    rng=np.random.default_rng(11);train=rng.integers(0,256,(40,8,8,3),dtype=np.uint8)
    args=fit_features(train,42);before=args[0].components_.copy()
    transform(np.zeros_like(train),*args)
    assert np.array_equal(before,args[0].components_)
    assert not np.allclose(texture_features(train),texture_features(np.full_like(train,127)))


def test_visual_packet_deterministic_by_class_without_fake_review(tmp_path):
    rng=np.random.default_rng(3);x=rng.integers(0,256,(12,8,8,3),dtype=np.uint8);labels=np.arange(12)%2
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    r=visual_packet(a,x,x,labels,labels,np.arange(12),np.arange(12),42)
    s=visual_packet(b,x,x,labels,labels,np.arange(12),np.arange(12),42)
    assert r==s and r['review_status']=='not_yet_reviewed'
    for entry in r['samples']:
        assert all(labels[i]==int(entry['class']) for i in entry['original_indices'])


def test_pipeline_stopping_is_not_all_quality_flags_gate():
    from genmodel_agent.pipeline import screening
    r=evidence();r['model']='image_gan';r['counts']={'generated':100,'holdout':100}
    result={'status':'completed','mode':'validation','test_accessed':False,'models':[r]}
    assert screening(result)=='passed'  # utility supports stopping despite quality concerns
    r['utility']['paired_bootstrap_95_ci']=[-.1,.1]
    assert screening(result)=='failed'
