from genmodel_agent import diagnostics


def test_opposing_trends_are_hypothesis_not_quality_verdict():
    rows=[{'epoch':i,'train_nll':20-i,'validation_nll':10+i} for i in range(1,7)]
    d=diagnostics.summarize(rows,'realnvp_flow')
    assert d['hypotheses'][0]['signal']=='possible_overfitting'
    assert d['series']['validation_nll']['recent_start_epoch']==2
    assert d['series']['validation_nll']['recent_change']==4
    assert 'quality_screen_passed' not in d


def test_improving_validation_and_short_history():
    rows=[{'epoch':i,'train_negative_elbo':30-i,'validation_negative_elbo':40-i,'validation_kl':0.0} for i in range(1,7)]
    assert diagnostics.summarize(rows,'image_vae')['hypotheses'][0]['signal']=='validation_still_improving'
    assert diagnostics.summarize(rows[:2],'image_vae')['hypotheses']==[]
    assert not any(h['signal']=='posterior_collapse' for h in diagnostics.summarize(rows,'image_vae')['hypotheses'])


def test_gan_loss_scale_does_not_diagnose_collapse():
    d=diagnostics.summarize([{'epoch':i,'generator_loss':100*i,'discriminator_loss':0.0} for i in range(1,7)],'image_gan')
    assert not d['hypotheses']
    assert 'not sample quality' in ' '.join(d['limitations'])


def test_missing_and_nonfinite_history():
    assert diagnostics.summarize([],'count_vae')['status']=='unavailable'
    rows=[{'epoch':i,'train_nll':20-i,'validation_nll':10+i} for i in range(1,7)]
    rows[0]['train_nll']=float('nan')
    d=diagnostics.summarize(rows,'radial_flow')
    assert d['invalid_metrics']==['train_nll'] and not d['hypotheses']


def test_separate_evidence_does_not_upgrade_quality_to_utility():
    a=diagnostics.assessment({'validity':{'passed':True},'quality':{'status':'computed','quality_screen_passed':True},'utility':{'status':'not_applicable'}})
    assert a['downstream_utility']['status']=='not_applicable'
    assert a['downstream_utility']['macro_f1_delta'] is None
