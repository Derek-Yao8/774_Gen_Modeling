from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from genmodel_agent.models import ImageGAN,build_model
from genmodel_agent.gan_trainer import step,score_samples
from genmodel_agent.planning import validate,save,CATALOG
from genmodel_agent.execution import run
from genmodel_agent.evaluation import run as evaluate
from genmodel_agent.evaluation_contract import CONTRACT,DEFAULTS
from genmodel_agent.training_data import prepare
from test_execution import make_image_plan


def gan_plan(tmp_path):
    data,path,p,plan=make_image_plan(tmp_path)
    plan['models'][0].update(name='image_gan',preprocessing='scale_pixels_minus1_1',likelihood='implicit',learning_rate=.0002)
    plan['excluded_models']=[{'name':n,'reason':'Alternative not selected'} for n in CATALOG if n!='image_gan']
    plan['task']='class_balancing'
    plan['evaluation']['metrics']+=['real_test_macro_f1','per_class_recall']
    plan['evaluation']['baseline']='real_only_classifier'
    envelope=json.loads(path.read_text());envelope.update(plan=plan,evaluation_contract=CONTRACT)
    save(path,envelope)
    return data,path,p,plan,envelope


def test_gan_updates_and_gradient_isolation():
    torch.set_num_threads(2)
    model=ImageGAN((8,8,3),4,16,2)
    x=torch.rand(8,3,8,8)*2-1;y=torch.arange(8)%2;z=torch.randn(8,4)
    model.discriminator_loss(x,z,y).backward()
    assert all(p.grad is None for p in model.generator.parameters())
    assert any(p.grad is not None for p in model.discriminator.parameters())
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    g=torch.optim.Adam(model.generator.parameters(),lr=.0002,betas=(.5,.999))
    d=torch.optim.Adam(model.discriminator.parameters(),lr=.0002,betas=(.5,.999))
    losses=step(model,x,y,g,d)
    assert np.isfinite(losses).all()
    for prefix in ('generator','discriminator'):
        assert any(not torch.equal(before[k],v) for k,v in model.named_parameters() if k.startswith(prefix))
    assert all(p.grad is None for p in model.discriminator.parameters())
    assert model.decode(z,y).shape==(8,3,8,8)
    with pytest.raises(ValueError):model.decode(z)
    unconditional=ImageGAN((8,8),4,16)
    assert unconditional.decode(z).shape==(8,1,8,8)


def test_gan_planning_and_representation_rejection(tmp_path):
    _,_,p,plan,envelope=gan_plan(tmp_path)
    response={'status':'ready','explanation':'GAN alternative','questions':[],'plan_json':json.dumps(plan)}
    assert validate(response,p,envelope['budget'])['models'][0]['name']=='image_gan'
    plan['task']='representation_learning';plan['models'][0]['conditional']=False
    response['plan_json']=json.dumps(plan)
    with pytest.raises(ValueError,match='representation encoder'):validate(response,p,envelope['budget'])


def test_gan_training_evaluation_and_test_isolation(tmp_path):
    data,path,p,plan,envelope=gan_plan(tmp_path)
    prepared=tmp_path/'prepare';prepared.mkdir()
    bundle=prepare(data,p,plan,plan['models'][0],prepared)
    pixels,_=bundle.batch(np.array([0,1]))
    np.testing.assert_allclose(pixels,bundle.train[:2].transpose(0,3,1,2)/255*2-1,atol=1e-6)
    result=run(path,tmp_path/'training',device='cpu')
    assert result['status']=='completed' and not result['test_used']
    folder=tmp_path/'training/image_gan'
    checkpoint=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    assert set(checkpoint['optimizer_state_dict'])=={'generator','discriminator'}
    model=build_model(checkpoint['config'],checkpoint['shape'],len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'])
    with np.load(folder/'generated.npz') as arrays:
        assert arrays['samples'].shape==(6,8,8,3) and arrays['samples'].dtype==np.uint8
        assert 'decoder_means' not in arrays
        z=torch.from_numpy(np.random.default_rng(checkpoint['seed']+2000).standard_normal((6,4)).astype(np.float32))
        with torch.no_grad():
            predicted=model.decode(z,torch.from_numpy(arrays['label_indices'])).numpy().transpose(0,2,3,1)
        expected=np.rint(np.clip((predicted+1)/2,0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(arrays['samples'],expected)
        assert np.array_equal(np.bincount(arrays['label_indices']),[3,3])
    history=json.loads((folder/'history.json').read_text())
    assert 'train_negative_elbo' not in history[0]
    assert result['models'][0]['best_epoch']==min(history,key=lambda r:r['validation_feature_wasserstein'])['epoch']
    options={**DEFAULTS,'bootstrap':20,'reference_repeats':3,'max_train':40,'max_holdout':12,'metric_rows':12}
    e=evaluate(tmp_path/'training',tmp_path/'evaluation',options=options,ai=False)
    assert e['status']=='completed' and not e['test_accessed']
    assert e['models'][0]['utility']['status']=='computed'
    assert e['models'][0]['validity']['passed']


def test_planner_can_select_vae_and_gan_together(tmp_path):
    _,_,p,plan,envelope=gan_plan(tmp_path)
    vae=deepcopy(plan['models'][0])
    vae.update(name='image_vae',preprocessing='scale_pixels_0_1',likelihood='gaussian')
    plan['models'].append(vae)
    plan['excluded_models']=[{'name':n,'reason':'Incompatible or unavailable'} for n in CATALOG if n not in ('image_vae','image_gan')]
    response={'status':'ready','explanation':'Compare families','questions':[],'plan_json':json.dumps(plan)}
    budget={**envelope['budget'],'total_minutes':2}
    assert len(validate(response,p,budget)['models'])==2
