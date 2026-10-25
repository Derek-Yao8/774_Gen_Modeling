from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from genmodel_agent.flows import Planar,Radial,Coupling,ExpressionFlow,log_dequantize,FLOW_NAMES
from genmodel_agent.capabilities import FLOW_EVALUATION
from genmodel_agent.data import load_dataset
from genmodel_agent.profile import profile_dataset
from genmodel_agent.planning import EXAMPLE,CATALOG,validate,save
from genmodel_agent.execution import run
from genmodel_agent.evaluation import run as evaluate
from genmodel_agent.evaluation_contract import CONTRACT,DEFAULTS
from genmodel_agent.evaluation_metrics import quality
from genmodel_agent.flow_trainer import prepare_flow
from test_execution import write_counts


@pytest.mark.parametrize('name',FLOW_NAMES)
def test_flow_jacobian_inverse_and_gradients(name):
    torch.manual_seed(2)
    layer={'planar_flow':lambda:Planar(4),'radial_flow':lambda:Radial(4),'realnvp_flow':lambda:Coupling(4,16,0)}[name]().double()
    if name=='realnvp_flow':
        with torch.no_grad():layer.net[-1].weight.normal_(0,.05)
    x=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    y,ld=layer(x)
    for i in range(len(x)):
        jac=torch.autograd.functional.jacobian(lambda a:layer(a[None])[0][0],x[i])
        torch.testing.assert_close(ld[i],torch.linalg.slogdet(jac)[1],rtol=1e-7,atol=1e-7)
    with torch.no_grad():torch.testing.assert_close(layer.inverse(y),x,atol=1e-7,rtol=1e-7)
    (y.square().sum()-ld.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    config={'name':name,'selected_features':4,'hidden_dim':16,'flow_layers':4}
    model=ExpressionFlow(config).double()
    sample=torch.randn(5,4,dtype=torch.float64)
    z,_=model(sample)
    with torch.no_grad():torch.testing.assert_close(model.decode(z),sample,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(model.log_prob(sample)).all()


def fixture(tmp_path,name):
    source=tmp_path/'cells';rng=np.random.default_rng(5)
    matrix=rng.poisson(2,(80,20))
    write_counts(source,matrix)
    profile=profile_dataset(load_dataset(source),'Generate continuous log-expression for exploratory simulation')
    plan=deepcopy(EXAMPLE)
    plan['models'][0].update(name=name,preprocessing='log_dequantize_train_standardize',likelihood='continuous',
                            conditional=False,selected_features=4,latent_dim=4,hidden_dim=16,flow_layers=3,
                            epochs=2,patience=2,batch_size=16,max_minutes=1,generate_samples=16)
    plan['excluded_models']=[{'name':n,'reason':'Alternative not selected'} for n in CATALOG if n!=name]
    folder=tmp_path/'plan';folder.mkdir()
    envelope={'status':'ready','explanation':'Flow verification','plan':plan,'catalog_version':2,
              'budget':{'total_minutes':1,'max_epochs':2,'max_samples':16},'evaluation_contract':CONTRACT,
              'flow_evaluation_contract':FLOW_EVALUATION}
    save(folder/'input_profile.json',profile);save(folder/'modeling_plan.json',envelope)
    return source,profile,plan,envelope,folder/'modeling_plan.json',matrix


@pytest.mark.parametrize('name',FLOW_NAMES)
def test_flow_training_and_evaluation(tmp_path,name):
    source,profile,plan,envelope,path,_=fixture(tmp_path,name)
    response={'status':'ready','explanation':'Verification','questions':[],'plan_json':json.dumps(plan)}
    validate(response,profile,envelope['budget'])
    result=run(path,tmp_path/'training',device='cpu')
    assert result['status']=='completed' and not result['test_used']
    folder=tmp_path/'training'/name
    checkpoint=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    model=ExpressionFlow(checkpoint['config']);model.load_state_dict(checkpoint['state_dict'])
    with np.load(folder/'generated.npz') as arrays:
        assert arrays['samples'].shape==(16,4) and np.isfinite(arrays['samples']).all()
        assert str(arrays['output_domain'])=='log_dequantized_expression'
    opts={**DEFAULTS,'bootstrap':20,'reference_repeats':3,'metric_rows':12,'max_holdout':12}
    e=evaluate(tmp_path/'training',tmp_path/'evaluation',options=opts,ai=False)
    assert e['status']=='completed' and not e['test_accessed']
    m=e['models'][0]
    assert m['output_domain']=='log_dequantized_expression' and m['validity']['passed']
    assert set(m['quality']['fidelity'])=={'continuous_feature_wasserstein','continuous_correlation_mae'}
    assert m['utility']['status']=='not_applicable'


def test_flow_dequantization_train_only_and_subset_stability(tmp_path):
    source,profile,plan,envelope,path,matrix=fixture(tmp_path,'realnvp_flow')
    job={'dataset':str(source),'profile':profile,'plan':plan,'config':plan['models'][0],'pilot':False,'max_bytes':2**30}
    a=tmp_path/'a';a.mkdir();before=prepare_flow(job,a)
    with np.load(a/'splits.npz') as split:
        test=split['test_indices'];val=split['validation_indices'];train=split['train_indices']
    matrix[np.r_[val,test],0]=10000
    write_counts(source,matrix);job['profile']=profile_dataset(load_dataset(source),'Generate')
    b=tmp_path/'b';b.mkdir();after=prepare_flow(job,b)
    np.testing.assert_array_equal(before.train,after.train)
    assert before.metadata['flow_mean']==after.metadata['flow_mean']
    assert before.metadata['selected_feature_positions']==after.metadata['selected_feature_positions']
    x=np.zeros((3,4));idx=np.array([4,6,9])
    np.testing.assert_array_equal(log_dequantize(x,idx,42)[[2,0]],log_dequantize(x[[2,0]],idx[[2,0]],42))


def test_flow_rejects_dimension_reduction_and_class_balancing(tmp_path):
    _,p,plan,envelope,_,_=fixture(tmp_path,'planar_flow')
    plan['models'][0]['latent_dim']=2
    response={'status':'ready','explanation':'Invalid','questions':[],'plan_json':json.dumps(plan)}
    with pytest.raises(ValueError,match='dimension'):validate(response,p,envelope['budget'])
    plan['models'][0]['latent_dim']=4;plan['task']='class_balancing'
    response['plan_json']=json.dumps(plan)
    with pytest.raises(ValueError,match='unconditional'):validate(response,p,envelope['budget'])
