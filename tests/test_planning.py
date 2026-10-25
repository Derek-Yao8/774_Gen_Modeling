from copy import deepcopy
import json
from pathlib import Path

import pytest

from genmodel_agent.data import InputError
from genmodel_agent.planning import EXAMPLE, CATALOG, run, validate

BUDGET = dict(total_minutes=60, max_epochs=50, max_samples=5000)


def fixture():
    p = {"stage": "generative_input_inspection", "modality": "image", "image_shape": [28,28,3],
         "n_observations": 100, "n_features": 2352, "feature_types": {"numeric":2352},
         "missing_values": 0, "infinite_values": 0, "ambiguities": [], "warnings": [],
         "numeric_zero_fraction": 0, "value_semantics": {"raw_count_evidence":"not established"},
         "split_policy": {"inspected_split":"train"}, "compute": {},
         "constant_feature_positions": [], "objective": {"text":"Generate images"},
         "labels": {"available":True,"columns":[{"missing":0,"unique_nonmissing":2,"class_counts":[{"count":50},{"count":50}]}]}}
    plan = deepcopy(EXAMPLE)
    plan["split"].update(strategy="official",train_fraction=0,validation_fraction=0,test_fraction=0)
    plan["excluded_models"] = [{"name": n, "reason":"Not appropriate for this objective/data"} for n in CATALOG if n != "image_vae"]
    return p, plan


def response(plan):
    return dict(status="ready", explanation="Evidence-based plan", questions=[], plan_json=json.dumps(plan))


def test_valid_plan():
    p, plan = fixture()
    assert validate(response(plan), p, BUDGET) == plan


@pytest.mark.parametrize('mutation', ['test_split','count_model','budget','labels','metrics','preprocessing','extra','nan'])
def test_rejects_invalid_plans(mutation):
    p, plan = fixture()
    m = plan['models'][0]
    if mutation == 'test_split': p['split_policy']['inspected_split'] = 'test'
    if mutation == 'count_model': m['name'] = 'count_vae'
    if mutation == 'budget': m['max_minutes'] = 61
    if mutation == 'labels':
        p['labels']['columns'] = []
        m['conditional'] = True
    if mutation == 'metrics': plan['task'] = 'representation_learning'
    if mutation == 'preprocessing': plan['split']['fit_preprocessing_on'] = 'all'
    if mutation == 'extra': m['arbitrary_code'] = 'bad'
    if mutation == 'nan': m['learning_rate'] = float('nan')
    with pytest.raises(InputError): validate(response(plan), p, BUDGET)


def test_repair_saved_and_bounded(tmp_path):
    p, plan = fixture()
    source = tmp_path/'profile.json'
    source.write_text(json.dumps(p))
    class Fake:
        backend = 'test_fake'
        calls = 0
        def propose(self, prompt, directory):
            self.calls += 1
            candidate = deepcopy(plan)
            if self.calls == 1: candidate['models'][0]['epochs'] = 999
            return response(candidate)
    fake = Fake()
    result = run(source, tmp_path/'run', planner=fake)
    assert result['status'] == 'ready' and fake.calls == 2
    assert len(result['validation_history']) == 1
    assert result['execution_ready']
    assert (tmp_path/'run/modeling_plan.md').exists()


def test_clarification_has_no_plan():
    p,_ = fixture()
    assert validate(dict(status='needs_clarification',explanation='Unsupported objective',questions=['Choose a supported objective?'],plan_json=''),p,BUDGET) is None
