from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import os

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite
from scipy.stats import nbinom
import torch

from genmodel_agent.data import InputError, load_dataset
from genmodel_agent.execution import run, supervise
from genmodel_agent.models import CountVAE, ImageVAE, build_model, nb_log_prob
from genmodel_agent.planning import CATALOG, EXAMPLE, save, validate
from genmodel_agent.profile import profile_dataset
from genmodel_agent.training_data import prepare


def make_image_plan(tmp_path, conditional=True):
    data_path = tmp_path / "images.npz"
    rng = np.random.default_rng(3)
    # Object-valued test arrays cannot be accessed with allow_pickle=False.
    # Successful training therefore proves they were not loaded.
    np.savez_compressed(data_path, train_images=rng.integers(0,256,(40,8,8,3),dtype=np.uint8),
                        train_labels=np.arange(40)%2, val_images=rng.integers(0,256,(12,8,8,3),dtype=np.uint8),
                        val_labels=np.arange(12)%2, test_images=np.array([object()],dtype=object),
                        test_labels=np.array([object()],dtype=object))
    profile = profile_dataset(load_dataset(data_path, split="train"), "Generate images")
    plan = deepcopy(EXAMPLE)
    plan["models"][0].update(conditional=conditional, latent_dim=4, hidden_dim=16,
                              batch_size=8, epochs=2, patience=2, max_minutes=1, generate_samples=6)
    plan["split"].update(strategy="official",train_fraction=0,validation_fraction=0,test_fraction=0)
    plan["excluded_models"] = [{"name": name,"reason":"Not selected"} for name in CATALOG if name != "image_vae"]
    folder = tmp_path / "plan"
    folder.mkdir()
    save(folder/"input_profile.json",profile)
    envelope = {"status":"ready","explanation":"Test plan","plan":plan,
                "budget":{"total_minutes":1,"max_epochs":2,"max_samples":6}}
    save(folder/"modeling_plan.json",envelope)
    return data_path, folder/"modeling_plan.json", profile, plan


def test_nb_matches_scipy_and_has_finite_gradients():
    x = torch.tensor([[0.,1.,12.],[4.,8.,2.]], dtype=torch.float64)
    mu = torch.tensor([[0.2,1.,10.],[3.,9.,2.]], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor([0.5,2.,5.], dtype=torch.float64, requires_grad=True)
    result = nb_log_prob(x, mu, theta)
    expected = nbinom.logpmf(x.numpy(),theta.detach().numpy(),theta.detach().numpy()/(theta.detach().numpy()+mu.detach().numpy()))
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=1e-10)
    (-result.sum()).backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(theta.grad).all()


@pytest.mark.parametrize("name",["image", "count"])
def test_conditional_models_train_and_shapes(name):
    torch.set_num_threads(2)
    if name == "image":
        model = ImageVAE((8,8,3),4,16,2)
        x = torch.rand(4,3,8,8)
    else:
        model = CountVAE(8,4,16,2)
        x = torch.ones(4,8)
    y = torch.tensor([0,1,0,1])
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, _, kl = model.loss(x,y)
    loss.backward()
    optimizer.step()
    assert kl >= 0 and torch.isfinite(loss)
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    with pytest.raises(ValueError): model.loss(x)


def test_executor_checkpoint_samples_and_test_isolation(tmp_path):
    data_path, plan_path, _, _ = make_image_plan(tmp_path)
    original = hashlib.sha256(data_path.read_bytes()).hexdigest()
    result = run(plan_path,tmp_path/"execution",device="cpu")
    assert result["status"] == "completed" and not result["test_used"]
    folder = tmp_path/"execution/image_vae"
    checkpoint = torch.load(folder/"best.pt",weights_only=True,map_location="cpu")
    model = build_model(checkpoint["config"],checkpoint["shape"],len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    samples = np.load(folder/"generated.npz",allow_pickle=False)
    assert samples["samples"].shape == (6,8,8,3)
    assert samples["samples"].dtype == np.uint8
    assert np.array_equal(np.bincount(samples["label_indices"]),[3,3])
    assert not result["quality_evaluation_complete"]
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == original
    assert (folder/"learning_curve.png").exists()


def test_changed_source_rejected(tmp_path):
    data_path, plan_path, _, _ = make_image_plan(tmp_path)
    # Valid NPZ but a different source from the one inspected.
    with np.load(data_path,allow_pickle=True) as archive:
        values = dict(archive)
    values["train_images"][0,0,0,0] ^= 1
    np.savez_compressed(data_path,**values)
    result = run(plan_path,tmp_path/"execution",device="cpu")
    assert result["status"] == "failed_or_partial"
    assert "hashes differ" in result["models"][0]["error"]
    assert not (tmp_path/"execution/image_vae/best.pt").exists()


def test_unavailable_trainer_is_rejected_before_launch(tmp_path):
    _, path, profile, plan = make_image_plan(tmp_path)
    plan["models"][0].update(name="continuous_flow",preprocessing="library_normalize_log1p_standardize",likelihood="continuous")
    response = {"status":"ready","explanation":"GAN", "questions":[],"plan_json":json.dumps(plan)}
    with pytest.raises(InputError,match="Trainer unavailable"):
        validate(response,profile,{"total_minutes":2,"max_epochs":2,"max_samples":6})


def write_counts(folder, matrix):
    folder.mkdir(exist_ok=True)
    mmwrite(folder/"matrix.mtx",sparse.coo_matrix(matrix.T))
    (folder/"genes.tsv").write_text("".join(f"g{i}\tGene{i}\n" for i in range(matrix.shape[1])))
    (folder/"barcodes.tsv").write_text("".join(f"c{i}\n" for i in range(len(matrix))))


def test_gene_selection_never_fits_validation_or_test(tmp_path):
    rng = np.random.default_rng(7)
    matrix = rng.poisson(3,(60,40))
    source = tmp_path/"counts"
    write_counts(source,matrix)
    p = profile_dataset(load_dataset(source),"Generate cells")
    plan = deepcopy(EXAMPLE)
    config = plan["models"][0]
    config.update(name="count_vae",selected_features=12,conditional=False)
    out1=tmp_path/"first"
    out1.mkdir()
    before = prepare(source,p,plan,config,out1)
    splits = np.load(out1/"splits.npz")
    train_ids, val_ids, test_ids = [splits[k] for k in ("train_indices","validation_indices","test_indices")]
    assert not (set(train_ids)&set(val_ids) or set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids))
    assert len(set(train_ids)|set(val_ids)|set(test_ids)) == len(matrix)
    matrix[np.r_[val_ids,test_ids],0] = 100000
    write_counts(source,matrix)
    p2 = profile_dataset(load_dataset(source),"Generate cells")
    out2=tmp_path/"second"
    out2.mkdir()
    after = prepare(source,p2,plan,config,out2)
    assert before.metadata["selected_feature_positions"] == after.metadata["selected_feature_positions"]
    np.testing.assert_array_equal(before.train.toarray(),after.train.toarray())
    np.testing.assert_array_equal(np.load(out1/"gene_selection.npz")["training_means"],np.load(out2/"gene_selection.npz")["training_means"])


def test_supervisor_terminates_stuck_worker(tmp_path):
    result = supervise([sys.executable,"-c","import time; time.sleep(20)"],str(tmp_path),os.environ.copy(),
                       tmp_path/"worker.log",wall_seconds=0.1,max_bytes=2**30)
    assert result["limit_triggered"] == "wall_time_limit"
    assert result["exit_code"] != 0
