import json

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from genmodel_agent.cli import main
from genmodel_agent.data import load_dataset
from genmodel_agent.profile import profile_dataset


def test_images_keep_layout_labels_and_split(tmp_path):
    path = tmp_path / "images.npz"
    images = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
    np.savez(path, train_images=images, train_labels=np.array([0, 0, 1]),
             test_images=images[:1], test_labels=np.array([1]))
    data = load_dataset(path, split="train")
    np.testing.assert_array_equal(data.images, images)
    p = profile_dataset(data, "Generate labeled images")
    assert p["n_observations"] == 3
    assert p["labels"]["columns"][0]["majority_to_minority_ratio"] == 2
    assert p["value_semantics"]["raw_count_evidence"] == "not established"
    assert p["value_semantics"]["image_dtype"] == "uint8"
    assert p["value_semantics"]["pixel_range"] == [0.0, 35.0]
    from genmodel_agent.planning import compact_profile
    assert compact_profile(p)["value_semantics"]["pixel_range"] == [0.0, 35.0]
    assert p["objective"]["interpretation_status"] == "pending_planner"
    assert main([str(path), "--objective", "Generate", "--out", str(tmp_path / "bad")]) == 2


def test_sparse_counts_include_implicit_zeros(tmp_path):
    folder = tmp_path / "counts"
    folder.mkdir()
    mmwrite(folder / "matrix.mtx", sparse.coo_matrix([[0, 2, 0], [1, 0, 3]]))
    (folder / "genes.tsv").write_text("g1\tGene1\ng2\tGene2\n")
    (folder / "barcodes.tsv").write_text("c1\nc2\nc3\n")
    data = load_dataset(folder)
    p = profile_dataset(data, "Generate cells")
    assert sparse.issparse(data.X)
    assert p["n_observations"] == 3
    assert p["numeric_zero_fraction"] == 0.5
    assert p["missing_values"] == 0
    assert p["value_semantics"]["nonnegative_integer_matrix"]
    assert not p["labels"]["available"]


def test_cli_objective_missingness_and_outputs(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("x,category,label\n1,a,yes\n,b,no\n")
    out = tmp_path / "out"
    assert main([str(path), "--orientation", "rows", "--label-column", "label",
                 "--objective", "Impute missing values", "--out", str(out)]) == 0
    p = json.loads((out / "profile.json").read_text())
    assert p["missing_values"] == 1
    assert p["n_features"] == 2
    assert p["feature_types"]["categorical_or_text"] == 1
    assert "Impute missing values" in (out / "profile.md").read_text()
    assert main([str(path), "--objective", " ", "--orientation", "rows",
                 "--out", str(tmp_path / "empty")]) == 2
