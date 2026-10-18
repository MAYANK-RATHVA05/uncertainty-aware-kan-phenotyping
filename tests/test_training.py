from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kan_phenotyping.experiments.common import limit_split, validate_saved_split
from kan_phenotyping.data.splitting import group_split
from kan_phenotyping.training.trainer import (
    FeatureDataset, Trainer, TrainingConfig, make_loader, seed_everything,
)


def features(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    return SimpleNamespace(tabular=x, temporal=x[:, None, :], static=np.empty((len(x), 0), np.float32))


def test_trainer_restores_best_validation_checkpoint(tmp_path):
    seed_everything(12, num_threads=1)
    x = np.linspace(-1, 1, 32, dtype=np.float32)
    config = TrainingConfig(max_epochs=12, batch_size=32, learning_rate=0.1, weight_decay=0,
                            patience=2, num_threads=1)
    train = make_loader(FeatureDataset(features(x), 2 * x), config, shuffle=True)
    validation = make_loader(FeatureDataset(features(x), -2 * x), config, shuffle=False)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    trainer = Trainer(model, config, input_kind="tabular")
    result = trainer.fit(train, validation, tmp_path, model_metadata={"name": "test_linear"})
    assert result.best_epoch == 1
    assert len(result.history) == 3  # two worsening epochs trigger stopping
    checkpoint = torch.load(result.checkpoint, weights_only=True)
    assert checkpoint["selection_metric"] == "rmse"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model_state"][key])
    assert trainer.evaluate(validation)["rmse"] == pytest.approx(result.best_score)
    assert (tmp_path / "history.json").is_file()
    with pytest.raises(FileExistsError):
        trainer.fit(train, validation, tmp_path)


def test_training_is_seed_reproducible(tmp_path):
    x = np.linspace(-1, 1, 20, dtype=np.float32)
    outputs = []
    for run in range(2):
        config = TrainingConfig(seed=7, max_epochs=3, batch_size=6, num_threads=1)
        seed_everything(config.seed, config.num_threads)
        dataset = FeatureDataset(features(x), x * 3)
        model = nn.Linear(1, 1)
        trainer = Trainer(model, config, input_kind="tabular")
        trainer.fit(make_loader(dataset, config, shuffle=True), make_loader(dataset, config, shuffle=False),
                    tmp_path / str(run))
        outputs.append(trainer.predict(make_loader(dataset, config, shuffle=False))[1])
    np.testing.assert_array_equal(*outputs)


def test_group_preserving_smoke_subset_and_persistence(tmp_path):
    groups = np.repeat(np.arange(100), 4)
    full = group_split(groups, seed=42)
    first = limit_split(full, groups, 80, seed=42)
    second = limit_split(full, groups, 80, seed=42)
    validate_saved_split(first, groups)
    for name in ("train", "validation", "test"):
        part = getattr(first, name)
        np.testing.assert_array_equal(part, getattr(second, name))
        assert set(part).issubset(set(getattr(full, name)))
        for group in np.unique(groups[part]):
            assert np.count_nonzero(groups[part] == group) == 4
    path = tmp_path / "splits.npz"
    np.savez(path, train=first.train, validation=first.validation, test=first.test)
    with np.load(path, allow_pickle=False) as restored:
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(restored[name], getattr(first, name))


@pytest.mark.parametrize("settings", [{"max_epochs": 0}, {"batch_size": 0}, {"patience": 0},
                                      {"learning_rate": -1}, {"regression_loss": "focal"}])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


def test_feature_dataset_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        FeatureDataset(features([np.nan]), np.array([0.0]))


def test_binary_trainer_never_converts_continuous_labels(tmp_path):
    config = TrainingConfig(task="binary", max_epochs=1, num_threads=1)
    dataset = FeatureDataset(features([0, 1]), np.array([0.1, 0.2]))
    trainer = Trainer(nn.Linear(1, 1), config, input_kind="tabular")
    loader = make_loader(dataset, config, shuffle=False)
    with pytest.raises(ValueError, match="only 0 and 1"):
        trainer.fit(loader, loader, tmp_path)
