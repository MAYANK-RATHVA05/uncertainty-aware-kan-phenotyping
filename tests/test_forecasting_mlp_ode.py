"""Synthetic tests for the fixed dynamics-only MLP replacement."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn

from kan_phenotyping.experiments import forecasting_mlp_ode as experiment
from kan_phenotyping.experiments.forecasting_baselines import BaselinePartition, SavedBaselineData
from kan_phenotyping.experiments.forecasting_sequential_kan import prepare_sequential_data
from kan_phenotyping.models.latent_kan_ode import LatentKANODE
from kan_phenotyping.models.latent_mlp_ode import LatentMLPODE
from kan_phenotyping.models.sequential_kan import KANEncodedRecurrentBaseline
from kan_phenotyping.training.trainer import seed_everything


@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(42, 1)


class FrozenPreprocessor:
    def __init__(self):
        w, d = [f"w{i}" for i in range(48)], [f"d{i}" for i in range(62)]
        self.feature_names = {"wearable": w, "demographics": d, "sm0_phq": ["measured_sm0_phq9_standardized"],
            "tabular": [f"SM{m}__{n}" for m in (1, 2) for n in w]
                + [f"baseline_demographic__{n}" for n in d] + ["measured_sm0_phq9_standardized"]}

    def export_metadata(self):
        return {"fitted_partition": "training_only"}

    def fit(self, *a, **k):
        raise AssertionError("No fitting allowed")

    def transform(self, *a, **k):
        raise AssertionError("No transformation allowed")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(13)
    def part(n, offset):
        x = rng.normal(0, .3, (n, 159)).astype(np.float32).astype(np.float64)
        return BaselinePartition(np.arange(offset, offset+n), np.array([f"synthetic-{offset}-{i//2}" for i in range(n)]),
                                 np.ones(n), x, (2+np.arange(n)%4).astype(float))
    preprocessor = FrozenPreprocessor()
    source = SavedBaselineData(part(24, 0), part(8, 100), preprocessor,
        {"preprocessing": preprocessor.export_metadata()}, tmp_path, (), "synthetic")
    return prepare_sequential_data(source)


@pytest.fixture
def initial_path(tmp_path):
    seed_everything(42, 1)
    model = LatentKANODE()
    path = tmp_path / "kan" / "seed_42" / "initial.pt"
    path.parent.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "initialization_seed": 42, "model_metadata": experiment.kan_spec()}, path)
    return path


def test_exact_counts_and_only_dynamics_replaced():
    model = LatentMLPODE()
    assert sum(p.numel() for p in model.dynamics.parameters()) == 3085
    assert sum(p.numel() for p in model.parameters()) == 26254
    assert model.dynamics[0].in_features == model.dynamics[2].out_features == 16
    assert model.dynamics[0].out_features == model.dynamics[2].in_features == 93
    assert isinstance(model.dynamics[1], nn.Tanh)
    assert model.step_size == .25
    assert LatentMLPODE.forward is LatentKANODE.forward
    assert LatentMLPODE.trace_states is LatentKANODE.trace_states


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_shared_initialization_and_standard_mlp_scaling(seed):
    seed_everything(seed, 1)
    shared = KANEncodedRecurrentBaseline()
    ordinary = nn.Sequential(nn.Linear(16, 93), nn.Tanh(), nn.Linear(93, 16))
    seed_everything(seed, 1)
    model = LatentMLPODE()
    for name, value in shared.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for name in ("weight", "bias"):
        torch.testing.assert_close(getattr(model.dynamics[0], name), getattr(ordinary[0], name), rtol=0, atol=0)
    torch.testing.assert_close(model.dynamics[2].weight, ordinary[2].weight*.01, rtol=0, atol=0)
    assert torch.count_nonzero(model.dynamics[2].bias) == 0


def test_zero_dynamics_gradients_order_prefix_and_state_isolation(data):
    checks = experiment.architecture_checks(LatentMLPODE(), experiment.diagnostic_partition(data.train))
    assert checks["zero_dynamics_matches_recurrent"] and checks["no_state_carry"]
    assert len(checks["finite_nonzero_gradients"]) == 17
    assert all(checks["finite_nonzero_gradients"].values())
    assert checks["prefix"]["sm1_gradient_wrt_sm2_exactly_zero"]


def test_flow_uses_shared_encoders_and_requested_interval_schedule():
    model = LatentMLPODE()
    w, c = torch.randn(5, 2, 48), torch.randn(5, 63)
    packets, contexts, intervals = [], [], []
    handles = [model.static_encoder.register_forward_pre_hook(lambda m,a: contexts.append(a[0].clone())),
        model.wearable_encoder.register_forward_pre_hook(lambda m,a: packets.append((id(m),a[0].clone())))]
    original = model.integrator.integrate
    def observe(f, state, t0, t1):
        intervals.append((id(f),t0,t1))
        return original(f,state,t0,t1)
    model.integrator.integrate = observe
    try:
        states = model.trace_states(w,c)
    finally:
        for h in handles:h.remove()
    assert states.shape == (5,6,16) and len(contexts)==1
    assert intervals == [(id(model.dynamics),0.,1.),(id(model.dynamics),1.,2.),(id(model.dynamics),2.,3.)]
    assert [p[0] for p in packets] == [id(model.wearable_encoder)]*2
    for i in range(2):torch.testing.assert_close(packets[i][1],w[:,i],rtol=0,atol=0)


def test_initial_matching_and_derivatives_do_not_modify_weights_or_rng(data, initial_path):
    seed_everything(42, 1)
    model = LatentMLPODE()
    before = deepcopy(model.state_dict())
    matching = experiment.initialization_check(model, initial_path, 42)
    rng = torch.random.get_rng_state().clone()
    result = experiment.derivative_diagnostic(model, experiment.diagnostic_partition(data.train), initial_path, 42)
    assert matching["shared_state_tensors_exact"] == 16
    assert not result["function_space_initialization_matched"]
    assert result["statistics"]["difference_coordinate_rms"] > 0
    assert result["statistics"]["mlp"]["coordinate_rms"] > 0
    torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
    for name,value in before.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)


@pytest.mark.parametrize("mutation", ["seed","shared_weight","dtype","metadata"])
def test_initial_mismatch_rejected(initial_path, mutation):
    state = torch.load(initial_path,weights_only=True)
    if mutation=="seed":state["initialization_seed"]=43
    if mutation=="shared_weight":state["model_state"]["head.bias"] += 1
    if mutation=="dtype":state["model_state"]["head.bias"]=state["model_state"]["head.bias"].double()
    if mutation=="metadata":state["model_metadata"]={}
    torch.save(state,initial_path)
    seed_everything(42,1)
    with pytest.raises(ValueError):experiment.initialization_check(LatentMLPODE(),initial_path,42)


def test_trained_checkpoint_path_rejected_before_open(tmp_path,monkeypatch):
    monkeypatch.setattr(torch,"load",lambda *a,**k:pytest.fail("Must not read trained reference"))
    with pytest.raises(ValueError):experiment.initial_artifact(tmp_path/"seed_42/best.pt",42)


@pytest.mark.parametrize("seed",[42,43,44])
def test_fixed_training_configuration(seed):
    c=experiment.experiment_config(seed)
    assert (c.max_epochs,c.patience,c.batch_size,c.gradient_clip)==(50,10,128,5)
    assert (c.learning_rate,c.weight_decay,c.device,c.num_threads)==(.001,.0001,"cpu",2)


def test_short_fits_restore_exactly_and_reproduce(data,initial_path,tmp_path,monkeypatch):
    original=experiment.experiment_config
    monkeypatch.setattr(experiment,"experiment_config",lambda seed:replace(original(seed),max_epochs=2,batch_size=8,num_threads=1))
    results,predictions=[],[]
    for n in range(2):
        result=experiment.train_seed(data,tmp_path/f"repeat{n}",42,initial_path)
        results.append(result)
        metrics,p=experiment.restore_validation(tmp_path/f"repeat{n}/best.pt",data)
        predictions.append(p)
        assert metrics==result["validation"]
        assert result["stopped_epoch"]==2 and result["numerical_failures"]==[]
        assert result["gradient_checks"]["completed_backward_updates"]==6
        assert len(result["gradient_checks"]["backward_calls_by_parameter"])==17
        assert all(v>0 for v in result["dynamics_l2_change_from_initial"].values())
        assert result["solver_calls_during_fit_including_epoch_validation"]["rhs_evaluations"]==384
        assert not result["step_size_restored"]["instability_warning"]
        with np.load(tmp_path/f"repeat{n}/validation_predictions.npz") as s:
            np.testing.assert_array_equal(s["sample_indices"],data.validation.sample_indices)
            np.testing.assert_array_equal(s["prediction"],p)
    assert results[0]["validation"]==results[1]["validation"]
    np.testing.assert_array_equal(predictions[0],predictions[1])


def test_solver_failure_is_preserved(data,initial_path,tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,"step_size_diagnostic",lambda *a:{"all_states_and_predictions_finite":True,"instability_warning":True})
    with pytest.raises(FloatingPointError):experiment.train_seed(data,tmp_path/"failed",42,initial_path)
    failure=json.loads((tmp_path/"failed/failure.json").read_text())
    assert not failure["performance_interpretation_permitted"]
    assert (tmp_path/"failed/failed_state.pt").exists()
    assert not (tmp_path/"failed/best.pt").exists()


@pytest.mark.parametrize("value",[-7.,33.,4.125])
def test_output_remains_unbounded(value):
    model=LatentMLPODE()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.head.bias.fill_(value)
        result=model(torch.zeros(3,2,48),torch.zeros(3,63))
    torch.testing.assert_close(result,torch.full((3,),value),rtol=0,atol=0)


def test_metric_difference_direction():
    records=[{"seed":s,"validation":{"mae":2.,"rmse":3.,"r2":.6}} for s in (42,43,44)]
    refs={"results":{"KAN_ODE":{"seed_metrics":{str(s):{"mae":1.,"rmse":4.,"r2":.5} for s in (42,43,44)}},
        "ridge":{"metrics":{"B":{"mae":2.5,"rmse":3.5,"r2":.7}}}}}
    result=experiment.metric_differences(records,refs)
    assert result[0]["MLP_ODE_minus_reference"]["KAN_ODE"]=={"mae":1.,"rmse":-1.,"r2":pytest.approx(.1)}
    assert result[0]["MLP_ODE_minus_reference"]["Ridge_B"]["rmse"]==-.5
