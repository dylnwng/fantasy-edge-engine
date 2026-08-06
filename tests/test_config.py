import yaml
import pytest

from edge_engine.model.config import load_model_config
from edge_engine.simulation.config import load_simulation_config


def _write(path, data):
    path.write_text(yaml.dump(data))


# ---- model_config.yaml / flag_margin ----

def _model_config_data(**overrides):
    data = {"train_seasons": [2020, 2021], "validation_season": 2022, "flag_margin": 3.0}
    data.update(overrides)
    return data


def test_model_config_loads_with_valid_flag_margin(tmp_path):
    path = tmp_path / "model_config.yaml"
    _write(path, _model_config_data(flag_margin=3.0))
    config = load_model_config(path)
    assert config.flag_margin == 3.0


def test_model_config_rejects_negative_flag_margin(tmp_path):
    # A negative flag_margin flips confidence_tier()'s comparisons, so a
    # player predicted to score *below* their own baseline could be
    # labeled "High confidence."
    path = tmp_path / "model_config.yaml"
    _write(path, _model_config_data(flag_margin=-3.0))
    with pytest.raises(ValueError, match="flag_margin must be >= 0"):
        load_model_config(path)


def test_model_config_rejects_validation_season_inside_train_seasons(tmp_path):
    # The PRD's whole reason for a held-out SEASON split is to avoid
    # leakage -- a validation_season also present in train_seasons
    # doesn't crash anything downstream, it silently validates the model
    # on data it trained on, invalidating every accuracy number in
    # EVALUATION.md with no warning.
    path = tmp_path / "model_config.yaml"
    _write(path, _model_config_data(train_seasons=[2020, 2021, 2022], validation_season=2022))
    with pytest.raises(ValueError, match="is also in train_seasons"):
        load_model_config(path)


def test_model_config_rejects_flag_margin_as_a_quoted_string(tmp_path):
    # A QA pass found this crashes with a raw, confusing TypeError if
    # not explicitly type-checked first.
    path = tmp_path / "model_config.yaml"
    _write(path, _model_config_data(flag_margin="3.0"))
    with pytest.raises(ValueError, match="must be a number"):
        load_model_config(path)


# ---- simulation_config.yaml / n_sims, distribution, team_correlation ----

def test_simulation_config_loads_with_valid_values(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"n_sims": 5000, "distribution": "gamma", "team_correlation": 0.05})
    config = load_simulation_config(path)
    assert config.n_sims == 5000
    assert config.distribution == "gamma"
    assert config.team_correlation == 0.05


def test_simulation_config_rejects_zero_n_sims(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"n_sims": 0})
    with pytest.raises(ValueError, match="n_sims must be positive"):
        load_simulation_config(path)


def test_simulation_config_rejects_negative_n_sims(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"n_sims": -100})
    with pytest.raises(ValueError, match="n_sims must be positive"):
        load_simulation_config(path)


def test_simulation_config_rejects_n_sims_as_a_quoted_string(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"n_sims": "10000"})
    with pytest.raises(ValueError, match="must be a number"):
        load_simulation_config(path)


def test_simulation_config_rejects_unrecognized_distribution(tmp_path):
    # A typo (e.g. "Gamma" capitalized) would otherwise silently fall
    # back to "normal" with no warning.
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"distribution": "Gamma"})
    with pytest.raises(ValueError, match="is not recognized"):
        load_simulation_config(path)


def test_simulation_config_rejects_team_correlation_above_one(tmp_path):
    # team_correlation > 1.0 produces sqrt(negative) internally, silently
    # turning every simulated score into NaN while simulate_matchup still
    # reports an exact, fabricated 0.0% win probability.
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"team_correlation": 1.5})
    with pytest.raises(ValueError, match="team_correlation must be between"):
        load_simulation_config(path)


def test_simulation_config_rejects_negative_team_correlation(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"team_correlation": -0.5})
    with pytest.raises(ValueError, match="team_correlation must be between"):
        load_simulation_config(path)


def test_simulation_config_rejects_team_correlation_as_a_quoted_string(tmp_path):
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"team_correlation": "0.5"})
    with pytest.raises(ValueError, match="must be a number"):
        load_simulation_config(path)


def test_simulation_config_rejects_a_bool_where_a_number_is_expected(tmp_path):
    # YAML's bool/int overlap (bool is a subclass of int in Python) means
    # `True <= 1.0` would otherwise silently pass as 1 without an
    # explicit type check.
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"team_correlation": True})
    with pytest.raises(ValueError, match="must be a number"):
        load_simulation_config(path)


def test_simulation_config_accepts_boundary_values(tmp_path):
    # 0.0 and 1.0 are the inclusive, valid boundary of team_correlation.
    path = tmp_path / "simulation_config.yaml"
    _write(path, {"team_correlation": 0.0})
    assert load_simulation_config(path).team_correlation == 0.0

    _write(path, {"team_correlation": 1.0})
    assert load_simulation_config(path).team_correlation == 1.0
