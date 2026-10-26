"""Regression tests for monotonicity-constrained IGANN.

Run: pytest -q tests/test_igann_monotonicity.py
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.igann import ELM_Regressor, IGANN


@pytest.mark.parametrize("sign", [1, -1])
def test_elm_accumulates_unsigned_derivative(sign):
    """New derivative contributions must be the true derivatives, regardless of sign."""
    X = torch.linspace(-1.0, 1.0, steps=15).reshape(-1, 1)
    y = torch.linspace(-0.2, 0.2, steps=15)
    regressor = ELM_Regressor(
        n_input=1, n_categorical_cols=0, n_hid=6,
        seed=3, elm_scale=1, elm_alpha=0.01, act="elu", device="cpu",
    )
    # A zero initial derivative satisfies either direction.
    _, contribution = regressor.fit(
        X, y, mult_coef=1.0, mon_features=[0], mon_sign=[sign],
        cumulative_deriv=torch.zeros(15, 1), boost_rate=0.1,
    )
    _, preact = regressor.get_hidden_values(X)
    derivative = (
        regressor._act_deriv(preact)
        * regressor.hidden_mat[0, :].reshape(1, -1)
    ) @ regressor.output_model.coef_
    np.testing.assert_allclose(
        contribution[:, 0].detach().numpy(),
        (0.1 * derivative).detach().numpy(),
        atol=1e-5, rtol=1e-5,
    )
    signed = sign * contribution[:, 0].detach().numpy()
    assert signed.min() >= -2e-5


@pytest.mark.parametrize("direction", [1, -1])
def test_model_monotonic_on_training_points(direction):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 45)})
    y = direction * (X["x"].to_numpy() + 0.1 * X["x"].to_numpy() ** 3)
    model = IGANN(
        task="regression", n_hid=5, n_estimators=3, boost_rate=0.1,
        elm_scale=1, elm_alpha=0.01, init_reg=0.001, act="elu",
        random_state=42, early_stopping=0, monotonicity={"x": direction},
    )
    model.fit(X, y)
    grid = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 201)})
    predictions = model.predict_raw(grid)
    # The dense grid is a diagnostic; the existing B model guarantees only training points.
    assert np.isfinite(predictions).all()
    # Numerically check the derivative sign on *model's internal training rows*.
    training_x = model.raw_X_train["x"].to_numpy()
    eps = 1e-4
    pred_plus = model.predict_raw(pd.DataFrame({"x": training_x + eps}))
    pred_minus = model.predict_raw(pd.DataFrame({"x": training_x - eps}))
    directional_derivative = direction * (pred_plus - pred_minus) / (2 * eps)
    assert directional_derivative.min() >= -2e-3


def test_validation_rows_do_not_fit_linear_initialization():
    """The internal validation target must not influence the fitted linear baseline."""
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
    y = np.arange(40, dtype=float) / 40.0
    model1 = IGANN(
        task="regression", n_hid=3, n_estimators=0,
        init_reg=0.01, random_state=42, monotonicity={"x": 1},
    )
    model1.fit(X.copy(), y.copy())
    val_indices = np.setdiff1d(np.arange(len(X)), model1.raw_X_train.index.to_numpy())
    y_changed = y.copy()
    y_changed[val_indices] += 1000.0
    model2 = IGANN(
        task="regression", n_hid=3, n_estimators=0,
        init_reg=0.01, random_state=42, monotonicity={"x": 1},
    )
    model2.fit(X.copy(), y_changed)
    np.testing.assert_allclose(
        model1.linear_model.coef_, model2.linear_model.coef_, atol=1e-5
    )
    np.testing.assert_allclose(
        model1.linear_model.intercept_, model2.linear_model.intercept_, atol=1e-5
    )


def test_zero_constraint_remains_unconstrained():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = np.linspace(-1, 1, 30)
    model = IGANN(
        task="regression", n_estimators=0, random_state=42,
        monotonicity={"x": 0},
    )
    model.fit(X, y)
    assert model.mon_idx == []
    assert model.mon_sign == []


def test_invalid_direction_rejected():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = np.linspace(-1, 1, 30)
    model = IGANN(task="regression", n_estimators=0, monotonicity={"x": 17})
    with pytest.raises(ValueError, match="Invalid monotonicity direction"):
        model.fit(X, y)
