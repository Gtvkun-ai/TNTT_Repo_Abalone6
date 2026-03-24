from src.models.evaluate import evaluate_regression_model


def test_evaluate_regression_model_returns_metrics():
    metrics = evaluate_regression_model([1, 2, 3], [1, 2, 3])

    assert set(metrics.keys()) == {"mae", "mse", "r2"}
