"""Danh gia mo hinh."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(y_true, y_pred) -> dict:
    """Tra ve cac chi so danh gia hoi quy."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
