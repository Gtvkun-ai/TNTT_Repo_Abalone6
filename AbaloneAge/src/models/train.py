"""Huan luyen mo hinh co ban."""

from sklearn.ensemble import RandomForestRegressor


def train_baseline_model(x_train, y_train):
    """Huan luyen mo hinh baseline."""
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    return model
