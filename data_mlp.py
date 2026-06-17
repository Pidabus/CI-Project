"""
MLP regression module for the Student Performance prediction project.
Trains an MLPRegressor with GridSearchCV and evaluates on the test set.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_evaluate_mlp(X_train, X_test, y_train, y_test):
    """
    Train an MLP regressor via GridSearchCV and evaluate on the test set.

    Args:
        X_train: Training feature matrix (preprocessed).
        X_test:  Test feature matrix (preprocessed).
        y_train: Training target values (G3).
        y_test:  Test target values (G3).

    Returns:
        best_model: Best estimator found by GridSearchCV.
        y_pred:     Predicted values on X_test.
    """
    # Base MLP regressor (random_state fixed for reproducibility)
    # early_stopping stops training when validation loss plateaus, which helps
    # convergence within max_iter and avoids useless extra epochs
    mlp = MLPRegressor(random_state=42, early_stopping=True)

    # Hyperparameter grid required by the project report
    param_grid = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [500],
    }

    # 48 combinations × 5-fold CV = 240 model fits (not an infinite loop)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=1)

    print("Starting grid search (48 hyperparameter combinations, 5-fold CV)...")
    print("This may take several minutes. Progress will be shown below.\n")

    # Suppress repeated ConvergenceWarning spam when max_iter is reached
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict on the held-out test set
    y_pred = best_model.predict(X_test)

    # Evaluation metrics (np.sqrt for RMSE — compatible with all sklearn versions)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MLP evaluation on test set:")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    return best_model, y_pred


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess

    X_train, X_test, y_train, y_test = load_and_preprocess()
    best_model, y_pred = train_and_evaluate_mlp(X_train, X_test, y_train, y_test)
