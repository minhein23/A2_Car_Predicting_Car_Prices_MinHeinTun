import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, regularization=None, alpha=0.01, momentum=0.0, init_method="zero"):
        """
        Initialize Linear Regression model.

        Args:
        - regularization (str): "lasso", "ridge", or None.
        - alpha (float): Learning rate.
        - momentum (float): Momentum factor (0 means no momentum).
        - init_method (str): "zero" or "xavier" weight initialization.
        """
        self.regularization = regularization
        self.alpha = alpha
        self.momentum = momentum
        self.init_method = init_method
        self.theta = None  # Model weights
        self.prev_step = 0  # Momentum term

    def _initialize_weights(self, n_features):
        """ Initialize weights using zero or Xavier initialization. """
        if self.init_method == "zero":
            self.theta = np.zeros(n_features)
        elif self.init_method == "xavier":
            limit = np.sqrt(1 / n_features)
            self.theta = np.random.uniform(-limit, limit, n_features)

    def fit(self, X, y, epochs=1000):
        """
        Train the model using gradient descent.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        for _ in range(epochs):
            gradient = (X.T @ (X @ self.theta - y)) / n_samples
            
            # Apply regularization
            if self.regularization == "ridge":
                gradient += 0.1 * self.theta  # L2 penalty (lambda = 0.1)
            elif self.regularization == "lasso":
                gradient += 0.1 * np.sign(self.theta)  # L1 penalty

            # Apply momentum-based gradient descent
            step = self.alpha * gradient
            self.theta -= step + self.momentum * self.prev_step
            self.prev_step = step

    def predict(self, X):
        """
        Predict output values.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return X @ self.theta

    def r2_score(self, y_true, y_pred):
        """
        Compute R² Score.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance based on absolute coefficient values.
        """
        importance = np.abs(self.theta[1:])  # Ignore bias term
        plt.figure(figsize=(10, 5))
        plt.bar(feature_names, importance)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance (Coefficient Magnitude)")
        plt.xticks(rotation=45)
        plt.show()


# Normal Linear Regression (No Regularization)
class Normal(LinearRegression):
    def __init__(self, alpha=0.01, momentum=0.0, init_method="zero"):
        super().__init__(regularization=None, alpha=alpha, momentum=momentum, init_method=init_method)


# Ridge Regression (L2 Regularization)
class Ridge(LinearRegression):
    def __init__(self, alpha=0.01, momentum=0.0, init_method="zero"):
        super().__init__(regularization="ridge", alpha=alpha, momentum=momentum, init_method=init_method)


# Lasso Regression (L1 Regularization)
class Lasso(LinearRegression):
    def __init__(self, alpha=0.01, momentum=0.0, init_method="zero"):
        super().__init__(regularization="lasso", alpha=alpha, momentum=momentum, init_method=init_method)
