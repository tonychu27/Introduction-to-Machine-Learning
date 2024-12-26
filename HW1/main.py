import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        bX = np.c_[np.ones(X.shape[0]), X]
        beta = np.linalg.inv(bX.T @ bX) @ bX.T @ y
        self.intercept = beta[0]
        self.weights = beta[1:]

    def predict(self, X):
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        m, n = X.shape
        y = y.flatten()
        self.weights = np.zeros(n)
        self.intercept = -33.8
        losses = []

        for epoch in range(epochs):
            pred = self.predict(X)

            weight_gd = (X.T @ (pred - y)) / m
            intercpet_gd = (pred - y).mean()

            self.intercept -= intercpet_gd * learning_rate
            self.weights -= weight_gd * learning_rate

            loss = compute_mse(pred, y)
            losses.append(loss)

        return losses

    def predict(self, X):
        return X @ self.weights + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Learning Curve')
        plt.show()


def compute_mse(prediction, ground_truth):
    err = (ground_truth - prediction) ** 2
    return err.mean()


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=20000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
