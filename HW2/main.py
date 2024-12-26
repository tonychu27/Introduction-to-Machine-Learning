import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None
        self.losses = []

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in
        self.weights and self.intercept.
        """
        m, n = inputs.shape
        self.weights = np.zeros(n)
        self.intercept = -1

        for epoch in range(self.num_iterations):
            hypothesis = self.sigmoid(inputs @ self.weights + self.intercept)
            weight_gd = inputs.T @ (hypothesis - targets) / m
            intercept_gd = (hypothesis - targets).mean()

            self.weights -= self.learning_rate * weight_gd
            self.intercept -= self.learning_rate * intercept_gd

            loss = self.cross_entropy(targets, hypothesis)
            self.losses.append(loss)

            # if (epoch % 5000 == 0):
            #     logger.info(f'LR: Epoch={epoch}, loss={loss:.6f}')

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        z = inputs @ self.weights + self.intercept
        prob = self.sigmoid(z)
        predict = (prob > 0.5).astype(int)

        return (prob, predict)

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def cross_entropy(self, y_trues, y_preds):
        loss = (y_trues @ np.log(y_preds) + (1 - y_trues) @ np.log(1 - y_preds)).mean()
        return -loss


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:

        class0 = inputs[np.where(targets == 0)]
        class1 = inputs[np.where(targets == 1)]
        self.m0 = class0.mean(axis=0)
        self.m1 = class1.mean(axis=0)

        self.sb = np.outer(self.m1 - self.m0, self.m1 - self.m0)

        self.sw = np.zeros((2, 2))
        for x in class0:
            self.sw += np.outer(x - self.m0, x - self.m0)
        for x in class1:
            self.sw += np.outer(x - self.m1, x - self.m1)

        self.w = (self.m1 - self.m0) @ np.linalg.inv(self.sw)
        self.w /= np.linalg.norm(self.w)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        m, _ = inputs.shape
        y_pred = np.zeros(m)
        data_projected = inputs @ self.w
        m0_projected = self.m0 @ self.w
        m1_projected = self.m1 @ self.w

        for idx in range(m):
            m0_dist = np.abs(m0_projected - data_projected[idx])
            m1_dist = np.abs(m1_projected - data_projected[idx])

            if m0_dist > m1_dist:
                y_pred[idx] = 1
            else:
                y_pred[idx] = 0

        return y_pred

    def plot_projection(self, inputs: npt.NDArray[float]):
        y_pred = self.predict(inputs)
        self.slope = self.w[1] / self.w[0]
        x = np.linspace(-2.0, 2.0)
        intercept = 0
        y = self.slope * x + intercept

        class0 = inputs[np.where(y_pred == 0)]
        class1 = inputs[np.where(y_pred == 1)]

        class0_projectX = np.outer((class0 @ self.w), self.w)[:, 0]
        class0_projectY = np.outer((class0 @ self.w), self.w)[:, 1]
        class1_projectX = np.outer((class1 @ self.w), self.w)[:, 0]
        class1_projectY = np.outer((class1 @ self.w), self.w)[:, 1]

        fig, ax = plt.subplots()
        ax.set_title(f'Projection Line: w={self.slope}, b={intercept}')
        ax.set_ylim(-2.0, 2.0)
        ax.plot(x, y)
        ax.plot([class0[:, 0], class0_projectX], [class0[:, 1], class0_projectY], c='black', alpha=0.05)
        ax.plot([class1[:, 0], class1_projectX], [class1[:, 1], class1_projectY], c='black', alpha=0.05)
        ax.scatter(class0[:, 0], class0[:, 1], c='r', s=10)
        ax.scatter(class1[:, 0], class1[:, 1], c='b', s=10)
        ax.scatter(class0_projectX, class0_projectY, c='r', s=10)
        ax.scatter(class1_projectX, class1_projectY, c='b', s=10)
        plt.show()


def compute_auc(y_trues, y_preds):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true=y_trues, y_score=y_preds)


def accuracy_score(y_trues, y_preds):
    correct = 0
    for truth, predict in zip(y_trues, y_preds):
        if truth == predict:
            correct += 1

    return correct / len(y_trues)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    # (n_samples, n_features)
    x_train = train_df.drop(['target'], axis=1).to_numpy()
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-4,  # You can modify the parameters as you want
        num_iterations=40000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train)


if __name__ == '__main__':
    main()
