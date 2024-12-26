import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def exp_loss(self, y_true, y_pred):
        return torch.mean(torch.exp(-y_true * y_pred))

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []

        n_samples = X_train.shape[0]

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_tensor = 2 * y_tensor - 1

        self.sample_weights = torch.ones(n_samples) / n_samples

        for i, model in enumerate(self.learners):
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()

                y_pred = model(X_tensor)
                loss = self.exp_loss(self.sample_weights * y_tensor, y_pred)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                y_pred = model(X_tensor).sign()
                incorrect = (y_pred != y_tensor).float()
                weighted_error = (self.sample_weights * incorrect).sum()

            alpha = 0.7 * torch.log((1 - weighted_error) / (weighted_error))
            self.alphas.append(alpha.item())

            self.sample_weights *= torch.exp(torch.tensor([alpha if error else -alpha
                                                           for error in incorrect], dtype=torch.float32))
            self.sample_weights /= self.sample_weights.sum()

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:

        n_sample = X.shape[0]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = torch.zeros(n_sample)
        probs = []

        for alpha, model in zip(self.alphas, self.learners):
            model.eval()

            logit = model(X_tensor)
            y_pred = logit.sign()

            prob = torch.sigmoid(logit)
            probs.append(prob.detach().numpy())

            preds += alpha * y_pred

        preds = preds.detach().numpy()
        preds = (preds > 0).astype(int)

        return preds, np.array(probs).T

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importances = np.zeros(self.learners[0].model[0].in_features)

        for alpha, learner in zip(self.alphas, self.learners):
            layer_weights = learner.model[0].weight.abs().detach().numpy().sum(axis=0)
            feature_importances += alpha * layer_weights

        feature_importances = abs(feature_importances)
        return feature_importances.tolist()
