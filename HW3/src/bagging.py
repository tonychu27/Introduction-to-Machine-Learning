import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        losses_of_models = []
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        for i, model in enumerate(self.learners):
            # Bootstrap sampling
            indices = torch.randint(0, len(X_train), (len(X_train),))
            X_bagging = X_tensor[indices]
            y_bagging = y_tensor[indices]

            model.train()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()

                y_pred = model(X_bagging).squeeze()
                loss = entropy_loss(y_pred, y_bagging)

                # if epoch % 1500 == 0:
                #     logger.info(f'Model {i:2}, Epoch: {epoch:4}, Loss: {loss:.6f}')

                loss.backward()
                optimizer.step()

            losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Tuple[np.ndarray, np.ndarray]:

        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = []
        probs = []

        for model in self.learners:
            model.eval()

            pred = model(X_tensor).squeeze()
            prob = torch.sigmoid(pred)
            probs.append(prob.detach().numpy())
            preds.append(pred.detach().numpy())

        preds = np.array(preds)
        y_class = (preds.mean(axis=0) >= 0.5).astype(int)
        return y_class, np.array(probs).T

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importances = np.zeros(self.learners[0].model[0].in_features)

        for model in self.learners:
            layer_weights = model.model[0].weight.abs().detach().numpy()
            feature_importances += layer_weights.sum(axis=0)

        return feature_importances.tolist()
