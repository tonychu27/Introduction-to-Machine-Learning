import typing as t
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    # For Binary Class
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})

    # For Categorical Class
    categorical = {
        'person_education':
        {'Master': 4, 'High School': 1, 'Bachelor': 3, 'Associate': 2, 'Doctorate': 5},
        'person_home_ownership':
        {'OWN': 4, 'RENT': 3, 'MORTGAGE': 2, 'OTHER': 1},
        'loan_intent':
        {'VENTURE': 6, 'MEDICAL': 5, 'PERSONAL': 4, 'HOMEIMPROVEMENT': 3, 'DEBTCONSOLIDATION': 2, 'EDUCATION': 1}
    }

    for col, mapping in categorical.items():
        df[col] = df[col].map(mapping)

    numerical = [
        'person_age',
        'person_income',
        'person_emp_exp',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score'
    ]

    for col in numerical:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

    return df.to_numpy(dtype='float32')


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        k = 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, k),
            nn.Linear(k, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x.squeeze()


def accuracy_score(y_trues, y_preds) -> float:
    correct = sum(1 for truth, predict in zip(y_trues, y_preds) if truth == predict)
    return correct / len(y_trues)


def entropy_loss(outputs, targets):
    outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
    return -torch.mean(targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure(figsize=(10, 8))

    for i in range(y_preds.shape[1]):
        pred_prob = y_preds[:, i]

        fpr, tpr, _ = roc_curve(y_trues, pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'Learner {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(fpath)
