import typing as t
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 4):
        super(WeakClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def preprocess(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    discrete_cols = X_train.select_dtypes(include=["object"]).columns
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns

    for col in discrete_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train.values, X_test.values


def entropy_loss(outputs, targets):
    epsilon = 1e-10
    outputs = torch.clamp(outputs, epsilon, 1 - epsilon)
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    loss = - (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    y_trues = np.array(y_trues)
    plt.figure(figsize=(8, 6))

    color_map = plt.get_cmap('tab10')
    for idx, preds in enumerate(y_preds):
        preds = np.array(preds)
        if np.any(np.isnan(preds)):
            print(f"Warning: NaN detected in learner {idx}, replaced with 0.5")
            preds = np.nan_to_num(preds, nan=0.5)

        fpr, tpr, thresholds = roc_curve(y_trues, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color_map(idx % 10), label=f'AUC={roc_auc:.4f}')

    # Baseline
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve of Weak Learners')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.show()


def plot_feature_importance(
        feature_importance: t.Sequence[float],
        feature_names: t.Sequence[str],
        title: str = "Feature Importance"
):
    feature_importance = np.array(feature_importance)
    feature_names = np.array(feature_names)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance, color='skyblue')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()
