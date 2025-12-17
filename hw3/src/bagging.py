import typing as t
import numpy as np
import torch
import random
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10, random_state: int = 777) -> None:
        # Free to add args as you need, like batch-size, learning rate, etc.
        self.num_learners = num_learners
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]
        if random_state is not None:
            random.seed(random_state)
            torch.manual_seed(random_state)

    def fit(self, X_train, y_train, num_epochs: int = 50, learning_rate: float = 0.01):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        n_samples = X_train.shape[0]

        for learner in self.learners:
            idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X_train[idxs]
            y_sample = y_train[idxs]

            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                preds = learner(X_sample)
                losses = entropy_loss(preds, y_sample).mean()
                losses.backward()
                optimizer.step()
        return self

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        # Soft voting
        X = torch.tensor(X, dtype=torch.float32)
        y_pred_probs = []

        for learner in self.learners:
            learner.eval()
            with torch.no_grad():
                prob = learner(X).view(-1)
                y_pred_probs.append(prob)

        # hard label
        y_pred_stack = torch.stack(y_pred_probs, dim=1)
        majority_vote = (y_pred_stack >= 0.5).float().sum(dim=1)
        y_pred_classes = (majority_vote >= len(self.learners) / 2).int().cpu().numpy()
        y_pred_probs_list = [p.detach().cpu().numpy() for p in y_pred_probs]

        return y_pred_classes, y_pred_probs_list

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importances = None
        for learner in self.learners:
            w1 = learner.fc1.weight.data
            w2 = learner.fc2.weight.data.view(-1)
            learner_importance = torch.sum(torch.abs(w1) * torch.abs(w2).unsqueeze(1), dim=0)

            if feature_importances is None:
                feature_importances = learner_importance
            else:
                feature_importances += learner_importance

        return (feature_importances / len(self.learners)).cpu().numpy()
