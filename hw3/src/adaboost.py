import typing as t
import random
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10,
                 learning_rate: float = 0.65, random_state: int = 777) -> None:
        # Free to add args as you need, like batch-size, learning rate, etc
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alphas = []

        if random_state is not None:
            random.seed(random_state)
            torch.manual_seed(random_state)

    def fit(self, X_train, y_train, num_epochs: int = 150, learning_rate: float = 0.005):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        n_samples = X_train.shape[0]

        self.sample_weights = torch.ones(n_samples) / n_samples
        for learner in self.learners:
            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()  # avoid result is added to previous gradient
                preds = learner(X_train)
                losses = entropy_loss(preds, y_train)
                weight_losses = losses.mean()
                weight_losses.backward()
                optimizer.step()

            with torch.no_grad():
                preds_class = (learner(X_train) >= 0.5).float().view(-1)
                wrong = (preds_class != y_train).float()  # 0/1
                error_t = torch.dot(self.sample_weights, wrong) / self.sample_weights.sum()
                error_t = torch.clamp(error_t, 1e-6, 1 - 1e-6)

            alpha = self.learning_rate * 0.5 * torch.log((1 - error_t) / error_t)
            self.alphas.append(alpha.item())

            y = 2 * y_train - 1  # convert to 1/-1
            h_preds = 2 * preds_class - 1
            self.sample_weights *= torch.exp(-alpha * y * h_preds)
            self.sample_weights /= self.sample_weights.sum()

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        X = torch.tensor(X, dtype=torch.float32)
        final_score = torch.zeros(X.shape[0])
        y_pred_probs = []
        for alpha, learner in zip(self.alphas, self.learners):
            learner.eval()
            with torch.no_grad():
                prob = learner(X).detach().numpy().reshape(-1)
                y_pred_probs.append(prob)

                preds = (learner(X) >= 0.5).float().view(-1)  # 0/1
                h_preds = 2 * preds - 1  # convert to -1/1
                final_score += alpha * h_preds

        y_pred_classes = (final_score >= 0).long().numpy()  # 0/1
        return y_pred_classes, y_pred_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importance = None

        for alpha, learner in zip(self.alphas, self.learners):
            w1 = learner.fc1.weight.data  # (hidden, input)
            w2 = learner.fc2.weight.data.view(-1)  # (1, hidden) -> (hidden,)

            # calculate sum of each column
            learner_feature = torch.sum(torch.abs(w1) * torch.abs(w2).unsqueeze(1), dim=0)
            learner_feature = abs(alpha) * learner_feature

            if feature_importance is None:
                feature_importance = learner_feature
            else:
                feature_importance += learner_feature

        return feature_importance.cpu().numpy()
