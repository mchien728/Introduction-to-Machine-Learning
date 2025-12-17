import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-2, num_iterations: int = 5000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        X = inputs
        t = np.array(targets)
        sample_num, feature_num = X.shape
        self.weights = np.zeros(feature_num)
        self.intercept = 0

        for _ in range(self.num_iterations):
            a_i = X @ self.weights + self.intercept
            y_predict = self.sigmoid(a_i)
            #  (1 / sample_num) to average the gradient
            # avoid batch size updating to modify learning rate
            dw = (1 / sample_num) * (X.T @ (y_predict - t))
            d_intercept = (1 / sample_num) * np.sum(y_predict - t)
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * d_intercept
            if _ % 10000 == 0:
                loss = -np.mean(t * np.log(y_predict + 1e-8) + (1 - t) * np.log(1 - y_predict + 1e-8))
                print(f"iter {_}: loss = {loss:.4f}")

    def predict(self, inputs: npt.NDArray[np.float64]) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        a_i = inputs @ self.weights + self.intercept
        y_predict = self.sigmoid(a_i)
        y_pred_class = (y_predict >= 0.5).astype(int)
        return y_predict, y_pred_class

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


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
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        #  Inputs have 2 features
        #  Inputs dimension: N x 2
        x_0 = inputs[np.array(targets) == 0]
        x_1 = inputs[np.array(targets) == 1]
        #  m0, m1 dimension: 1 x 2
        self.m0 = x_0.mean(axis=0)
        self.m1 = x_1.mean(axis=0)
        #  sw dimension: 2 x 2
        sw_0 = (x_0 - self.m0).T @ (x_0 - self.m0)
        sw_1 = (x_1 - self.m1).T @ (x_1 - self.m1)
        self.sw = sw_0 + sw_1
        #  sb dimension: 2 x 2
        self.sb = np.outer(self.m1 - self.m0, self.m1 - self.m0)
        #  w dimension: 2 x 1, but m1-m0 is 1 x 2
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)

    def predict(self, inputs: npt.NDArray[np.float64]) -> t.Sequence[t.Union[int, bool]]:
        #  Inputs dimansion: N x 2
        #  w dimension: 2 x 1
        y_project = inputs @ self.w
        #  w_0: -threshold
        threshold = (self.m0 @ self.w + self.m1 @ self.w) / 2
        y_predict = (y_project >= threshold).astype(int)
        return y_predict

    def plot_projection(self, inputs: npt.NDArray[np.float64], targets: t.Sequence[int]):
        y_predict = self.predict(inputs)

        center = inputs.mean(axis=0)
        self.slope = self.w[1] / self.w[0]
        intercept = center[1] - self.slope * center[0]
        points_x = np.linspace(inputs[:, 0].min(), inputs[:, 0].max(), 100)
        points_y = self.slope * points_x + intercept
        plt.plot(points_x, points_y, color='gray', label='Projection line')

        threshold = (self.m0 @ self.w + self.m1 @ self.w) / 2
        points_y_bound = (threshold - self.w[0] * points_x) / self.w[1]
        plt.plot(points_x, points_y_bound, color='blue', linestyle='--', label='Decision boundary')

        w_unit = self.w / np.linalg.norm(self.w)
        projections = ((inputs - center) @ w_unit)[:, None] * w_unit + center

        for (x, y), (xp, yp), y_true, y_pred in zip(inputs, projections, targets, y_predict):
            marker = 'o' if y_true == 0 else '^'
            color = 'green' if y_true == y_pred else 'red'
            plt.scatter(x, y, c=color, marker=marker)
            plt.scatter(xp, yp, c='black', s=20, zorder=3)
            plt.plot([x, xp], [y, yp], color='gray', alpha=0.3)

        plt.title(f'Projection onto FLD axis (slope={self.slope:.2f}, intercep={intercept:.2f})')
        plt.legend()
        plt.show()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    return np.mean(y_trues == y_preds)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(x_train.shape)
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=2e-2,  # You can modify the parameters as you want
        num_iterations=9000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
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
    y_predict = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    ...

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test)


if __name__ == '__main__':
    main()
