"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        """Question1
        Complete this function
        """
        sample_num = X.shape[0]
        y = y.reshape((sample_num, 1))

        X_add_bias = np.c_[np.ones((sample_num, 1)), X]
        beta = np.linalg.inv(X_add_bias.T @ X_add_bias) @ X_add_bias.T @ y
        # np.ones: create matrix with row=X.shape[0] col=1
        # # x: equal to Fi() -> N*M, y: equal to t -> N*1
        self.intercept = beta[0, 0]
        self.weights = beta[1:, 0]

    def predict(self, X):
        """Question4
        Complete this function
        """
        X_add_bias = np.c_[np.ones((X.shape[0], 1)), X]
        y_predict = X_add_bias @ np.r_[self.intercept, self.weights]
        return y_predict


class LinearRegressionGradientdescent:
    def fit(
        self,
        X,
        y,
        epochs: int,
        learning_rate: float

    ):
        """Question2
        Complete this function
        """
        X_mean = X.mean(axis=0)
        X_standard = X.std(axis=0)
        X_norm = (X - X_mean) / X_standard

        y_mean = y.mean()
        y_standard = y.std()
        y_norm = ((y - y_mean) / y_standard).reshape((-1, 1))

        sample_num, feature_num = X.shape
        # X: Fi(x) in slide = sample_num * feature_num (design matrix)
        # # beta: w in slide = feature_num * sample_num
        # # y: t in slide = sample_num * 1
        X_add_bias = np.c_[np.ones((sample_num, 1)), X_norm]
        beta = np.zeros((feature_num + 1, 1))

        losses, lr_history = [], []
        for epoch in range(epochs):
            y_predict = X_add_bias @ beta
            # (N*M) @ (M*K), N=sample_num, M=feature_num+1, K=target_num
            error = y_norm - y_predict  # N*1
            gradient = (-2 / sample_num) * (X_add_bias.T @ error)
            beta -= learning_rate * gradient
            loss_norm = np.mean(error ** 2)
            loss = loss_norm * (y_standard ** 2)
            # MSE: divide by sample_num(N) losses.append(loss) lr_history.append(learning_rate)
            losses.append(loss)
            lr_history.append(learning_rate)

            if epoch % 100 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {learning_rate=:.4f}')

            self.intercept = y_mean + y_standard * (beta[0, 0] - np.sum(beta[1:, 0] * X_mean / X_standard))
            self.weights = beta[1:, 0] / X_standard * y_standard

        return losses, lr_history

    def predict(self, X):
        """Question4
        Complete this
        """
        X_add_bias = np.c_[np.ones((X.shape[0], 1)), X]
        y_predict = X_add_bias @ np.r_[self.intercept, self.weights]
        return y_predict


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')  # Load training data
    test_df = pd.read_csv('./test.csv')   # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.6f}')

    LR_GD = LinearRegressionGradientdescent()
    losses, lr_history = LR_GD.fit(train_x, train_y, epochs=500, learning_rate=0.04)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.6f}')

    """
    Question3: Plot the learning curve.
    Implement here
    """
    plt.plot(losses, label='Train MSE loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.20f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.20f}, {mse_gd=:.20f}. Difference: {diff:.20f}%')


if __name__ == '__main__':
    main()
