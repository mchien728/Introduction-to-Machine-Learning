import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, preprocess, plot_feature_importance


def main():
    """You can control the seed for reproducibility"""
    random.seed(777)
    torch.manual_seed(777)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)  # (n_samples, features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)
    X_train, X_test = preprocess(X_train, X_test)

    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    # Learning_rate: alpha = learning_rate * 1/2 ln(1-ε/ε)
    # use 777 (as same as torch.manual seed)
    # These params are set in adaboost.py
    input_dim = X_train.shape[1]
    clf_adaboost = AdaBoostClassifier(input_dim=input_dim)
    _ = clf_adaboost.fit(X_train, y_train)

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')

    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./temp.png"
    )
    feature_importance = clf_adaboost.compute_feature_importance()

    plot_feature_importance(feature_importance, feature_names, title="Adaboost Feature Importance")

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=input_dim)
    _ = clf_bagging.fit(X_train, y_train)

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)

    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
    )
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, title="Bagging Feature Importance")

    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    feature_importance = clf_tree.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, title="Decision Tree Feature Importance")


if __name__ == '__main__':
    main()
