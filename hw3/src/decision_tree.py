import numpy as np
import typing as t


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {
                "leaf": True,
                "value": np.bincount(y).argmax()
            }
        best_feature, best_threshold, best_info_gain = find_best_split(X, y)
        if best_info_gain == -1:
            return {
                "leaf": True,
                "value": np.bincount(y).argmax()
            }
        X_left, y_left, X_right, y_right = split_dataset(X, y, best_feature, best_threshold)
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "info_gain": best_info_gain,
            "left": self._grow_tree(X_left, y_left, depth + 1),
            "right": self._grow_tree(X_right, y_right, depth + 1)
        }

    def predict(self, X):
        preds = [self._predict_tree(x, self.tree) for x in X]
        return np.array(preds)

    def _predict_tree(self, x, tree_node):
        if tree_node["leaf"]:
            return tree_node["value"]
        if x[tree_node["feature"]] <= tree_node["threshold"]:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importances = np.zeros(self.n_features)

        def traverse_calc(tree_node):
            if tree_node["leaf"]:
                return
            feature_idx = tree_node["feature"]
            info_gain = tree_node.get("info_gain", 0.0)
            feature_importances[feature_idx] += info_gain
            traverse_calc(tree_node["left"])
            traverse_calc(tree_node["right"])
        traverse_calc(self.tree)
        total = feature_importances.sum()
        if total > 0:
            feature_importances /= total

        return feature_importances.tolist()


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_node = X[:, feature_index] <= threshold
    right_node = ~left_node
    return X[left_node], y[left_node], X[right_node], y[right_node]


# Find the best split for the dataset
def find_best_split(X, y):
    n_samples, n_features = X.shape
    best_info_gain = -1
    best_feature = None
    best_threshold = None

    parent_entropy = entropy(y)
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_idx, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            weight_left = len(y_left) / n_samples
            weight_right = len(y_right) / n_samples
            child_entropy = weight_left * entropy(y_left) + weight_right * entropy(y_right)

            info_gain = parent_entropy - child_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_info_gain


def entropy(y):
    sample_counts = np.bincount(y)  # calculate the number of each type
    probs = sample_counts / len(y)
    entropy_value = -np.sum([p * np.log2(p) for p in probs if p > 0])

    return entropy_value
