"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_class = len(set(y))

        if depth >= self.max_depth or n_samples < 2 or n_class < 2:
            val = self.most_common(y)
            return {"leaf": True, "value": val}

        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            leaf_val = self.most_common(y)
            return {"leaf": True, "value": leaf_val}

        left, right = self.split_dataset(X, y, feature, threshold)
        left_tree = self._grow_tree(X[left], y[left], depth + 1)
        right_tree = self._grow_tree(X[right], y[right], depth + 1)

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if tree_node["leaf"]:
            return tree_node["value"]

        feature = tree_node["feature"]
        threshold = tree_node["threshold"]

        if x[feature] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])

    def most_common(self, y):
        values, cnt = np.unique(y, return_counts=True)
        return values[np.argmax(cnt)]

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):
        left = np.where(X[:, feature_index] <= threshold)[0]
        right = np.where(X[:, feature_index] > threshold)[0]
        return left, right

    # Find the best split for the dataset
    def find_best_split(self, X, y):
        n_features = X.shape[1]

        best_feature, best_threshold = None, None
        best_gain = -1

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left, right = self.split_dataset(X, y, feature, threshold)
                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self.information_gain(y, y[left], y[right])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def gini(self, arr):
        _, counts = np.unique(arr, return_counts=True)
        prob = counts / len(arr)
        gini = 1 - np.sum(prob ** 2)

        return gini

    def entropy(self, y):
        cnt = np.bincount(y)
        prob = cnt / len(y)

        return -np.sum([p * np.log2(p) for p in prob if p > 0])

    def information_gain(self, parent, left_child, right_child):
        left = len(left_child) / len(parent)
        right = len(right_child) / len(parent)

        gain = self.entropy(parent) - (left * self.entropy(left_child) + right * self.entropy(right_child))

        return gain

    def traversal(self, X, y, node, feature_importance, parent_entropy=1.0):
        if node["leaf"]:
            return

        feature = node["feature"]
        threshold = node["threshold"]

        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold
        X_left, y_left = X[left], y[left]
        X_right, y_right = X[right], y[right]

        left_entropy = self.entropy(y_left)
        right_entropy = self.entropy(y_right)

        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right

        if parent_entropy is None:
            parent_entropy = self.entropy(y)

        impurity_reduction = parent_entropy - (
            (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
        )

        feature_importance[feature] += impurity_reduction

        self.traversal(X_left, y_left, node["left"], feature_importance, left_entropy)
        self.traversal(X_right, y_right, node["right"], feature_importance, right_entropy)

    def compute_feature_importance(self, X, y):
        feature_importance = np.zeros(self.n_features)
        self.traversal(X, y, self.tree, feature_importance)

        # feature_importance = abs(feature_importance)
        return feature_importance
