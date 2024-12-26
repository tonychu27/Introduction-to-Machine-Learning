import pandas as pd
from loguru import logger
import random
import matplotlib.pyplot as plt
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc, accuracy_score


def plot_features_importance(feature_importance, title, name):
    feature_names = [
        "person_age", "person_gender", "person_education", "person_income",
        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file"
    ]

    plt.figure(figsize=(20, 12))
    plt.barh(feature_names, feature_importance, color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(name)


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)  # (36000, 13)
    y_train = train_df['target'].to_numpy()  # (36000, )

    X_test = test_df.drop(['target'], axis=1)  # (9000, 13)
    y_test = test_df['target'].to_numpy()  # (9000)

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=100,
        learning_rate=1e-3,
    )

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')

    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./adaboost.png'
    )

    feature_importance = clf_adaboost.compute_feature_importance()
    # (TODO) Draw the feature importance
    plot_features_importance(feature_importance, "Adaboost", "./Adaboost_features.png")

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=3001,
        learning_rate=1e-3,
    )

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')

    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./bagging.png',
    )

    feature_importance = clf_bagging.compute_feature_importance()
    # (TODO) Draw the feature importance
    plot_features_importance(feature_importance, "Bagging", "./Bagging_features.png")

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )

    arr = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    logger.info(f"Gini: {clf_tree.gini(arr):.6f}, Entropy: {clf_tree.entropy(arr):.6f}")

    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)

    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    feature_importance = clf_tree.compute_feature_importance(X_train, y_train)
    plot_features_importance(feature_importance, "Decision Tree", "./DecisionTree_features.png")


if __name__ == '__main__':
    main()
