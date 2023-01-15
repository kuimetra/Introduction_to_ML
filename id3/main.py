# Alina Artemiuk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import numpy as np


class Tree:
    def __init__(self, data, label=None, split=None, parent=None, children=None, error=0.0):
        self.data = data
        self.label = label
        self.split = split
        self.parent = parent
        self.children = children or []
        self.error = error

    def get_children(self):
        return self.children

    def set_label(self, label):
        self.label = label

    def set_split(self, split):
        self.split = split

    def add_child(self, child):
        self.children.append(child)

    def add_error(self):
        self.error += 1

    def is_leaf(self):
        return not self.children

    def tree_accuracy(self, X_test, y_test):
        y_pred = [predict(x, self) for x in X_test]
        return accuracy_score(y_test, y_pred)


def shuffle_dataset(data):
    return data.sample(frac=1)


def train_test_split(data, split_ratio=0.7, shuffle=False):
    if shuffle:
        data = shuffle_dataset(data)
    train_size = round(split_ratio * len(data))
    return data[:train_size], data[train_size:]


def matrix_vector_split(data):
    return np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])


def matrix_vector_merge(X, y):
    return pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)


def max_information_gain(X, y, attr, impurity_measure):
    return attr[np.argmax([information_gain(X, y, f, impurity_measure) for f in attr])]


def information_gain(X, y, feature, impurity_measure):
    if impurity_measure == "entropy":
        ig = calc_entropy(y)
    elif impurity_measure == "gini":
        ig = calc_gini(y)
    else:
        return None
    label, label_counts = np.unique(X[:, feature], return_counts=True)
    for l in label:
        target_values = y[np.where(X[:, feature] == l)]
        lb_counts = np.unique(target_values, return_counts=True)[1]
        proportions = lb_counts / label_counts.sum()
        if impurity_measure == "entropy":
            ig += np.sum(-proportions * calc_entropy(target_values))
        elif impurity_measure == "gini":
            ig += np.sum(-proportions * calc_gini(target_values))
    return ig


def calc_entropy(y):
    label, label_counts = np.unique(y, return_counts=True)
    if len(label) == 1:
        return 0
    elif len(np.unique(label_counts)) == 1:
        return 1
    proportions = label_counts / label_counts.sum()
    return np.sum(-proportions * np.log2(proportions))


def calc_gini(y):
    label, label_counts = np.unique(y, return_counts=True)
    if len(label) == 1:
        return 0
    elif len(np.unique(label_counts)) == 1:
        return 1
    proportions = label_counts / label_counts.sum()
    return 1 - np.sum(proportions ** 2)


def learn(X, y, impurity_measure, shuffle=False, pruning=False, split_ratio=0.7):
    attr = list(range(X.shape[1]))
    if pruning:
        train_prune_ds = matrix_vector_merge(X, y)
        train_ds, prune_ds = train_test_split(train_prune_ds, split_ratio, shuffle)
        (X_train, y_train) = matrix_vector_split(train_ds)
        (X_prune, y_prune) = matrix_vector_split(prune_ds)
        node = id3(X_train, y_train, attr, impurity_measure)
        for X_p, y_p in zip(X_prune, y_prune):
            add_error(X_p, y_p, node)
        prune(node)
    else:
        node = id3(X, y, attr, impurity_measure)
    return node


def id3(X, y, attr, impurity_measure, parent_node=None):
    label, label_counts = np.unique(y, return_counts=True)
    if len(label) == 1:  # If all data points have the same label
        return Tree(label[0], label=label[0])  # return a leaf with that label
    if len(np.unique(X.astype("<U22"), axis=0)) == 1:  # Else if all data points have identical feature values
        label = label[np.argmax(label_counts)]
        return Tree(label, label=label)  # return a leaf with the most common label
    feature_with_max_ig = max_information_gain(X, y, attr, impurity_measure)
    max_ig_label, max_ig_label_counts = np.unique(X[:, feature_with_max_ig], return_counts=True)
    node = Tree(feature_with_max_ig, parent_node)

    children_labels = []
    for v in max_ig_label:
        ds_rows_indices = np.where(X[:, feature_with_max_ig] == v)
        x_for_this_child, y_for_this_child = X[ds_rows_indices], y[ds_rows_indices]
        new_child = id3(x_for_this_child, y_for_this_child, attr, impurity_measure, parent_node=node)
        new_child.set_split(v)
        node.add_child(new_child)
        children_labels.append(new_child.label)
    node.set_label(max(children_labels, key=children_labels.count))
    return node


def prune(node):
    if node.is_leaf():
        return node.error

    remove_child, error = True, 0
    for child in node.get_children():
        error += prune(child)
        if not child.is_leaf():
            remove_child = False
    if remove_child and error > node.error:
        node.data = node.label
        node.children = []
    return node.error


def add_error(X, y, node):
    if node.label != y:
        node.add_error()
    if node.is_leaf():
        return
    for child in node.get_children():
        if child.split == X[node.data]:
            add_error(X, y, child)


def predict(x, tree):
    if tree.is_leaf():
        return tree.data
    for child in tree.get_children():
        if child.split == x[tree.data]:
            return predict(x, child)
    return tree.label


if __name__ == '__main__':
    ds = pd.read_csv('magic04.data', header=None)

    train, test = train_test_split(ds, 0.7, True)
    (X_train, y_train) = matrix_vector_split(train)
    (X_test, y_test) = matrix_vector_split(test)

    start_time = datetime.now()
    tree_entropy_with_pruning = learn(X_train, y_train, "entropy", shuffle=True, pruning=True)
    tree_gini_with_pruning = learn(X_train, y_train, "gini", shuffle=True, pruning=True)

    tree_entropy = learn(X_train, y_train, "entropy", shuffle=True)
    tree_gini = learn(X_train, y_train, "gini", shuffle=True)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    print("\nEntropy with pruning:", tree_entropy_with_pruning.tree_accuracy(X_test, y_test))
    print("Gini with pruning:", tree_gini_with_pruning.tree_accuracy(X_test, y_test))

    print("\nEntropy:", tree_entropy.tree_accuracy(X_test, y_test))
    print("Gini:", tree_gini.tree_accuracy(X_test, y_test), "\n")

    for crit in ["entropy", "gini"]:
        clf = DecisionTreeClassifier(criterion=crit)
        clf.fit(X_train, y_train)

        correct_predictions, total_number_of_predictions = 0, 0
        for i in range(len(X_test)):
            predicted_class_label = clf.predict([X_test[i]])
            if y_test[i] == predicted_class_label:
                correct_predictions += 1
            total_number_of_predictions += 1
            
        print(f"sklearn {crit}:", correct_predictions / total_number_of_predictions)
