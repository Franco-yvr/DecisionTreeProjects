from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
from scipy import stats

def mode(y):
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]
        DecisionTree.fit(self, bootstrap_X, bootstrap_y)

    def predict(self, X):
        return DecisionTree.predict(self,X)


class RandomForest:

    max_depth = None
    num_trees = None
    forest = None

    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.forest = []

    def fit(self, X, y):
        for t in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y)
            self.forest.append(tree)

    def predict(self, X):
        pred_num = X.shape[0]
        pred_table = np.zeros((self.num_trees, pred_num))
        for index, tree in enumerate(self.forest):
            pred = tree.predict(X)
            pred_table[index,:] = pred

        y_pred = np.zeros(pred_num)
        for p in range(pred_num):
            y_pred[p] = mode(pred_table[:,p])

        return y_pred
