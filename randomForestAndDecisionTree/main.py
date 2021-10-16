#!/usr/bin/env python
import os
import pickle
from pathlib import Path
import pickle as pkl
import numpy as np
from decision_stump import DecisionStumpInfoGain
from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomForest, RandomTree


def load_dataset(filename):
    with open(Path(".", "data", filename), "rb") as f:
        return pickle.load(f)

def evaluate_model(X, y, X_test, y_test, model):
	model.fit(X, y)
	y_pred = model.predict(X)
	tr_error = np.mean(y_pred != y)
	y_pred = model.predict(X_test)
	te_error = np.mean(y_pred != y_test)
	print(f"    Training error: {tr_error:.3f}")
	print(f"    Testing error: {te_error:.3f}")

def main():
	dataset = load_dataset("vowel.pkl")
	X = dataset["X"]
	y = dataset["y"]
	X_test = dataset["Xtest"]
	y_test = dataset["ytest"]
	
	print("Decision tree info gain")
	evaluate_model(X, y, X_test, y_test, DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

	print("Random Forest info gain")
	evaluate_model(X, y, X_test, y_test, RandomForest(max_depth=np.inf, num_trees = 50))



if __name__ == "__main__":
    main()
