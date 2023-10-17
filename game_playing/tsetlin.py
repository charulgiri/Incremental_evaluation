import argparse
import logging
import numpy as np
from sklearn.feature_selection import chi2
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import random
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    path="../Datasets/Hex/dataset270k.txt"
    with open(path) as file:
        data=file.readlines()
    data=[f.strip().split(" ") for f in data]
    X=[list(f[0]) for f in data]
    X=[list(map(int, f)) for f in X]
    # X=[f[:0] for f in X]
    Y=[1 if f[1]=='w' else 0 for f in data ]

    tempData = list(zip(X, Y))
    random.shuffle(tempData)
    X, Y = zip(*tempData)
    return X,Y

def load_go_data():
    go_data = pd.read_csv("Winner Prediction/Go Winner Prediction/Go_binary_data_9x9.csv", delimiter=",")
    data = go_data.iloc[:,0]
    labels = go_data.iloc[:,1]
    Y=[int(f) for f in labels]
    go_string = []
    for x in data:
        black = [1 if bit == "1" else 0 for bit in x]
        white = [1 if bit == "2" else 0 for bit in x]
        go_string.append( black+white)

    tempData = list(zip(go_string, Y))
    random.shuffle(tempData)
    X, Y = zip(*tempData)
    return X,Y
    
#train-test split
def data_split(X, Y, t=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=t, random_state=42)
    X_train=np.array(X_train)
    X_test=np.asarray(X_test)
    y_train=np.asarray(y_train)
    y_test=np.asarray(y_test)
    return X_train, y_train, X_test, y_test

_LOGGER = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--num_clauses", default=10000, type=int)
parser.add_argument("--T", default=7000, type=int)
parser.add_argument("--s", default=50.0, type=float)
parser.add_argument("--device", default="CPU", type=str)
parser.add_argument("--weighted_clauses", default=True, type=bool)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--clause_drop_p", default=0.0, type=float)
parser.add_argument("--incremental", default=True, type=bool)

args = parser.parse_args()
dataset_name = "go"
if dataset_name == "go":
    X, Y = load_go_data()
else:
    X, Y = load_data()
X_train, Y_train, X_test, Y_test = data_split(X,Y)

# cluases = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
clauses = [10000]
time_taken = {}
_LOGGER.info(f"Incremental: {args.incremental}, Dataset: {dataset_name}")
for c in clauses:
    T = c*0.7
    tm = TMClassifier(args.num_clauses, args.T, args.s, platform=args.device, incremental=args.incremental, weighted_clauses=args.weighted_clauses,
                        clause_drop_p=args.clause_drop_p)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train)

        batches = X_test.shape[0]//1000
        batches = 1
        init_batch = 1000
        for batch in range(batches):
            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()
            _LOGGER.info(f"Incremental: {args.incremental}, Data:{X_test.shape[0]} Clauses: {c}, T:{T}, s: {args.s} ")
            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                            f"Testing Time: {benchmark2.elapsed():.2f}s")
            time_taken[f"{init_batch//1000}"]=(benchmark2.elapsed())
            init_batch+=1000
_LOGGER.info(f"Incremental: {args.incremental}, Dataset: {dataset_name}")
print(time_taken)