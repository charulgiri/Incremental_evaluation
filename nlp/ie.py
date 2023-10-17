import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from datasets import load_dataset

_LOGGER = logging.getLogger(__name__)

def preprocess_tweeteval(data):
    temp_x = []
    temp_y = []
    for d in data:
        temp_x.append(d["text"])
        temp_y.append(d["label"])
    temp_y = np.array(temp_y)
    return temp_x, temp_y

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=5000, type=int)
    parser.add_argument("--T", default=4000, type=int)
    parser.add_argument("--s", default=6.0, type=float)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--clause_drop_p", default=0.75, type=float)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--imdb-num-words", default=5000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    parser.add_argument("--incremental", default=True, type=bool)

    args = parser.parse_args()

    _LOGGER.info("Preparing dataset")
    dataset_name = "imdb"

    if dataset_name == "imdb":
        train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
        train_x, train_y = train
        test_x, test_y = test

        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        _LOGGER.info("Preparing dataset.... Done!")

        _LOGGER.info("Producing bit representation...")

        id_to_word = {value: key for key, value in word_to_id.items()}

        training_documents = []
        for i in range(train_y.shape[0]):
            terms = []
            for word_id in train_x[i]:
                terms.append(id_to_word[word_id].lower())

            training_documents.append(terms)

        testing_documents = []
        for i in range(test_y.shape[0]):
            terms = []
            for word_id in test_x[i]:
                terms.append(id_to_word[word_id].lower())

            testing_documents.append(terms)

    elif dataset_name == "tweets":
        training_documents = load_dataset("tweet_eval", "emoji", split="train")
        testing_documents = load_dataset("tweet_eval", "emoji", split="test")
        training_documents, train_y = preprocess_tweeteval(training_documents)
        testing_documents, test_y = preprocess_tweeteval(testing_documents)
    
    elif dataset_name=="mr":
        training_documents = load_dataset("rotten_tomatoes", split="train")
        testing_documents =  load_dataset("rotten_tomatoes", split="test")
        training_documents, train_y = preprocess_tweeteval(training_documents)
        testing_documents, test_y = preprocess_tweeteval(testing_documents)


    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    _LOGGER.info("Selecting Features....")

    if dataset_name =="imdb":
        SKB = SelectKBest(chi2, k=args.features)
    else:
        SKB = SelectKBest(chi2, k="all")
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train).toarray()
    X_test = SKB.transform(X_test).toarray()

    _LOGGER.info("Selecting Features.... Done!")

    # cluases = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    clauses = [10000]
    time_taken = {}
    _LOGGER.info(f"Incremental: {args.incremental}, Dataset: {dataset_name}")
    for c in clauses:
        tm = TMClassifier(c, args.T, args.s, platform=args.device, incremental=args.incremental, weighted_clauses=args.weighted_clauses,
                        clause_drop_p=args.clause_drop_p)

        _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)

            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()

            _LOGGER.info(f"Clauses:{c}, Inference Time: {benchmark2.elapsed():.2f}s")
        time_taken[f"{c//1000}"]=round((benchmark2.elapsed()), 2)
_LOGGER.info(f"Incremental: {args.incremental}, Dataset: {dataset_name}")
print(time_taken)