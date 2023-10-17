import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import time 

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras


parser = argparse.ArgumentParser()
parser.add_argument("--imdb-num-words", default=5000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
args = parser.parse_args()
train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
X_train, y_train = train
X_test, y_test = test
X_train = pad_sequences(X_train, maxlen=2494)
X_test = pad_sequences(X_test, maxlen=2494)
# train_x = np.asarray(train_x)
# test_x = np.asarray(test_x)

# print(train_x[0])
output_dim = 32
max_input_lenght = X_train.shape[1]
num_classes = np.unique(y_train).shape[0]
max_features = 5000
print(max_input_lenght, num_classes, X_test.shape)
# exit(0)
# ----- Define model -----
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=max_features, output_dim=output_dim, input_length=max_input_lenght))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


# ----- Compile model -----
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["accuracy"])

# ----- Print model -----
# model.summary()

# ----- Train model -----
with tf.device('cpu:0'):
    history_1 = model.fit(X_train, y_train, batch_size=64,epochs=10)

# ----- Evaluate model -----
start_inference = time.time()
with tf.device('cpu:0'):
    probabilities = model.predict(X_test)
end_inference = time.time()
print(f"Model:DNN\tInference time:{end_inference-start_inference:.2f}s")
pred = np.argmax(probabilities, axis=1)
accuracy = accuracy_score(y_test, pred)
print('Accuracy: {:.4f}'.format(accuracy))
