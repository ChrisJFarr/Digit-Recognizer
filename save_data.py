import pandas as pd
import numpy as np
from keras.utils import np_utils
import random

"""Transform Train Data"""

# Load in train data
train = pd.read_csv("train.csv")

# Convert into arrays (28, 28)
train_data = np.array([val.reshape(28, 28, 1) for val in train.iloc[:, 1:].values])
train_labels = np_utils.to_categorical(train.iloc[:, 0].values)

# Shuffle dataset
c = list(zip(train_data, train_labels))
random.shuffle(c)
train_data, train_labels = zip(*c)
del c
train_data, train_labels = np.array(train_data), np.array(train_labels)

# Create train/test/valid from train data
test_size = int(len(train_data) * .1)
valid_size = int(len(train_data) * .1)
train_start = test_size + valid_size
x_train, y_train = train_data[train_start:], train_labels[train_start:]
x_test, y_test = train_data[valid_size:train_start], train_labels[valid_size:train_start]
x_valid, y_valid = train_data[:valid_size], train_labels[:valid_size]

# Save numpy data
np.save("train_data/x_train", x_train)
np.save("train_data/y_train", y_train)
np.save("train_data/x_test", x_test)
np.save("train_data/y_test", y_test)
np.save("train_data/x_valid", x_valid)
np.save("train_data/y_valid", y_valid)


"""Save Test Data"""
# Save test data for submission
test = pd.read_csv("test.csv")

# Convert to 28 by 28 arrays
test_data = np.array([val.reshape(28, 28, 1) for val in test.values])

# Save data
np.save("test_data/test_data", test_data)
