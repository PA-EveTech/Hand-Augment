from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import argparse
from preprocess_data import process_data
import numpy as np

import 

parser = argparse.ArgumentParser(description='EMG data training using an SVM')
parser.add_argument('--dataset_num', metavar='NUM', type=int, default=2,
                    help='which dataset to use (1 or 2)')
parser.add_argument('--gender', metavar='GEN', type=str, default='male',
                    help='which gender in the dataset to use (male or female')
args = parser.parse_args()

assert args.dataset_num in [1, 2], "Dataset number must be either 1 or 2"
assert args.gender in ['male', 'female'], "Dataset gender must be either male or female"

if args.gender == 'male':
    dataset, labels = process_data(args.dataset_num, 1)
else:
    dataset, labels = process_data(args.dataset_num, 0)
train_set, test_set, train_labels, test_labels = train_test_split(dataset, labels, test_size = 0.2)
new_train_set = []
new_test_set = []

for i in range(train_set.shape[0]):
    new_train_set.append(train_set[i].flatten())
for i in range(test_set.shape[0]):
    new_test_set.append(test_set[i].flatten())

new_train_set = np.array(new_train_set)
new_test_set = np.array(new_test_set)

model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model.fit(new_train_set, train_labels)
print(model.score(new_test_set, test_labels))