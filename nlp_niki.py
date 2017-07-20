#! /usr/bin/env python
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import KFold


punctuation_stopwords = [
    ",", ":", ";", "'", '"', "'", "/", "-", "+", "&", "(", ")",
    "a", "an", "the", "of", "to", "for"]


def load_trainset(filename, feature_columns, target_column):
    dataset = []
    with open(filename) as fp:
        for line in fp:
            question, label = line.split(' ,,, ')
            dataset.append({
                feature_columns[0]: question.strip(),
                target_column: label.strip()
            })
    return pd.DataFrame(dataset)

def load_testset(filename, feature_columns, target_column):
    dataset = []
    with open(filename) as fp:
        for statement in fp:
            dataset.append({
                feature_columns[0]: statement.strip(),
                target_column: ''
            })
    return pd.DataFrame(dataset)

def feature_extraction(statements):
    features = []
    rare_tokens = []

    row_feature = []
    for line in statements:
        cleaned_tokens = []
        for token in line.split(' '):
            token = token.strip().lower()
            if not token or token in punctuation_stopwords:
                continue
            cleaned_tokens.append(token)
            
            if token not in rare_tokens:
                rare_tokens.append(token)
            elif token not in features:
                features.append(token)
        row_feature.append(cleaned_tokens)
    
    feature_matrix = pd.DataFrame(0, index=np.arange(len(row_feature)), columns=features)
    for i, row in enumerate(row_feature):
        for feature in row:
            if feature in features:
                feature_matrix.loc[i, feature] += 1

    word_count = feature_matrix.sum(axis=0)
    feature_matrix = feature_matrix.loc[:,(word_count>5)&(word_count<1000)]
    return feature_matrix.columns, feature_matrix.as_matrix()

def get_feature_matrix(statements, features):
    row_feature = []
    for line in statements:
        cleaned_tokens = []
        for token in line.split(' '):
            token = token.strip().lower()
            if not token or token in punctuation_stopwords:
                continue
            cleaned_tokens.append(token)            
        row_feature.append(cleaned_tokens)
    
    feature_matrix = pd.DataFrame(0, index=np.arange(len(row_feature)), columns=features)
    for i, row in enumerate(row_feature):
        for feature in row:
            if feature in features:
                feature_matrix.loc[i, feature] += 1
    return feature_matrix.as_matrix()


if __name__ == '__main__':
    # list of file names
    train_file = 'LabelledData (1).txt'
    test_file = 'train_1000.label.txt'
    predicted_file = 'predicted_labels.csv'

    feature_columns = ['question']
    target_column = 'label'
    
    train_dataset = load_trainset(train_file, feature_columns, target_column)
    labels = train_dataset.label.unique().tolist()
    train_dataset["label_num"] = train_dataset.label.apply(labels.index)
    features, feature_matrix = feature_extraction(train_dataset[feature_columns[0]])
    target_label = train_dataset["label_num"]

    classifier = SVC(kernel="linear")
    
    print('Training...')
    kf = KFold(n_splits=3)
    t1 = time.time()
    for train_index, test_index in kf.split(feature_matrix):
        X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
        y_train, y_test = target_label[train_index], target_label[test_index]
        classifier.fit(X_train, y_train)
    print('Done in {} secs'.format(time.time() - t1))
    
    print('Testing...')
    test_dataset = load_testset(test_file, feature_columns, target_column)
    testset_features = get_feature_matrix(test_dataset[feature_columns[0]], features)
    predicted = classifier.predict(testset_features)
    test_dataset[target_column] = [labels[int(p)] for p in predicted]
    test_dataset.to_csv(predicted_file, index=False, header=False)
    print('Predicted labels of "train_1000.label.txt" are in the file "predicted_labels.csv" ')
