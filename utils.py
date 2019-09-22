import csv
import datetime
import random
import warnings
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import naive_bayes, model_selection, metrics, preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imad_5.no_kfold import NoKfold
from imad_5.printer import show_data


def unpack_data(filename):
    dataset = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        dataset.columns = ["petalLength", "petalWidth", "sepalLength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        dataset.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
        dataset = dataset.set_index('id')

    if filename == 'files/pima-diabetes.csv':
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc",
                           "Age", "class"]

    if filename == 'files/wine.csv':
        dataset.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                           "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue",
                           "OD280/OD315", "Proline"]

    if filename == 'files/customers.csv':
        dataset.columns = ["Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

    return dataset


def preprocess_data(X):
    # columns = X.columns
    # X = pandas.DataFrame(preprocessing.normalize(X), columns=columns)
    return X


def extract_labels(dataset):
    # extract labels
    dataset_labels = dataset["class"].copy()
    dataset = dataset.drop("class", axis=1)

    return dataset, dataset_labels


def split_data(dataset, test_size=0.2):
    # split into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42, stratify=dataset['class'])

    # extract labels
    train_set_labels = train_set["class"].copy()
    train_set = train_set.drop("class", axis=1)

    test_set_labels = test_set["class"].copy()
    test_set = test_set.drop("class", axis=1)

    return train_set, train_set_labels, test_set, test_set_labels


def cross_validation(X, y, kfold, model):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for train, test in kfold.split(X, y):
        train = train.tolist()
        test = test.tolist()
        params = model.get_params(deep=False)
        t = type(model)
        model = t(**params)
        model.fit(X.iloc[train], y.iloc[train])
        labels_predicted = model.predict(X.iloc[test])
        labels_true = y.iloc[test]
        accuracy, precision, recall, f1 = evaluate(labels_true, labels_predicted)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    return accuracy, precision, recall, f1


def evaluate(labels_true, labels_predicted):
    accuracy = metrics.accuracy_score(y_true=labels_true, y_pred=labels_predicted)
    precision = metrics.precision_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    recall = metrics.recall_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    f1 = metrics.f1_score(y_true=labels_true, y_pred=labels_predicted, average='macro')

    return accuracy, precision, recall, f1


def main_single(filename, show_mode, n_estimators=100, criterion="gini", splits=5, stratified=True,
                model="random_forest", max_features_pct=0.5):
    # unpack the data from .csv
    dataset = unpack_data(filename)

    # choose the model
    if model == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                       max_features=max_features_pct)
    elif model == "boosting":
        base = DecisionTreeClassifier()
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=max_features_pct,
                                   base_estimator=base)
    elif model == "bagging":
        base = DecisionTreeClassifier()
        model = BaggingClassifier(n_estimators=n_estimators, base_estimator=base,
                                  n_jobs=-1, max_features=max_features_pct)
    elif model == "tree":
        model = DecisionTreeClassifier()
    else:
        raise Exception("Choose valid model")

    # split the data
    dataset, dataset_labels = extract_labels(dataset)
    dataset = preprocess_data(dataset)

    # TODO: no crossvalidation?
    if stratified is None:
        kfold = NoKfold(n_splits=splits)
    elif stratified:
        kfold = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=random.randint(0, 10000))
    else:
        kfold = model_selection.KFold(n_splits=splits, shuffle=True, random_state=random.randint(0, 10000))

    accuracy, precision, recall, f1 = cross_validation(dataset, dataset_labels, kfold, model)

    if show_mode:
        print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(accuracy, precision, recall, f1))

    return accuracy, precision, recall, f1
