from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def split_sample(sample):
    return train_test_split(sample.drop(columns=['Replace']), sample['Replace'], test_size=0.20, random_state=42)

def get_scaler_X(X_sample):
    scaler = StandardScaler()
    scaler.fit(X_sample)
    
    return scaler.transform(X_sample)

def fit_by_alg(X_train, y_train, alg, parameters, cv=None):
    average = 'weighted'
    scoring = {
        'accuracy': 'accuracy',
        'precision': f'precision_{average}',
        'recall': f'recall_{average}',
        'f1': f'f1_{average}',
    }

    clf = GridSearchCV(alg(), parameters, cv=cv, scoring=scoring, refit='accuracy')
    clf.fit(X_train, y_train)
    
    return {
            'name': str(alg)[8:-2].split('.')[-1],
            'clf': clf
    }

def fit_by_kneighbors(X_train, y_train):
    params = {
        'n_neighbors': [3, 5, 9, 13]
    }

    return fit_by_alg(X_train, y_train, KNeighborsClassifier, params)

def fit_by_logit(X_train, y_train):
    params = {
        'C': [0.01, 0.1, 0.5, 1]
    }

    return fit_by_alg(X_train, y_train, LogisticRegression, params, cv=5)

def fit_by_svm(X_train, y_train):
    params = {
        'C': [0.5, 1, 10, 15],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    return fit_by_alg(X_train, y_train, SVC, params, cv=5)

def fit_by_mlp(X_train, y_train):
    params = {
        'random_state': [42],
        'hidden_layer_sizes': [(10,), (15,), (18,), (10, 10), (15, 15), (18, 18)],
        #'solver': ['sgd', 'adam'],
        #'max_iter': [50, 100, 150, 200]
    }

    return fit_by_alg(X_train, y_train, MLPClassifier, params, cv=5)

def fit_by_rforest(X_train, y_train):
    params = {
        'random_state': [42],
        'n_estimators': [60, 80, 100, 120, 140, 160, 180, 200],
        'criterion': ['gini', 'entropy']
    }

    return fit_by_alg(X_train, y_train, RandomForestClassifier, params, cv=5)

def get_predict(model, vectors):
    return model.predict(vectors)

def print_report(X_test, y_test, model):
    predict = get_predict(model.get('clf').best_estimator_, X_test)

    print(model.get('name') + ':')
    print('best_score - ' + str(model.get('clf').best_score_))
    print('best_params - ' + str(model.get('clf').best_params_))
    print('test -\n' + classification_report(y_test, predict))
    print('Confusion matix for "no replacement" class -')
    print(pd.DataFrame(multilabel_confusion_matrix(y_test, predict)[-1],
                        columns=['pred_neg', 'pred_pos'],
                        index=['neg', 'pos']))
