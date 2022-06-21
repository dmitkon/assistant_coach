from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from decimal import Decimal

def split_sample(sample):
    return train_test_split(sample.drop(columns=['Replace']), sample['Replace'], test_size=0.33, random_state=42)

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
        'n_neighbors': [3, 5, 9, 13],
        'p': [1, 2, 3, 4],
        'metric': ['minkowski', 'chebyshev', 'cosine'],
        'weights': ['uniform', 'distance']
    }

    return fit_by_alg(X_train, y_train, KNeighborsClassifier, params)

def fit_by_logit(X_train, y_train):
    params = {
        'C': [0.01, 0.1, 0.5, 1]
    }

    return fit_by_alg(X_train, y_train, LogisticRegression, params, cv=5)

def fit_by_svm(X_train, y_train):
    params = {
        'C': [1, 5, 10, 15, 20],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    return fit_by_alg(X_train, y_train, SVC, params, cv=5)

def fit_by_mlp(X_train, y_train):
    params = {
        'random_state': [42],
        'hidden_layer_sizes': [(10,), (15,), (18,), (10, 10), (15, 15), (18, 18)],
        #'solver': ['sgd', 'adam'],
        'max_iter': [100, 150, 200]
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

def get_best(model):
    best_params = model.get('clf').best_params_
    best_params['Best_score'] = model.get('clf').best_score_
    
    best_params = dict(map(lambda a: (a[0], [a[1]]), zip(best_params.keys(), best_params.values())))

    return pd.DataFrame(best_params)

def get_classes_report(X_test, y_test, model):
    predict = get_predict(model.get('clf').best_estimator_, X_test)
    
    clf_report = classification_report(y_test, predict, output_dict=True)
    categories = list(clf_report)
    metrics_names = list(clf_report.get(categories[0]))
    
    report = {}
    
    for j, metric in enumerate(metrics_names):
        metric_list = []
        
        for categ in categories:
            if isinstance(clf_report.get(categ), dict):
                metric_list.append(float(Decimal(clf_report.get(categ).get(metric)).quantize(Decimal('1.00'))))
            elif j == 0:
                metric_list.append(float(Decimal(clf_report.get(categ)).quantize(Decimal('1.00'))))
            else:
                metric_list.append(np.NaN)
        report[metric] = metric_list

    return pd.DataFrame(report, index=categories)

def get_matrix_display(X_test, y_test, model):
    predict = get_predict(model.get('clf').best_estimator_, X_test)

    matrix = confusion_matrix(y_test, predict, labels=model.get('clf').classes_)

    return ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.get('clf').classes_)

def get_nr_matrix_display(X_test, y_test, model):
    predict = get_predict(model.get('clf').best_estimator_, X_test)

    matrix = multilabel_confusion_matrix(y_test, predict)[-1]
    
    return ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Neg', 'Pos'])

# Подсчитать кол-во меток классов
def get_labels_cnt(target):
    classes = range(1, target.max() + 1)
    df = pd.DataFrame()
    
    for label in classes:
        df[label] = [target[target == label].shape[0]]

    return df
