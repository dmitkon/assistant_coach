from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def split_sample(sample):
    return train_test_split(sample.drop(columns=['Replace']), sample['Replace'], test_size=0.20, random_state=42)

def get_scaler_sample(sample_X):
    scaler = StandardScaler()
    scaler.fit(sample_X)
    
    return scaler.transform(sample_X)

def fit_by_alg(sample, alg, parameters, cv=None, scaler=True):
    X_train, X_test, y_train, y_test = split_sample(sample)
    
    if scaler:
        X_train = get_scaler_sample(X_train)
        X_test = get_scaler_sample(X_test)

    model = alg()
    clf = GridSearchCV(model, parameters, cv=cv)
    clf.fit(X_train, y_train)
    
    model = clf.best_estimator_
    model.fit(X_train, y_train)
    
    return {'model': model,
            'name': str(alg)[8:-2].split('.')[-1],
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'test_report': classification_report(y_test, get_predict(model, X_test))}

def fit_by_kneighbors(sample):
    return fit_by_alg(sample, KNeighborsClassifier, {'n_neighbors': [3, 5, 9, 13]})

def fit_by_mlp(sample):
    return fit_by_alg(sample, MLPClassifier, {'hidden_layer_sizes': [(10,), (15,), (18,), (10, 10), (15, 15), (18, 18)]}, cv=5)

def fit_by_logit(sample):
    return fit_by_alg(sample, LogisticRegression, {'C': [0.01, 0.1, 0.5, 1]}, cv=5)

def fit_by_svm(sample):
    return fit_by_alg(sample, SVC, {'C': [0.5, 1, 10, 15], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}, cv=5)

def get_predict(model, vectors):
    return model.predict(vectors)

def get_report_by_models(models):
    return None
