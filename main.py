import reports as rp
import assistent as at
import heuristics as hr
import pandas as pd

PLAYERS = 5
MATCHES = 24
PARTS = 3

# Главный класс
class Main:
    def run(self):
        print('Run app')

        print('Use sample or create: 1 - use, other - create')

        if input() != '1':
            reports = rp.read_reports(MATCHES, PARTS, 'reports/reception/')
            
            #sample = rp.get_target(rp.get_sample(reports, PLAYERS), PLAYERS)
            sample = rp.get_target(hr.shift_sample(rp.get_sample(reports, PLAYERS), hr.get_random_shift(PLAYERS)), PLAYERS)
            write_sample(sample, 'sample/sample.xls')
        else:
            sample = rp.read_sample('sample/sample.xls')

        X_train, X_test, y_train, y_test = at.split_sample(sample)

        #X_train['Replace'] = y_train
        #X_train = rp.get_target(hr.shift_sample(X_train, hr.get_cicle_shifts(PLAYERS)), PLAYERS)
        #y_train = X_train['Replace']
        #X_train = hr.drop_features(X_train, 'Replace')

        models = get_models(X_train, 
                            y_train, 
                            kneighbors=at.fit_by_kneighbors, 
                            svm=at.fit_by_svm, 
                            mlp=at.fit_by_mlp, 
                            rforest=at.fit_by_rforest)

        write_fit_reports(get_fit_reports(X_test, y_test, models))

# Проитерировать модели по ключам
def models_iteration(models: dict, f, X, y) -> dict:
    result = {}
    X_scaler = at.get_scaler_X(X)

    for key in models:
        if key != 'rforest':
            result[key] = f(models, key, X_scaler, y)
        else:
            result[key] = f(models, key, X, y)

    return result

# Получить обученные модели по заданным алгоритмам
def get_models(X_train, y_train, **algs):
    return models_iteration(algs, lambda models, key, X, y: models.get(key)(X, y), X_train, y_train)

# Получить отчёты по обучению моделей
def get_fit_report(models, key, X, y):
    return {
        'name': models.get(key).get('name'),
        'best': at.get_best(models.get(key)),
        'classes_report': at.get_classes_report(X, y, models.get(key)),
        'nr_matrix': at.get_nr_matrix(X, y, models.get(key))
    }

def get_fit_reports(X_test, y_test, models):
    return models_iteration(models, get_fit_report, X_test, y_test)

# Записать отчёт по обучению моделей
def write_fit_reports(reports):
    writer = pd.ExcelWriter('fit_reports/report.xls', engine='openpyxl')

    for key in reports:
        report = reports.get(key)
        report.get('best').to_excel(writer, sheet_name='Best - ' + report.get('name'), index=False)
        report.get('classes_report').to_excel(writer, sheet_name='Classes - ' + report.get('name'), index=True)
        report.get('nr_matrix').to_excel(writer, sheet_name='NR matrix - ' + report.get('name'), index=True)
        
        #for j in range(len(report.get('classes_report')) + 1):
            #writer.sheets['Best - ' + report.get('name')].column_dimensions[chr(j + 65)].width = 14
            #writer.sheets['Classes - ' + report.get('name')].column_dimensions[chr(j + 65)].width = 14

    writer.save()

# Вывести отчёт по обучению моделей
def print_fit_report(models, X_test, y_test):
    X_test_scaler = at.get_scaler_X(X_test)

    for model in models:
        if model != 'rforest':
            at.print_report(X_test_scaler, y_test, models.get(model))
        else:
            at.print_report(X_test, y_test, models.get(model))

# Записать выборку в файл
def write_sample(sample, path):
    writer = pd.ExcelWriter(path, engine='openpyxl')
    sample.to_excel(writer, sheet_name='Sample', index=False)
    writer.save()
