import reports as rp
import assistent as at
import heuristics as hr

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
            rp.write_sample(sample, 'sample/sample.xls')
        else:
            sample = rp.read_sample('sample/sample.xls')

        X_train, X_test, y_train, y_test = at.split_sample(sample)

        X_train['Replace'] = y_train
        X_train = rp.get_target(hr.shift_sample(X_train, hr.get_cicle_shifts(PLAYERS)), PLAYERS)
        y_train = X_train['Replace']
        X_train = hr.drop_features(X_train, 'Replace')

        print('Sample classes cnt -')
        print(rp.get_class_cnt(sample['Replace']))
        print('Train classes cnt -')
        print(rp.get_class_cnt(y_train))
        print('Test classes cnt -')
        print(rp.get_class_cnt(y_test))

        X_train_scaler = at.get_scaler_X(X_train)
        X_test_scaler = at.get_scaler_X(X_test)
        
        kneighbors = at.fit_by_kneighbors(X_train_scaler, y_train)
        svm = at.fit_by_svm(X_train_scaler, y_train)
        mlp = at.fit_by_mlp(X_train_scaler, y_train)
        rforest = at.fit_by_rforest(X_train, y_train)

        at.print_report(X_test_scaler, y_test, kneighbors)
        at.print_report(X_test_scaler, y_test, svm)
        at.print_report(X_test_scaler, y_test, mlp)
        at.print_report(X_test, y_test, rforest)

        at.write_report(X_test_scaler, y_test, kneighbors)
        at.write_report(X_test_scaler, y_test, svm)
        at.write_report(X_test_scaler, y_test, mlp)
        at.write_report(X_test, y_test, rforest)
