import reports as rp
import assistent as at
import heuristics as hr

PLAYERS = 3
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
            sample = rp.get_target(hr.shift_sample(rp.get_sample(reports, PLAYERS), PLAYERS), PLAYERS)
            #sample = hr.drop_numbers(hr.add_player_index(sample, PLAYERS), PLAYERS)
            rp.write_sample(sample, 'sample/sample.xls')
        else:
            sample = rp.read_sample('sample/sample.xls')

        #print(sample)
        print('Classes cnt -')
        print(rp.get_class_cnt(sample))

        X_train, X_test, y_train, y_test = at.split_sample(sample)
        X_train_scaler = at.get_scaler_X(X_train)
        X_test_scaler = at.get_scaler_X(X_test)
        
        kneighbors = at.fit_by_kneighbors(X_train_scaler, y_train)
        logit = at.fit_by_logit(X_train_scaler, y_train)
        svm = at.fit_by_svm(X_train_scaler, y_train)
        mlp = at.fit_by_mlp(X_train_scaler, y_train)
        rforest = at.fit_by_rforest(X_train, y_train)

        at.print_report(X_test_scaler, y_test, kneighbors)
        at.print_report(X_test_scaler, y_test, logit)
        at.print_report(X_test_scaler, y_test, svm)
        at.print_report(X_test_scaler, y_test, mlp)
        at.print_report(X_test, y_test, rforest)
