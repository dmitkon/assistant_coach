import reports as rp
import assistent as at

# Главный класс
class Main:
    def run(self):
        print("Run app")

        print ("Use sample or create: 1 - use, other - create")

        if input() != "1":
            reports = rp.read_reports(24, 3, "reports/reception/")
            
            sample = rp.get_target(rp.get_sample(reports))
            rp.write_sample(sample, "sample/sample.xls")
        else:
            sample = rp.read_sample("sample/sample.xls")

        #print(sample)
        kneighbors = at.fit_by_kneighbors(sample)
        logit = at.fit_by_logit(sample)
        svm = at.fit_by_svm(sample)
        mlp = at.fit_by_mlp(sample)
        rforest = at.fit_by_rforest(sample)

        at.print_reports(kneighbors, logit, svm, mlp, rforest)
        #at.print_reports(rforest)
        #at.write_report_by_models(kneighbors, logit, svm, mlp)
