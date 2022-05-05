import reports as rp

# Главный класс
class Main:
    def run(self):
        print("Run app")

        print ("Use sample or create: 1 - use, other - create")

        if input() != "1":
            reports = rp.read_reports(24, 3, "reports/reception/")
            
            sample = rp.get_sample(reports)
            target = rp.get_target(sample)
            rp.write_sample(sample, "sample/sample.xls")
            rp.write_sample(target, "sample/target.xls")
        else:
            sample = rp.read_sample("sample/sample.xls")
            target = rp.read_sample("sample/target.xls")

        #print(sample)
        #print(target)
        #Обучение
