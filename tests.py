import csv
import datetime
import numpy as np

from imad_5.utils import main_single


def main_single_tests(num_of_tests, filename, show_mode=False, n_estimators=100, criterion="gini", splits=5, stratified=True,
                      model="random_forest", max_features_pct=0.5):
    accs = []
    prs = []
    recs = []
    f1s = []
    for _ in range(num_of_tests):
        acc, pr, rec, f1 = main_single(filename, show_mode, n_estimators=n_estimators, splits=splits,
                                       stratified=stratified, model=model, max_features_pct=max_features_pct)
        accs.append(acc)
        prs.append(pr)
        recs.append(rec)
        f1s.append(f1)

    acc = np.mean(accs)
    pr = np.mean(prs)
    rec = np.mean(recs)
    f1 = np.mean(f1s)
    var_acc = np.sqrt(np.var(accs))
    var_pr = np.sqrt(np.var(prs))
    var_rec = np.sqrt(np.var(recs))
    var_f1 = np.sqrt(np.var(f1s))

    return acc, pr, rec, f1, var_acc, var_pr, var_rec, var_f1


def main_tests(num_of_tests, filenames, estimators_numbers, max_features_pcts, models, splits_sizes, stratified, show_mode=False):
    start = datetime.datetime.now()
    print("time start: {}".format(start))
    outfiles = []

    for filename in filenames:
        file = 'results/res-{}-{}-{}.csv'.format(start, filename.split(".")[0], num_of_tests).\
            replace(' ', '_').replace(':', '.')
        outfiles.append(file)
        f = open(file, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(['file', 'tests', 'stratified', 'splits', 'model', 'n_estimators', 'max_features_pct',
                         'acc', 'pr', 'rec', 'f1', 'acc_std', 'pr_std', 'rec_std', 'f1_std'])

        splits = 10
        stratify = True
        n_estimators = 100
        max_features_pct = max_features_pcts[-1]
        model = models[0]

        for model in models:
            for stratify in stratified:
                for splits in splits_sizes:
                    results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                                n_estimators=n_estimators, splits=splits,
                                                stratified=stratified, model=model, max_features_pct=max_features_pct)
                    data = [filename, 'crossvalidation_{}'.format(model),
                            'none' if stratify is None else 'strat' if stratify else 'not',
                            splits, model, n_estimators, max_features_pct]
                    data.extend(results)
                    writer.writerow(data)
                    if show_mode:
                        print(data)

        stratify = True
        splits = 10
        n_estimators = 100
        max_features_pct = max_features_pcts[-1]
        model = models[0]

        for model in models:
            for n_estimators in estimators_numbers:
                results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                            n_estimators=n_estimators, splits=splits,
                                            stratified=stratified, model=model, max_features_pct=max_features_pct)
                data = [filename, 'n_estimators', 'strat' if stratify else 'not',
                        splits, model, n_estimators, max_features_pct]
                data.extend(results)
                writer.writerow(data)
                if show_mode:
                    print(data)

        stratify = True
        splits = 10
        n_estimators = 100
        max_features_pct = max_features_pcts[-1]
        model = models[0]

        for model in models:
            for max_features_pct in max_features_pcts:
                results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                            n_estimators=n_estimators, splits=splits,
                                            stratified=stratified, model=model, max_features_pct=max_features_pct)
                data = [filename, 'max_features_pct', 'strat' if stratify else 'not',
                        splits, model, n_estimators, max_features_pct]
                data.extend(results)
                writer.writerow(data)
                if show_mode:
                    print(data)

        end = datetime.datetime.now()
        print("done, time: {}, elapsed: {}".format(end, end - start))
        f.close()
    return outfiles
