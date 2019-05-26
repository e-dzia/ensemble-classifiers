import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_file(filename):
    dataset = pandas.read_csv(filename)
    return dataset


def plot_tests(df, test, filename, dir_name="plots"):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    test_name = test[0]
    test_groupby = test[1]
    test_x = test[2]

    scores = ['f1']  # 'acc', 'pr', 'rec',

    df = df.query("tests == '{}'".format(test_name))

    for score in scores:
        fig, ax = plt.subplots()
        for key, grp in df.groupby([test_groupby]):
            ax = grp.plot(ax=ax, kind='line', x=test_x, y=score, label=key, style='.-',
                          title='Miara {}, testy {}'.format(score, test_name))
            #ax.set_ylim(0, 1)
        #plt.show()
        plt.savefig('{}/{}-{}-{}.png'.format(dir_name, filename, test_name, score))
        plt.clf()


def plot_all_tests(filename):
    dir_name = "plots/{}".format(filename).replace("/results", "").replace(".csv", "")
    df = load_file(filename)
    models = ['random_forest', 'boosting', 'bagging', 'base_trees']
    tests = [('n_estimators', 'model', 'n_estimators'),
             ('max_features_pct', 'model', 'max_features_pct')]
    for model in models:
        tests.append(('crossvalidation_{}'.format(model), 'stratified', 'splits'))
    filename_base = filename.split('/')
    filename_base = filename_base[1].split('.csv')
    for test in tests:
        plot_tests(df, test, filename_base[0], dir_name=dir_name)


if __name__ == "__main__":
    plot_all_tests('results/res-2019-05-25_22.09.43.407925-iris-1.csv')
