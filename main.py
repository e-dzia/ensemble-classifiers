import datetime
import random
import warnings

from imad_5.plotter import plot_all_tests
from imad_5.utils import main_single
from imad_5.tests import main_tests


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    random.seed(datetime.datetime.now())
    show_mode = True
    filenames = ['iris.csv', 'pima-diabetes.csv', 'glass.csv', 'wine.csv']
    estimators_numbers = [2, 5, 10, 15, 20, 25, 30, 50, 100, 200, 300]
    max_features_pcts = [0.1, 0.25, 0.5, 0.75, 1.0]
    models = ['random_forest', 'boosting', 'bagging', 'tree']
    splits_sizes = [2, 3, 5, 10]
    stratified = [False, True, None]

    filename = filenames[0]
    splits = 10
    stratify = True
    n_estimators = estimators_numbers[7]
    max_features_pct = max_features_pcts[1]
    model = models[2]

    # main_single('files/' + filename, show_mode, n_estimators=n_estimators, splits=splits, stratified=stratify,
    #            model=model, max_features_pct=max_features_pct)
    outfiles = main_tests(1, filenames, estimators_numbers, max_features_pcts, models, splits_sizes, stratified)

    for filename in outfiles:
        plot_all_tests(filename)
