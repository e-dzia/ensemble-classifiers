import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class NoKfold:
    def __init__(self, n_splits, stratify=True):
        self.test_size = 1-(n_splits-1)/n_splits
        self.stratify = stratify

    def split(self, X, y):
        data = pd.merge(X, y, left_index=True, right_index=True)

        if self.stratify:
            train_set, test_set = train_test_split(data, test_size=self.test_size, stratify=data['class'])
        else:
            train_set, test_set = train_test_split(data, test_size=self.test_size)

        train_index = train_set.index.values
        test_index = test_set.index.values
        yield train_index, test_index
