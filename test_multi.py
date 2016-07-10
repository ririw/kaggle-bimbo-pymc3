import logging

import coloredlogs
import dask.dataframe as dd
import dask.multiprocessing
import dask
from functools import reduce
import dask.async
import ml_metrics
from sklearn import *

import features

coloredlogs.install(level=logging.INFO)


def merge_ests(e1, e2):
    e1.estimators_ += e2.estimators_
    e1.n_estimators += e2.n_estimators
    return e1


def build_tree(data):
    reg = ensemble.ExtraTreesRegressor(warm_start=True, verbose=0, n_jobs=1, n_estimators=4)
    reg.fit(data.drop('adjusted_demand', 1), data.adjusted_demand)
    return reg


with dask.set_options(get=dask.multiprocessing.get):
    datasets = [dask.delayed(features.make_train_batch)(ix) for ix in range(100)]
    dataset = dd.from_delayed(datasets)
    trees = dataset.map_partitions(build_tree).compute()
tree = reduce(merge_ests, trees)

for ix in range(100):
    data = features.make_test_batch(ix)
    y_pred = tree.predict(data.drop('adjusted_demand', 1))
    y = data.adjusted_demand
    print(ml_metrics.rmsle(y, y_pred))
