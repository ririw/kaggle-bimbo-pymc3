import glob
import gzip
import logging
import os
import pickle
import shutil
import tempfile
import warnings
from multiprocessing.pool import Pool

import coloredlogs
import dask.dataframe as dd
import dask.multiprocessing
import dask.bag
import ml_metrics
import numpy as np
import pandas

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn import *

import features

coloredlogs.install(level=logging.INFO)


def build_tree(data):
    reg = ensemble.ExtraTreesRegressor(verbose=0, max_depth=7,
                                       n_jobs=-1, n_estimators=4)
    X = data.drop(['week_num', 'adjusted_demand', 'rand'], 1)
    y = data.adjusted_demand
    reg.fit(X, y)
    if np.random.uniform() < 0.25:
        print(pandas.Series(reg.feature_importances_, index=X.columns).sort_values())
    fn = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/intermediate_trees/', suffix='.pkl.gzip')
    with gzip.open(fn.name, 'wb') as f:
        logging.info('Persisting to ' + fn.name)
        pickle.dump(reg, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fn.name


def retrieve_tree(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def parallel_predict(fn):
    tree = retrieve_tree(fn)
    ys = []
    for i in range(8):
        data = features.make_test_batch(i)
        X = data.drop(['week_num', 'adjusted_demand', 'rand'], 1)
        ys.append(tree.predict(X))
    y = np.concatenate(ys)
    return y

os.makedirs('/tmp/intermediate_trees/', exist_ok=True)
shutil.rmtree('/tmp/intermediate_trees')
os.makedirs('/tmp/intermediate_trees/')
with dask.set_options(get=dask.multiprocessing.get, pool=Pool(2)):
    if True:
        datasets = [dask.delayed(features.make_train_batch)(ix) for ix in range(8)]
        dataset = dd.from_delayed(datasets)
        tree_files = dataset.map_partitions(build_tree).compute()
    else:
        tree_files = glob.glob('/tmp/intermediate_trees/*')

    y_sum = dask.bag.from_sequence(tree_files, npartitions=4).map(parallel_predict).compute()
    y_pred = np.vstack(y_sum).mean(0)

    y = dd.from_delayed([dask.delayed(features.make_test_batch)(ix) for ix in range(8)])
    y = y.adjusted_demand.compute()

    print(ml_metrics.rmsle(y, y_pred))
