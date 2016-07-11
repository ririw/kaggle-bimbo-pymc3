from features import group_means
import numpy as np
import pandas

def features():
    return [
        group_means.GroupFnQuery('sales_depo',    0, 'mean'),
        group_means.GroupFnQuery('sales_channel', 0, 'mean'),
        group_means.GroupFnQuery('route_id',      0, 'mean'),
        group_means.GroupFnQuery('product_id',    0, 'mean'),
        #group_means.GroupFnQuery('sales_depo',    0, 'median'),
        #group_means.GroupFnQuery('sales_channel', 0, 'median'),
        #group_means.GroupFnQuery('route_id',      0, 'median'),
        #group_means.GroupFnQuery('product_id',    0, 'median'),
        group_means.GroupFnQuery('sales_depo',    0, 'std'),
        group_means.GroupFnQuery('sales_channel', 0, 'std'),
        group_means.GroupFnQuery('route_id',      0, 'std'),
        group_means.GroupFnQuery('product_id',    0, 'std'),
    ]


def make_train_batch(ix=None):
    if ix is None:
        ix = np.random.randint(100)
    file_path = '/tmp/split_data/{}/train/{}.csv'.format(0, ix)
    data = pandas.read_csv(file_path)
    for feature in features():
        feature.compute(data, inplace=True)

    return data.drop(['id'], 1).fillna(-1)


def make_test_batch(ix=None):
    if ix is None:
        ix = np.random.randint(100)
    file_path = '/tmp/split_data/{}/test/{}.csv'.format(0, ix)
    data = pandas.read_csv(file_path)
    for feature in features():
        feature.compute(data, inplace=True)

    return data.drop(['id'], 1).fillna(-1)
