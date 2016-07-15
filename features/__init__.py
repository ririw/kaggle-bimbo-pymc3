from features import group_means, latent_features
import numpy as np
import pandas

def features():
    res = []
    for task in group_means.GeneralMeans.all_tasks():
        res.append(group_means.GroupFnQuery(','.join(sorted(task)), 0, 'mean'))
    # res.append(latent_features.LatentProductTypeQuery(0))

    return res

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
