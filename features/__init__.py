from features import group_means, latent_features
import numpy as np
import pandas



def features():
    res = []
    names = ['sales_depo',
             'sales_channel',
             'route_id',
             'product_id']
    fns = ['mean', 'std']
    for name in names:
        for fn in fns:
            res.append(group_means.GroupFnQuery(name, 0, fn))
            for othername in names:
                if name != othername:
                    res.append(group_means.GroupFnQuery(','.join([name, othername]), 0, fn))

    res.append(latent_features.LatentProductTypeQuery(0))
    res.append(group_means.GroupFnQuery('client_id', 0, 'mean'))

    print(res)
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
