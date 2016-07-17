import os
import luigi

from features import group_means, latent_features
import numpy as np
import pandas

def features():
    res = []
    for task in group_means.GeneralMeans.all_tasks():
        res.append(group_means.GroupFnQuery(','.join(sorted(task))))
    # res.append(latent_features.LatentProductTypeQuery(0))

    return res

def make_train_batch(ix=None):
    if ix is None:
        ix = np.random.randint(100)
    file_path = '/tmp/split_data/{}/train/{}.csv'.format(0, ix)
    data = pandas.read_csv(file_path).drop(['id'], 1)
    for feature in features():
        feature.compute(data, inplace=True)

    return data


def make_test_batch(ix=None):
    if ix is None:
        ix = np.random.randint(100)
    file_path = '/tmp/split_data/{}/test/{}.csv'.format(0, ix)
    data = pandas.read_csv(file_path).drop(['id'], 1)
    for feature in features():
        feature.compute(data, inplace=True)

    return data


class Features(luigi.Task):
    rand_round = luigi.IntParameter()

    def requires(self):
        for i in range(100):
            yield FeatureFile(rand_round=self.rand_round, ix=i)

class FeatureFile(luigi.Task):
    rand_round = luigi.IntParameter()
    ix = luigi.IntParameter()

    def requires(self):
        # We actually run through these all because we can
        # use them as the feature list later on.
        for feature in group_means.GeneralMeans().requires():
            yield feature
        #yield latent_features.LatentProductTypeQuery(self.rand_round)

    def output(self):
        return luigi.file.LocalTarget('/tmp/features/{}/{}.DONE'.format(self.rand_round, self.ix))

    def run(self):
        train_path = '/tmp/split_data/{}/train/{}.csv'.format(self.rand_round, self.ix)
        train_data = pandas.read_csv(train_path)
        train_id = train_data.id
        train_ad = train_data.adjusted_demand
        train_data = train_data.drop(['adjusted_demand', 'id'], 1)

        test_path = '/tmp/split_data/{}/test/{}.csv'.format(self.rand_round, self.ix)
        test_data = pandas.read_csv(test_path)
        test_id = test_data.id
        test_ad = test_data.adjusted_demand
        test_data = test_data.drop(['adjusted_demand', 'id'], 1)

        score_path = '/tmp/split_data/{}/score/{}.csv'.format(self.rand_round, self.ix)
        score_data = pandas.read_csv(score_path)
        score_id = score_data.id
        score_ad = score_data.adjusted_demand
        score_data = score_data.drop(['adjusted_demand', 'id'], 1)

        for feature in self.requires():
            train_data = feature.compute(train_data)
            test_data = feature.compute(test_data)
            score_data = feature.compute(score_data)

        train_data['id'] = train_id
        test_data['id'] = test_id
        score_data['id'] = score_id
        train_data['adjusted_demand'] = train_ad
        test_data['adjusted_demand'] = test_ad
        score_data['adjusted_demand'] = score_ad

        os.makedirs('/tmp/features/{}/train/'.format(self.rand_round), exist_ok=True)
        os.makedirs('/tmp/features/{}/test/'.format(self.rand_round) , exist_ok=True)
        os.makedirs('/tmp/features/{}/score/'.format(self.rand_round), exist_ok=True)

        with open('/tmp/features/{}/train/{}.msgpack'.format(self.rand_round, self.ix), 'wb') as f:
            train_data.to_msgpack(f)
        with open('/tmp/features/{}/test/{}.msgpack'.format(self.rand_round, self.ix), 'wb') as f:
            test_data.to_msgpack(f)
        with open('/tmp/features/{}/score/{}.msgpack'.format(self.rand_round, self.ix), 'wb') as f:
            score_data.to_msgpack(f)
        # Touch the file.
        with self.output().open('w'):
            pass
