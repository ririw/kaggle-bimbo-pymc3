import itertools
import pickle
import sqlite3

import luigi
import numpy as np
import pandas
import sklearn.linear_model
import sklearn.dummy

from features import base_data


class SqlGroupFunction(luigi.Task):
    group_names = luigi.Parameter()

    def output(self):
        grp_name = '-'.join(sorted(self.group_names.split(',')))
        return luigi.file.LocalTarget(path='/tmp/group_stats/group_{}.msgpack'.format(grp_name))

    def run(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        con.create_aggregate("log1pMean", 1, Log1PMean)
        con.create_function("expm1", 1, np.expm1)
        query = """
            SELECT {},
                   expm1(log1pMean(adjusted_demand)) as adjusted_demand
              FROM data
             WHERE adjusted_demand is not NULL
                   and week_num < 8
          GROUP BY {}
        """.format(self.group_names, self.group_names)
        res = pandas.read_sql(query, con=con)
        self.output().makedirs()
        with open(self.output().path, 'wb') as f:
            res.to_msgpack(f, compress='zlib')


class MissingValueImputation(luigi.Task):
    group_names = luigi.Parameter()

    def output(self):
        groups = self.group_names.split(',')
        if len(groups) == 1:
            return luigi.file.LocalTarget(path='/tmp/group_stats/impute_single.pkl')
        else:
            grp_name = '-'.join(sorted(self.group_names.split(',')))
            return luigi.file.LocalTarget(path='/tmp/group_stats/impute_{}.pkl'.format(grp_name))

    def requires(self):
        groups = self.group_names.split(',')
        if len(groups) == 1:
            # This means luigi will understand that these
            # are all the same. Bascially, any impute with a
            # group name that _isnt_ just '' will require the
            # impute with group name ''. This will be the
            # only one actually calculated
            if self.group_names != '':
                yield MissingValueImputation(group_names='')
        else:
            yield base_data.SplitData(offset=0, rand_round=0)
            for combo in self.sub_tasks(groups):
                yield GroupFnQuery(','.join(combo))
            yield SqlGroupFunction(self.group_names)

    @staticmethod
    def sub_tasks(groups):
        return {tuple(sorted(s)) for s in itertools.combinations(groups, len(groups) - 1)}

    def run(self):
        groups = self.group_names.split(',')
        if len(groups) == 1:
            # See requires for an explanation of
            # why this chekc is here.
            if self.group_names == '':
                self.run_single()
        else:
            self.run_multi()

    def run_single(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        con.create_aggregate("log1pMean", 1, Log1PMean)
        con.create_function("expm1", 1, np.expm1)
        query = """
            SELECT expm1(log1pMean(adjusted_demand)) as adjusted_demand
              FROM data
             WHERE adjusted_demand is not NULL
                   and week_num < 8
        """
        mean_adjusted_demand = con.execute(query).fetchone()[0]
        con.close()
        cls = sklearn.dummy.DummyRegressor('constant', constant=mean_adjusted_demand)
        # It's naughty, but this lets us get around having to actually
        # fit anything.
        cls.fit(np.random.uniform(size=(10, 5)), np.random.uniform(size=10))
        cls.predict(np.random.uniform(size=(10, 1)))
        with open(self.output().path, 'wb') as f:
            pickle.dump(cls, f)

    def run_multi(self):
        groups = self.group_names.split(',')
        data = pandas.read_csv('/tmp/split_data/0/train/0.csv')
        target_groupmean = GroupFnQuery(self.group_names)
        target = target_groupmean.compute(data, inplace=False)[target_groupmean.col_name]
        features = []
        for combo in self.sub_tasks(groups):
            feature_groupmean = GroupFnQuery(','.join(combo))
            feature = feature_groupmean.compute(data, inplace=False)[feature_groupmean.col_name]
            features.append(feature)
        features = pandas.DataFrame(features).T
        cls = sklearn.linear_model.Ridge()
        cls.fit(features, target)
        with open(self.output().path, 'wb') as f:
            pickle.dump(cls, f)

    def impute(self, data):
        groups = self.group_names.split(',')
        if len(groups) == 1:
            return self.impute_single(data)
        else:
            return self.impute_multi(data)

    def impute_single(self, data):
        with open(self.output().path, 'rb') as f:
            cls = pickle.load(f)
        return cls.predict(data)

    def impute_multi(self, data):
        groups = self.group_names.split(',')
        with open(self.output().path, 'rb') as f:
            cls = pickle.load(f)
        assert isinstance(cls, sklearn.linear_model.Ridge)
        X = []
        for combo, coef in zip(self.sub_tasks(groups), cls.coef_):
            if coef == 0:
                feature = np.zeros(data.index.shape)
            else:
                feature_groupmean = GroupFnQuery(','.join(combo))
                feature = feature_groupmean.compute(data, inplace=False)[feature_groupmean.col_name]
            X.append(feature)
        X = pandas.DataFrame(X).T
        # print(pandas.Series(cls.coef_, index=X.columns).sort_values())

        with open(self.output().path, 'rb') as f:
            cls = pickle.load(f)
        return cls.predict(X)


class Log1PMean:
    def __init__(self):
        self.vec = []

    def step(self, value):
        self.vec.append(value)

    def finalize(self):
        return np.mean(np.log1p(self.vec))


class GeneralMeans(luigi.Task):

    @staticmethod
    def all_tasks():
        names = [
            'sales_depo',
            'sales_channel',
            'route_id',
            'product_id',
            'client_id'
        ]
        tasks = {frozenset(s) for s in GeneralMeans.names_maker(names) if len(s) <= 3}
        return tasks

    def requires(self):
        for t in GeneralMeans.all_tasks():
            tlist = sorted(list(t))
            yield GroupFnQuery(group_names=','.join(tlist))

    @staticmethod
    def names_maker(names):
        if len(names) == 1:
            yield names
        for name in names:
            subnames = [n for n in names if n != name]
            for subname in GeneralMeans.names_maker(subnames):
                yield subname
                yield [name] + subname

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True


class GroupFnQuery(luigi.Task):
    group_names = luigi.Parameter()

    def requires(self):
        return [SqlGroupFunction(group_names=self.group_names),
                MissingValueImputation(group_names=self.group_names)]

    def run(self):
        pass

    def complete(self):
        for task in self.requires():
            if not task.complete():
                return False
        return True

    def compute(self, data, prefix=None, inplace=False):
        group_names = self.group_names.split(',')
        self.col_name = 'mean_' + '-'.join(group_names)
        group_task, impute_task = self.requires()

        self.data_map = pandas.read_msgpack(group_task.output().path).rename({}, {'adjusted_demand': 'target'})
        self.data_map.name = self.col_name
        res = pandas.merge(data, self.data_map,
                           left_on=group_names,
                           right_on=group_names, how='left')
        if res.target.isnull().any():
            res.target = res.target.fillna(-1)
            imputed = impute_task.impute(data)
            res.ix[res.target.isnull(), 'target'] = imputed[res.target.isnull()]

        if prefix is None:
            prefix = ''
        col_name = prefix + self.col_name
        if not inplace:
            return data.assign(**{col_name: res['target']})
        else:
            data[col_name] = res['target']
            return data
