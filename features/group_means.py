import logging

import dask
import dask.async
import dask.dataframe
import dask.multiprocessing
import sqlite3
import nose.tools
import numpy as np
import luigi
import pandas
from dask.dataframe.groupby import DataFrameGroupBy

from features.base_data import BuildData


class SqlGroupFunction(luigi.Task):
    group_names = luigi.Parameter()

    def output(self):
        grp_name = '-'.join(sorted(self.group_names.split(',')))
        return luigi.file.LocalTarget(
            path='/tmp/group_stats/group_{}.msgpack'.format(grp_name))

    def run(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        con.create_aggregate("log1pMean", 1, Log1PMean)
        con.create_function("expm1", 1, np.expm1)
        query = """
            SELECT {},
                   log1pMean(adjusted_demand) as adjusted_demand
              FROM data
             WHERE adjusted_demand is not NULL
                   and week_num < 8
          GROUP BY {}
        """.format(self.group_names, self.group_names)
        res = pandas.read_sql(query, con=con)
        self.output().makedirs()
        with open(self.output().path, 'wb') as f:
            res.to_msgpack(f, compress='zlib')


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
        tasks = {frozenset(s) for s in GeneralMeans.names_maker(names) if len(s) <= 2}
        return tasks

    def requires(self):
        for t in GeneralMeans.all_tasks():
            tlist = sorted(list(t))
            yield SqlGroupFunction(group_names=','.join(tlist))

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


class GroupFnQuery:
    def __init__(self, group_names, rand_round, function_name):
        self.group_names = group_names.split(',')
        self.function_name = function_name
        task = SqlGroupFunction(group_names=group_names,
                                rand_round=rand_round,
                                function_name=function_name)
        assert task.complete(), 'Must run the task first: {} -- {}'.format(group_names, function_name)
        self.col_name = self.function_name + '_' + '-'.join(self.group_names)

        self.data_map = pandas.read_msgpack(task.output().path).rename({}, {'adjusted_demand': 'target'})
        self.data_map.name = self.col_name

    def compute(self, data, prefix=None, inplace=False):
        print(self.data_map.dtypes)
        print(self.group_names)
        res = pandas.merge(data, self.data_map,
                           left_on=self.group_names,
                           right_on=self.group_names, how='left')
        if res.target.isnull().any():
            print("Found nulls in res for groups {}, null prop was {}".format(
                self.group_names, str(res.target.isnull().mean())))
            res.target = res.target.fillna(-1)

        if prefix is None:
            prefix = ''
        col_name = prefix + self.col_name
        if not inplace:
            return data.assign({col_name: res['target']})
        else:
            data[col_name] = res['target']
            return data

    def __repr__(self):
        return 'GroupFnQuery({}, ???, {})'.format(repr(','.join(self.group_names)), self.function_name)
