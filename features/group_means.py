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


def count_contents(key_cols):
    def count(frame: pandas.DataFrame):
        res = frame.groupby(key_cols).adjusted_demand.agg({
        'n': lambda f: f.shape[0],
        'total': lambda v: np.log1p(v).sum()})
        return res.reset_index()

    return count


class GroupFunction(luigi.Task):
    rand_round = luigi.IntParameter()
    group_names = luigi.Parameter()
    function_name = luigi.Parameter()

    def requires(self):
        nose.tools.assert_in(self.function_name, {'mean'})
        return BuildData(rand_round=self.rand_round)

    def output(self):
        nose.tools.assert_in(self.function_name, {'mean'})
        grp_name = '-'.join(sorted(self.group_names.split(',')))
        return luigi.file.LocalTarget(
            path='/tmp/group_stats/{}/group_{}_{}.msgpack'.format(self.rand_round,
                                                                  self.function_name,
                                                                  grp_name))

    def run(self):
        with dask.set_options(get=dask.multiprocessing.get):
            names = self.group_names.split(',')
            data = dask.dataframe.read_csv('/tmp/split_data/{}/train/*.csv'.format(self.rand_round))
            if len(names) > 1:
                partition_groups = data.map_partitions(count_contents(names)).compute()
                partition_totals = partition_groups.groupby(names).sum()
                partition_totals['target'] = partition_totals.total / partition_totals.n
                res = partition_totals['target']
            else:
                res = data.groupby(names[0]).adjusted_demand.mean().compute()
                res.name = 'target'
        self.output().makedirs()
        logging.critical('Writing file: {}, do not interrupt'.format(self.output().path))
        with open(self.output().path, 'wb') as f:
            res.reset_index().to_msgpack(f)
        logging.critical('Finished writing file: {}, you can now interrupt'.format(self.output().path))

class SqlGroupFunction(luigi.Task):
    rand_round = luigi.IntParameter()
    group_names = luigi.Parameter()
    function_name = luigi.Parameter()

    def output(self):
        grp_name = '-'.join(sorted(self.group_names.split(',')))
        return luigi.file.LocalTarget(
            path='/tmp/group_stats/{}/group_{}_{}.msgpack'.format(self.rand_round, self.function_name, grp_name))

    def run(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        query = """
            SELECT {},
                   avg(adjusted_demand) as adjusted_demand
              FROM data
             WHERE adjusted_demand is not NULL
                   and week_num < 8
          GROUP BY {}
        """.format(self.group_names, self.group_names)
        res = pandas.read_sql(query, con=con)
        res.adjusted_demand = np.log1p(res.adjusted_demand)
        with open(self.output().path, 'wb') as f:
            res.to_msgpack(f, compress='zlib')




class GeneralMeans(luigi.Task):
    rand_round = luigi.IntParameter()

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
            yield SqlGroupFunction(rand_round=self.rand_round, group_names=','.join(tlist), function_name='mean')


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
        task = GroupFunction(group_names=group_names,
                             rand_round=rand_round,
                             function_name=function_name)
        assert task.complete(), 'Must run the task first: {} -- {}'.format(group_names, function_name)
        self.col_name = self.function_name + '_' + '-'.join(self.group_names)

        self.data_map = pandas.read_msgpack(task.output().path).rename({}, {'adjusted_demand': 'target'})
        self.data_map.name = self.col_name

    def compute(self, data, prefix=None, inplace=False):
        res = pandas.merge(data, self.data_map,
                           left_on=self.group_names,
                           right_on=self.group_names, how='left')
        assert res.target.notnull().all(), res.target.notnull().all()

        if prefix is None:
            prefix = ''
        col_name = prefix + self.col_name
        if not inplace:
            return data.assign(**{#col_name + '_log': np.log(res['target'] + 1),
                                  col_name: np.expm1(res['target'])})
        else:
            #data[col_name + '_log'] = np.log(res['target'] + 1)
            data[col_name] = np.expm1(res['target'])
            return data

    def __repr__(self):
        return 'GroupFnQuery({}, ???, {})'.format(repr(','.join(self.group_names)), self.function_name)
