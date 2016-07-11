import logging

import dask
import dask.async
import dask.dataframe
import dask.multiprocessing
import nose.tools
import luigi
import pandas
import numpy as np

from features.base_data import BuildData


class GroupFunction(luigi.Task):
    rand_round = luigi.IntParameter()
    group_name = luigi.Parameter()
    function_name = luigi.Parameter()

    def requires(self):
        nose.tools.assert_in(self.function_name, {'mean', 'median', 'std'})
        return BuildData(rand_round=self.rand_round)

    def output(self):
        nose.tools.assert_in(self.function_name, {'mean', 'median', 'std'})
        return luigi.file.LocalTarget(
            path='/tmp/group_stats/{}/group_{}_{}.msgpack'.format(self.rand_round,
                                                                  self.function_name,
                                                                  self.group_name))

    def run(self):
        if self.function_name == 'median':
            with dask.set_options(get=dask.async.get_sync):
                data = dask.dataframe.read_csv('/tmp/split_data/{}/train/*.csv'.format(self.rand_round))
                res = data.groupby(self.group_name).adjusted_demand.apply(np.median, columns='median').compute()

        with dask.set_options(get=dask.multiprocessing.get):
            data = dask.dataframe.read_csv('/tmp/split_data/{}/train/*.csv'.format(self.rand_round))
            if self.function_name == 'mean':
                res = data.groupby(self.group_name).adjusted_demand.mean().compute()
            elif self.function_name == 'std':
                res = data.groupby(self.group_name).adjusted_demand.std().compute()
            else:
                nose.tools.assert_in(self.function_name, {'mean', 'median', 'std'})

        self.output().makedirs()
        logging.critical('Writing file: {}, do not interrupt'.format(self.output().path))
        with open(self.output().path, 'wb') as f:
            res.to_msgpack(f)
        logging.critical('Finished writing file: {}, you can now interrupt'.format(self.output().path))


class GeneralMeans(luigi.Task):
    rand_round = luigi.IntParameter()

    def requires(self):
        names = [
            'sales_depo',
            'sales_channel',
            'route_id',
            'product_id',
        ]
        fns = ['mean', 'std']
        for fn in fns:
            for name in names:
                yield GroupFunction(rand_round=self.rand_round, group_name=name, function_name=fn)

    def complete(self):
        for r in self.requires():
            if not r.complete():
                return False
        return True


class GroupFnQuery:
    def __init__(self, group_name, rand_round, function_name):
        self.group_name = group_name
        self.function_name = function_name
        task = GroupFunction(group_name=group_name,
                             rand_round=rand_round,
                             function_name=function_name)
        assert task.complete(), 'Must run the task first'
        self.data_map = pandas.read_msgpack(task.output().path)

    def compute(self, data, prefix=None, inplace=False):
        col = data[self.group_name].astype(int)
        res = self.data_map[col].reset_index(drop=True)
        if prefix is None:
            prefix = ''
        col_name = prefix + self.function_name + '_' + self.group_name
        if not inplace:
            return data.assign(**{col_name: res})
        else:
            data[col_name] = res
            return data
