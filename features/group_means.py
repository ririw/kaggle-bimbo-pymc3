import logging

import dask
import dask.async
import dask.dataframe
import dask.multiprocessing
import nose.tools
import luigi
import pandas

from features.base_data import BuildData


class GroupFunction(luigi.Task):
    rand_round = luigi.IntParameter()
    group_names = luigi.Parameter()
    function_name = luigi.Parameter()

    def requires(self):
        nose.tools.assert_in(self.function_name, {'mean', 'std'})
        return BuildData(rand_round=self.rand_round)

    def output(self):
        nose.tools.assert_in(self.function_name, {'mean', 'std'})
        grp_name = '-'.join(self.group_names.split(','))
        return luigi.file.LocalTarget(
            path='/tmp/group_stats/{}/group_{}_{}.msgpack'.format(self.rand_round,
                                                                  self.function_name,
                                                                  grp_name))

    def run(self):
        with dask.set_options(get=dask.multiprocessing.get):
            data = dask.dataframe.read_csv('/tmp/split_data/{}/train/*.csv'.format(self.rand_round))
            if self.function_name == 'mean':
                res = data.groupby(self.group_names.split(',')).adjusted_demand.mean().compute()
            elif self.function_name == 'std':
                res = data.groupby(self.group_names.split(',')).adjusted_demand.std().compute()
            else:
                nose.tools.assert_in(self.function_name, {'mean', 'std'})
        if ',' in self.group_names:
            import ipdb; ipdb.set_trace()
        self.output().makedirs()
        logging.critical('Writing file: {}, do not interrupt'.format(self.output().path))
        with open(self.output().path, 'wb') as f:
            res.reset_index().to_msgpack(f)
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
        yield GroupFunction(rand_round=self.rand_round, group_names='client_id', function_name='mean')
        for fn in fns:
            for name in names:
                yield GroupFunction(rand_round=self.rand_round, group_names=name, function_name=fn)
                for n2 in names:
                    if n2 != name:
                        yield GroupFunction(rand_round=self.rand_round, group_names=','.join([name, n2]), function_name=fn)

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
        assert task.complete(), 'Must run the task first'
        self.col_name = self.function_name + '_' + '-'.join(self.group_names)

        self.data_map = pandas.read_msgpack(task.output().path).rename({}, {'adjusted_demand': 'target'})
        self.data_map.name = self.col_name

    def compute(self, data, prefix=None, inplace=False):
        if False and len(self.group_names) == 1:
            col = data[self.group_names[0]].astype(int)
            res = self.data_map[col].reset_index(drop=True)
        else:
            try:
                res = pandas.merge(data, self.data_map, left_on=self.group_names, right_on=self.group_names, how='left')['target']
            except KeyError:
                import ipdb; ipdb.set_trace()

        if prefix is None:
            prefix = ''
        col_name = prefix + self.col_name
        if not inplace:
            return data.assign(**{col_name: res})
        else:
            data[col_name] = res
            return data

    def __repr__(self):
        return 'GroupFnQuery({}, ???, {})'.format(repr(','.join(self.group_names)), self.function_name)