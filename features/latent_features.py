"""
This dataset has a lot of client_id and product_id values. Can we perhaps simplify it,
by finding latent variabes that better explain the dataset? For this, we'll represent
each product by a vector of a particular length, and see whether this improves our
later predictions.
"""
import logging

import coloredlogs
import luigi
import nose.tools
import pandas
import pymc3 as pm
import theano.tensor as T
import numpy as np
import theano
import dask
import dask.dataframe

from features.base_data import BuildData


class LatentProductType(luigi.Task):
    rand_round = luigi.IntParameter()

    def requires(self):
        return BuildData(rand_round=self.rand_round)

    def output(self):
        return luigi.file.LocalTarget('/tmp/latent/{}/product.msgpack'.format(self.rand_round))

    def minibatches(self, unique_products):
        while True:
            batch = np.random.randint(100)
            input_data = pandas.read_csv('/tmp/split_data/{}/train/{}.csv'.format(self.rand_round, batch))
            input_data.product_id = input_data.product_id.astype('category', categories=unique_products)
            yield input_data[['product_id', 'adjusted_demand']]

    @staticmethod
    def expand_batch(batch):
        return [
            batch.product_id.cat.codes.values,
            batch.adjusted_demand.values
        ]

    def run(self):
        coloredlogs.install()
        logging.info('Fetching some data')
        with dask.set_options(get=dask.multiprocessing.get):
            data = dask.dataframe.read_csv('/tmp/split_data/{}/train/*.csv'.format(self.rand_round))
            total_size = data.week_num.count().compute()
            nose.tools.assert_greater(total_size, 100, 'Not enought data!')

            unique_products = data['product_id'].unique().compute().astype(np.uint16)
            sample = data.head()
        logging.info('Got it!')

        product_id_var = theano.shared(
            value=sample.product_id.astype('category', categories=unique_products).cat.codes.values,
            name='product_id_var')
        adjusted_demand_var = theano.shared(
            value=sample.adjusted_demand.values,
            name='adjusted_demand_var')

        model = pm.Model()
        with model:
            product_category = pm.Uniform('cat', 0, 1, shape=(unique_products.shape[0], 5))
            product_vecs = pm.Normal('vecs', 0, 100, shape=5)
            adjusted_demand_variance = pm.HalfNormal('demand_variance', 10)
            product_pred = T.dot(product_category[product_id_var], product_vecs)

            adjusted_demand = pm.Normal('adjusted_demand',
                                        product_pred, adjusted_demand_variance,
                                        observed=adjusted_demand_var)

            minibatches = map(self.expand_batch, self.minibatches(unique_products))

            v_params = pm.variational.advi_minibatch(
                n=100,
                minibatch_tensors=[product_id_var, adjusted_demand_var],
                minibatch_RVs=[adjusted_demand],
                minibatches=minibatches,
                total_size=total_size,
                n_mcsamples=5,
                verbose=True
            )
            trace = pm.variational.sample_vp(v_params, draws=500)
            print(pm.summary(trace))

        res = trace[-100:]['cat'].mean(0)
        self.output().makedirs()
        pandas.DataFrame(res, index=unique_products.values).to_msgpack(self.output().path)


class LatentProductTypeQuery:
    def __init__(self, rand_round):
        task = LatentProductType(rand_round=rand_round)
        assert task.complete()
        self.product_types = pandas.read_msgpack(task.output().path)

    def compute(self, data, inplace=False):
        col = data['product_id'].astype(int)
        res = np.vstack(self.product_types.ix[col].reset_index(drop=True).values)
        if not inplace:
            return data.assign(**{'latent_%d' % i: res[:, i] for i in range(5)})
        else:
            for i in range(5):
                data['latent_%d' % i] = res[:, i]
            return data
