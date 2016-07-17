import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3.variational as variational
import theano
import theano.tensor as T
from numpy import log1p, expm1
from sklearn import base


class PM_NN_Model(base.RegressorMixin):
    def __init__(self, verbose=0):
        # We do a lazy init here, because it takes
        # so long to build pymc models
        self.verbose = verbose
        self.model = None

    def make_model(self, x, y):
        self.model = pm.Model()
        product_ids = x.product_id.astype('category')
        self.product_categories = product_ids.cat.categories.values
        self.product_id_var = theano.shared(value=product_ids.cat.codes.values.astype(np.uint32), name='product_id_var')

        route_ids = x.route_id.astype('category')
        self.route_categories = route_ids.cat.categories.values
        self.route_id_var = theano.shared(value=route_ids.cat.codes.values.astype(np.uint32), name='route_id_var')

        client_ids = x.client_id.astype('category')
        self.client_categories = client_ids.cat.categories.values
        self.client_id_var = theano.shared(value=client_ids.cat.codes.values.astype(np.uint32), name='client_id_var')

        self.adjusted_demand_var = theano.shared(value=y.values.astype(np.float32), name='adjusted_demand_var')

        with self.model:
            product_mean = pm.Normal('product_mean', 0, 10, shape=self.product_categories.shape[0])
            route_mean = pm.Normal('route_mean', 0, 10)
            route_demand = pm.Normal('route_demand', route_mean, 10, shape=self.route_categories.shape[0])
            client_mean = pm.Normal('client_mean', 0, 10)
            client_demand = pm.Normal('client_demand', client_mean, 10, shape=self.client_categories.shape[0])

            pooled_variance = pm.HalfNormal('pooled_variance', 10)

            input_vec = T.concatenate([
                product_mean[self.product_id_var][:, None],
                route_demand[self.route_id_var][:, None],
                client_demand[self.client_id_var][:, None]], axis=1)

            w1 = pm.Normal('w1', 0, 5, shape=[3, 32], testval=np.random.randn(3, 32))
            b1 = pm.Normal('b1', 0, 5, shape=[32], testval=np.random.randn(32))

            l1 = T.nnet.relu(T.dot(input_vec, w1) + b1)

            w2 = pm.Normal('w2', 0, 5, shape=[32, 32], testval=np.random.randn(32, 32))
            b2 = pm.Normal('b2', 0, 5, shape=[32], testval=np.random.randn(32))

            l2 = T.nnet.relu(T.dot(l1, w2) + b2)

            w3 = pm.Normal('w3', 0, 5, shape=[32], testval=np.random.randn(32))
            b3 = pm.Normal('b3', 0, 5, testval=np.random.randn())
            obs = T.dot(l2, w3) + b3

            self.adjusted_demand = pm.Normal(
                'adjusted_demand', obs, pooled_variance, observed=T.log1p(self.adjusted_demand_var))

    def minibatch_tensors(self, x_all, y_all):
        while True:
            sample = np.random.choice(y_all.shape, size=1000, replace=True)
            x = x_all.ix[sample]
            y = y_all.ix[sample]
            product_cats = x.product_id.astype(
                'category', categories=self.product_categories).cat.codes.values.astype(np.uint32)
            route_cats = x.route_id.astype(
                'category', categories=self.route_categories).cat.codes.values.astype(np.uint32)
            client_cats = x.client_id.astype(
                'category', categories=self.client_categories).cat.codes.values.astype(np.uint32)
            yield [product_cats, route_cats, client_cats, y.astype(np.float32)]

    def fit(self, x, y):
        if self.verbose:
            logging.info('Building pymc3 model')
        self.make_model(x, y)
        if self.verbose:
            logging.info('Sampling...')
        with self.model:
            minibatch_tensors = [
                self.product_id_var,
                self.route_id_var,
                self.client_id_var,
                self.adjusted_demand_var
            ]
            output_rvs = [self.adjusted_demand]
            total_size = x.shape[0]

            self.v_params = variational.advi_minibatch(
                n=int(1e6),
                minibatch_tensors=minibatch_tensors,
                minibatch_RVs=output_rvs,
                total_size=total_size,
                minibatches=self.minibatch_tensors(x, y)
            )
            plt.plot(self.v_params.elbo_vals[-int(1e5):])
            plt.savefig('./elbo.png')
            self.trace = variational.sample_vp(self.v_params)
            if self.verbose:
                print(pm.summary(self.trace[100:], varnames=['route_demand', 'client_demand']))
        return self

    def predict(self, x):
        with self.model:
            product_cats = x.product_id.astype('category', categories=self.product_categories)
            self.product_id_var.set_value(product_cats.cat.codes.values.astype(np.uint8))

            route_cats = x.route_id.astype('category', categories=self.route_categories)
            self.route_id_var.set_value(route_cats.cat.codes.values.astype(np.uint8))

            client_cats = x.client_id.astype('category', categories=self.client_categories)
            self.client_id_var.set_value(client_cats.cat.codes.values.astype(np.uint8))

            y_pred = pm.sample_ppc(self.trace[-500:])['adjusted_demand']
            res = expm1(y_pred).mean(axis=0)
            print(res)
            return res
            # return y_pred.mean(axis=0)


model = PM_NN_Model
