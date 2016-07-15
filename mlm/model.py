import numpy as np
import pymc3
import theano
import theano.tensor as T


def simple_halfnorm_priors(name_prefix, shape):
    prior_mean = pymc3.HalfNormal(name_prefix + '_pr_mu', 10)
    prior_var = pymc3.HalfNormal(name_prefix + '_pr_sd', 10)
    return pymc3.Normal(name_prefix, prior_mean, prior_var, shape=shape)


def build_model(data, unique_vals):
    simple_model = pymc3.Model()

    sales_chans_var = theano.shared(value=data.sales_channel.cat.codes.values, name='sales_chans_var')
    sales_depo_var = theano.shared(value=data.sales_depo.cat.codes.values.astype(np.int16), name='sales_depo_var')
    product_id_var = theano.shared(value=data.product_id.cat.codes.values.astype(np.int16), name='product_id_var')
    route_id_var = theano.shared(value=data.route_id.cat.codes.values.astype(np.int16), name='route_id_var')
    adj_demand_var = theano.shared(value=data.adjusted_demand.values.astype(np.float32), name='adj_demand_var')

    with simple_model:
        sales_channel = simple_halfnorm_priors('sales_channel', shape=unique_vals['sales_channel'].shape[0])
        sales_depo = simple_halfnorm_priors('sales_depo', shape=unique_vals['sales_depo'].shape[0])
        product_id = simple_halfnorm_priors('product_id', shape=unique_vals['product_id'].shape[0])
        route_id = simple_halfnorm_priors('route_id', shape=unique_vals['route_id'].shape[0])

        sales_channel_vs = sales_channel[sales_chans_var]
        sales_depo_vs = sales_depo[sales_depo_var]
        product_id_vs = product_id[product_id_var]
        route_id_vs = route_id[route_id_var]

        demand_mu = T.exp(sales_channel_vs * sales_depo_vs * product_id_vs * route_id_vs)

        adjusted_demand = pymc3.Poisson(name='adjusted_demand', mu=demand_mu, observed=adj_demand_var)

    tensor_vars = [sales_chans_var, sales_depo_var, product_id_var, route_id_var, adj_demand_var]
    return simple_model, tensor_vars, [adjusted_demand]
