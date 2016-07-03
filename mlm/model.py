import numpy as np
import pymc3
import theano


def simple_halfnorm_priors(name_prefix, shape):
    prior_mean = pymc3.HalfNormal(name_prefix + '_pr_mu', 10)
    prior_var = pymc3.HalfNormal(name_prefix + '_pr_sd', 10)
    return pymc3.Normal(name_prefix, prior_mean, prior_var, shape=shape)


def build_model(data):
    simple_model = pymc3.Model()

    week_var = theano.shared(
        value=data.week_num.cat.codes.values,
        name='week_var')
    sales_chans_var = theano.shared(
        value=data.sales_channel.cat.codes.values,
        name='sales_chans_var')
    sales_depo_var = theano.shared(
        value=data.sales_depo.cat.codes.values,
        name='sales_depo_var')
    adj_demand_var = theano.shared(
        value=data.adjusted_demand.values.astype(np.float16),
        name='adj_demand_var')

    # print(week_var.dtype)
    # print(sales_chans_var.dtype)
    # print(sales_depo_var.dtype)
    # print(adj_demand_var.dtype)

    with simple_model:
        week_rate = simple_halfnorm_priors('week', data.week_num.nunique())
        sales_channel = simple_halfnorm_priors('sales_channel', shape=data.sales_channel.nunique())
        sales_depo = simple_halfnorm_priors('sales_depo', shape=data.sales_depo.nunique())

        week_vs = week_rate[week_var]
        sales_channel_vs = sales_channel[sales_chans_var]
        sales_depo_vs = sales_depo[sales_depo_var]

        demand_mu = week_vs + sales_channel_vs + sales_depo_vs

        adjusted_demand = pymc3.Poisson(
            name='adjusted_demand',
            mu=demand_mu,
            observed=adj_demand_var
        )

    return simple_model, [week_var, sales_chans_var, sales_depo_var, adj_demand_var], [adjusted_demand]
