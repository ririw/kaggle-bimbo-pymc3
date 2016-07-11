import logging

import coloredlogs
import dask
import dask.dataframe
import ml_metrics
import pandas
from sklearn import *
import numpy as np
from tqdm import tqdm

import features

coloredlogs.install(level=logging.INFO)

dropped_cols = ['adjusted_demand', 'client_id',
                'product_id', 'route_id', 'sales_depo',
                'sales_channel', 'rand', 'week_num']

cls = linear_model.SGDRegressor('huber')
#X = dask.dataframe.from_delayed([dask.delayed(features.make_train_batch)(ix) for ix in range(10)]).drop(dropped_cols, axis=1).compute()
#y = dask.dataframe.from_delayed([dask.delayed(features.make_train_batch)(ix) for ix in range(10)])['adjusted_demand'].compute()
#cls.fit(X, y)
for i in tqdm(range(100)):
    data = features.make_train_batch(i % 100)
    X = data.drop(dropped_cols, 1)
    cls.fit(X, data.adjusted_demand)

ys = []
y_preds = []
for i in tqdm(range(100)):
    data = features.make_test_batch(i % 100)
    X = data.drop(dropped_cols, 1)
    ys.append(data.adjusted_demand)
    y_preds.append(cls.predict(X))

y = np.concatenate(ys)
y_pred = np.concatenate(y_preds)
del ys, y_preds

print(y_pred.shape)
print(y.shape)
print(y_pred[:10])
print(y[:10])
print(ml_metrics.rmse(y, y_pred))
print(pandas.Series(cls.coef_, index=X.columns).sort_values())