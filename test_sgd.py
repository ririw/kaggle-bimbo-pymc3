import logging
import warnings

import coloredlogs
import ml_metrics
import numpy as np
import pandas

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn import *
from tqdm import tqdm

import features

coloredlogs.install(level=logging.INFO)

dropped_cols = ['adjusted_demand', 'client_id',
                'product_id', 'route_id', 'sales_depo',
                'sales_channel', 'rand', 'week_num']

cls = linear_model.SGDRegressor('huber', 'elasticnet')
for i in tqdm(range(100)):
    logging.info('Starting batch {}'.format(i))
    data = features.make_train_batch(i % 100)
    logging.info('Got data')
    X = data.drop(dropped_cols, 1)
    y = data.adjusted_demand
    logging.info('Training...')
    cls.fit(X, y)
    logging.info('Trained!')

ys = []
y_preds = []
for i in tqdm(range(100)):
    data = features.make_test_batch(i % 100)
    X = data.drop(dropped_cols, 1)
    ys.append(data.adjusted_demand)
    y_pred = np.maximum(cls.predict(X), 1)
    y_preds.append(y_pred)

y = np.concatenate(ys)
y_pred = np.concatenate(y_preds)
del ys, y_preds

print(y_pred.shape)
print(y.shape)
print(y_pred[:10])
print(y[:10])
print(ml_metrics.rmse(y, y_pred))
print(ml_metrics.rmsle(y, y_pred))
print(pandas.Series(cls.coef_, index=X.columns).sort_values())
