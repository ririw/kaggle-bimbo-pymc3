import sqlite3
import logging
import coloredlogs
import pandas
import numpy as np
import tqdm
from sklearn import *
import ml_metrics
import joblib
from functools import reduce

coloredlogs.install(level=logging.INFO)

def fit_mini_batch(test=False, train_batch=None):
    # Fit three little estimators in each
    # round of boosting, all from the same
    # sample.
    reg = ensemble.ExtraTreesRegressor(
        warm_start=True,
        verbose=0,
        n_jobs=1,
        n_estimators=4)
    if train_batch is None:
        train_batch = np.random.randint(100)
    con = sqlite3.connect('/tmp/data.sqlite3')
    try:
        logging.info('Fetching data: %d' % train_batch)
        data = pandas.read_sql('''
            SELECT week_num,
                   sales_channel,
                   sales_depo,
                   adjusted_demand,
                   rand
              FROM data
             WHERE adjusted_demand is not null 
                   AND rand = ? AND week_num < 8''', con=con, params=[train_batch])
    finally:
        con.close()
    data = data.drop(['week_num', 'rand'], axis=1)
    X = data.drop('adjusted_demand', axis=1).as_matrix().copy(order='C')
    # See https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
    y = np.log(1+data.adjusted_demand.as_matrix().copy(order='C'))
    reg.fit(X, y)

    if test:
        test_batch = np.random.randint(100)
        con = sqlite3.connect('/tmp/data.sqlite3')
        try:
            data = pandas.read_sql('''
                SELECT week_num,
                       sales_channel,
                       sales_depo,
                       adjusted_demand,
                       rand
                  FROM test_data 
                 WHERE adjusted_demand is not null 
                       AND rand = ? AND week_num >= 8''', con=con, params=[test_batch])
        finally:
            con.close()

        data = data.drop(['week_num', 'rand'], axis=1)
        X = data.drop('adjusted_demand', axis=1).as_matrix().copy(order='C')
        pred = np.exp(reg.predict(X)) - 1
        logging.info(ml_metrics.rmsle(data.adjusted_demand, pred))
    return reg

def merge_ests(e1, e2):
    e1.estimators_ += e2.estimators_
    e1.n_estimators += e2.n_estimators
    return e1

trees = []


for i in range(0, 100, 10):
    trees.extend(
        joblib.Parallel(n_jobs=1)(
            joblib.delayed(fit_mini_batch)(test=False, train_batch=j) for j in range(i, i+10))
    )

reg = reduce(merge_ests, trees)
test_batch = np.random.randint(100)
con = sqlite3.connect('/tmp/data.sqlite3')
try:
    data = pandas.read_sql('''
        SELECT week_num,
               sales_channel,
               sales_depo,
               adjusted_demand,
               rand
          FROM data
         WHERE adjusted_demand is not null
               AND rand = ? AND week_num >= 8''', con=con, params=[test_batch])
finally:
    con.close()

data = data.drop(['week_num', 'rand'], axis=1)
X = data.drop('adjusted_demand', axis=1).as_matrix().copy(order='C')
pred = np.exp(reg.predict(X)) - 1
logging.info(ml_metrics.rmsle(data.adjusted_demand, pred))
