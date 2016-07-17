"""
This model works on a subset of the data. We slice out a few different
product's worth of data and use them, because that'll hopefully mean we
can use
a) More sophisticated models
b) Nuts or metripolis sampling.
"""

import importlib
import logging
import sqlite3

import coloredlogs
import numpy as np
import pandas
from ml_metrics import rmsle
from plumbum import cli
from tqdm import tqdm, tqdm_gui


def make_xy(data):
    x = data.drop(['adjusted_demand', 'week_num', 'id'], 1)
    y = data.adjusted_demand
    id = data.id
    return x, y, id


class SubsetModelCLI(cli.Application):
    split_field = cli.SwitchAttr(['-s', '--split-field'],
                                 argtype=str,
                                 default='product_id',
                                 help='The field to split upon, defaults to product_id')
    num_vals = cli.SwitchAttr('-n',
                              argtype=int,
                              default=7,
                              help='The number of values to do in a batch. Defaults to 10')
    score_run = cli.Flag('--score-run', help='Do a scoring run.')
    full_test = cli.Flag('--full-test', help='Do not sample during the testing run, go through everything')

    def get_random_field_vals(self):
        con = sqlite3.connect('/tmp/data.sqlite3')
        rng = np.random.RandomState(None)
        try:
            res = con.execute('''SELECT DISTINCT {} FROM data'''.format(self.split_field))
            a = np.array(res.fetchall()).flatten()
            return a[rng.choice(a.shape[0], size=self.num_vals)]
        finally:
            con.close()

    def get_field_vals(self):
        con = sqlite3.connect('/tmp/data.sqlite3')
        res = con.execute('''SELECT DISTINCT {} FROM data'''.format(self.split_field))
        a = np.array(res.fetchall()).flatten()
        con.close()
        return a

    def get_sklearn_like_model(self, model_class):
        model = importlib.import_module(model_class).model(verbose=1)
        return model

    def read_dataset(self, items):
        logging.info('Reading dataset')
        con = sqlite3.connect('/tmp/data.sqlite3')
        products = ','.join(str(v) for v in items)
        data = pandas.read_sql('''
             SELECT id,
                    week_num,
                    sales_depo,
                    sales_channel,
                    route_id,
                    client_id,
                    product_id,
                    adjusted_demand,
                    case when adjusted_demand is null then 'score'
                         when week_num >= 9 then 'test'
                         else 'train' end as ds
               FROM data
              WHERE {} IN ({})
        '''.format(self.split_field, products), con=con)
        con.close()
        logging.info('Building subsets')
        train_data = data[data.ds == 'train'].drop('ds', 1).copy()
        test_data = data[data.ds == 'test'].drop('ds', 1).copy()
        score_data = data[data.ds == 'score'].drop('ds', 1).copy()
        non_score_data = data[data.ds != 'score'].drop('ds', 1).copy()
        logging.info('Data prepared')
        return train_data, test_data, score_data, non_score_data

    def training_run(self, field_vals, model_class):
        cls = self.get_sklearn_like_model(model_class)
        train_data, test_data, score_data, non_score_data = self.read_dataset(field_vals)
        logging.info('Training')
        x, y, _ = make_xy(non_score_data)
        cls.fit(x, y)
        logging.info('Testing')
        tr_x, tr_y, _ = make_xy(test_data)
        # print(pandas.Series(cls.feature_importances_, index=tr_x.columns).sort_values())
        pred = cls.predict(tr_x)
        return rmsle(pred, tr_y), cls

    def main(self, model_class):
        coloredlogs.install(level=logging.INFO)
        # Check we can load the model
        self.get_sklearn_like_model(model_class)

        if self.score_run:
            all_fields = self.get_field_vals()
            # Max val is where we count up to. It will be
            # EITHER exactly the size of all_fields, if all fields is divisible by num_vals
            # OR it will be all_fields + self.num_vals, because of the ceiling operator.
            max_val = int(np.ceil(all_fields.shape[0] / self.num_vals)) * self.num_vals
            for i in range(0, max_val, self.num_vals):
                logging.info('Doing scoring run {} of {}'.format(i, max_val))
                last_ix = i + self.num_vals
                field_vals = all_fields[i:last_ix]
                _, _, score_data, non_score_data = self.read_dataset(field_vals)
                cls = self.get_sklearn_like_model(model_class)
                x, y, _ = make_xy(non_score_data)
                logging.info('Fitting model')
                cls.fit(x, y)
                score_x, _, id = make_xy(score_data)
                logging.info('Predicting with model')
                y = cls.predict(score_x)
                pred = pandas.Series(y, index=id)
                fname = './pred_{}.csv'.format(i)
                logging.info('Writing results to {}'.format(fname))
                pred.to_csv(fname)
        else:
            if self.full_test:
                # Full tests are a little quieter
                coloredlogs.install(level=logging.WARN)
                all_fields = self.get_field_vals()
                max_val = int(np.ceil(all_fields.shape[0] / self.num_vals)) * self.num_vals
                for i in tqdm(range(0, max_val, self.num_vals)):
                    logging.warning('Doing scoring run {} of {}'.format(i, max_val))
                    last_ix = i + self.num_vals
                    field_vals = all_fields[i:last_ix]
                    print(self.training_run(field_vals, model_class)[0])
            else:
                field_vals = self.get_random_field_vals()
                score, cls = self.training_run(field_vals, model_class)
                print(score)
