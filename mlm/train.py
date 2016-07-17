import logging
import sqlite3
import pickle

import numpy as np
import pandas


def get_dataset_unique_items():
    try:
        with open("/tmp/unique_items.pkl", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.info('Collecting unique items')
        con = sqlite3.connect('/tmp/data.sqlite3')
        channel_items = {}
        for chan in ['sales_channel', 'sales_depo', 'product_id', 'route_id']:
            logging.info('Collecting for %s', chan)
            res = con.execute('SELECT distinct {} FROM data'.format(chan))
            chan_unique_vals = np.array(res.fetchall()).flatten()
            channel_items[chan] = chan_unique_vals
        con.close()
        with open("/tmp/unique_items.pkl", 'wb') as f:
            pickle.dump(channel_items, f)
        return channel_items


def read_dataset_sample(rand, unique_items):
    logging.info('Gathering full frame')
    frame = pandas.read_csv('/tmp/split_data/0/train/{}.csv'.format(rand))
    frame.sales_channel = frame.sales_channel.astype('category', categories=unique_items['sales_channel'])
    frame.sales_depo = frame.sales_depo.astype('category', categories=unique_items['sales_depo'])
    frame.product_id = frame.product_id.astype('category', categories=unique_items['product_id'])
    frame.route_id = frame.route_id.astype('category', categories=unique_items['route_id'])
    return frame


def read_test_dataset(unique_items):
    logging.info('Gathering test frame')
    frame = pandas.read_csv('/tmp/split_data/0/test/{}.csv'.format(0))
    frame.sales_channel = frame.sales_channel.astype('category', categories=unique_items['sales_channel'])
    frame.sales_depo = frame.sales_depo.astype('category', categories=unique_items['sales_depo'])
    frame.product_id = frame.product_id.astype('category', categories=unique_items['product_id'])
    frame.route_id = frame.route_id.astype('category', categories=unique_items['route_id'])

    return frame


def iter_training_batch_frames(unique_items):
    while True:
        frame = read_dataset_sample(np.random.randint(100), unique_items)
        yield frame


def frame_vector_split(frame):
    return [
        frame.sales_channel.cat.codes.values.astype(np.int8),
        frame.sales_depo.cat.codes.values.astype(np.int16),
        frame.product_id.cat.codes.values.astype(np.int16),
        frame.route_id.cat.codes.values.astype(np.int16),
        frame.adjusted_demand.values.astype(np.float32),
    ]


def make_training_minibatch_iterator(unique_items):
    return map(frame_vector_split, iter_training_batch_frames(unique_items))
