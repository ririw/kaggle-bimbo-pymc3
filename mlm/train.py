import logging
import sqlite3

import numpy as np
import pandas


def read_dataset_frame(ix):
    logging.info('Gathering frame data for rand %d', ix)
    con = sqlite3.connect('/tmp/data.sqlite3')
    try:
        frame = pandas.read_sql('''
                SELECT week_num,
                       sales_channel,
                       sales_depo,
                       adjusted_demand
                  FROM data
                 WHERE week_num < 8
                       and adjusted_demand is not NULL
                       and rand = ?
            ''', con=con, params=[ix])
        frame.week_num = frame.week_num.astype('category')
        frame.sales_channel = frame.sales_channel.astype('category')
        frame.sales_depo = frame.sales_depo.astype('category')
        return frame
    finally:
        con.close()


cache = None


def read_dataset_fully():
    global cache
    if cache is None:
        logging.info('Gathering full frame')
        con = sqlite3.connect('/tmp/data.sqlite3')
        frame = pandas.read_sql('''
                SELECT week_num,
                       sales_channel,
                       sales_depo,
                       adjusted_demand
                  FROM data
                 WHERE week_num < 8
                       and adjusted_demand is not NULL
            ''', con=con)
        frame.week_num = frame.week_num.astype('category')
        frame.sales_channel = frame.sales_channel.astype('category')
        frame.sales_depo = frame.sales_depo.astype('category')
        con.close()
        cache = frame
    return cache.sample(1000)


def iter_training_batch_frames():
    while True:
        ix = np.random.randint(100)
        frame = read_dataset_fully()
        yield frame


def frame_vector_split(frame):
    return [
        frame.week_num.cat.codes.values.astype(np.int8),
        frame.sales_channel.cat.codes.values.astype(np.int8),
        frame.sales_depo.cat.codes.values.astype(np.int16),
        frame.adjusted_demand.values.astype(np.float16)
    ]


def make_training_minibatch_iterator():
    return map(frame_vector_split, iter_training_batch_frames())
