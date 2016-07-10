"""
This whole package is meant to be models and supporting training
infrastructure. It's quite simple. There's model.py and train.py

The model.py defines `build_model(data)`, a function that
builds a PYMC3 model, and returns:

1. the model object
2. the list of input shared variables,
3. the output random variable

Then, train.py contains:

1. `read_dataset_frame`, a function to read a dataset from the
   database (/tmp/data.sqlite3) and then pull out a random training
   subset
2. `iter_training_batch_frames`, which returns iterated frames from
   `read_dataset_frame`. Note that this is a generator
3. `frame_vector_split`, which splits a frame into the numpy arrays
   arrays that the model is expecting. This should match the shared
   input variables
4. `make_training_minibatch_iterator`, which makes the iterator
   that can be passed in to the training function.

# Todo
- Build in test dataset iterators
- Test the MLM.
- Add more sophistication to the model itself.
"""
import logging
import os

import joblib
import ml_metrics
import pymc3
import sqlite3

from mlm import train, model
from mlm.train import read_test_dataset, frame_vector_split

os.makedirs('/tmp/group-cacho', exist_ok=True)
memory = joblib.Memory('/tmp/gropo-cacho')


# This is _astoundingly_ slow.
@memory.cache()
def calc_dataset_size():
    with sqlite3.connect('/tmp/data.sqlite3') as con:
        res = con.execute('SELECT count(*) FROM data WHERE adjusted_demand is not null AND week_num < 8')
        total_size = res.fetchone()[0]
        return total_size


def main():
    logging.info('Getting sample data')
    unique_items = train.get_dataset_unique_items()
    data = train.read_dataset_sample(1, unique_items)
    logging.info('Calculating total dataset size')
    total_size = calc_dataset_size()
    logging.info('Building model')
    mdl, input_tensors, output_rvs = model.build_model(data, unique_items)

    minibatches = train.make_training_minibatch_iterator(unique_items)

    with mdl:
        logging.info('Doing ADVI batches...')
        v_params = pymc3.variational.advi_minibatch(
            n=10,
            minibatch_tensors=input_tensors,
            minibatch_RVs=output_rvs,
            minibatches=minibatches,
            total_size=total_size,
            n_mcsamples=10,
            verbose=True
        )
        trace = pymc3.variational.sample_vp(v_params, draws=500)
        print(pymc3.summary(trace))

    test_frame = read_test_dataset(unique_items)
    with mdl:
        for i in range(0, test_frame.shape[0], 10000):
            samp = test_frame.ix[i:i+10000]
            frame_parts = frame_vector_split(samp)
            for t, v in zip(input_tensors, frame_parts):
                t.set_value(v)
            samples = pymc3.sample_ppc(trace, samples=500)
            print(ml_metrics.rmsle(samp.adjusted_demand, samples['adjusted_demand'].mean(axis=0)))
