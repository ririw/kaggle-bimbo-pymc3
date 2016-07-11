import numpy as np

def rmsle(resp, pred):
    n = resp.shape[0]
    return np.sqrt(np.sum(np.square(np.log(pred + 1) - np.log(resp + 1)))/n)