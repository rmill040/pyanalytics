from __future__ import division

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd


__all__ = ["sliding_window"]

    
def sliding_window(data_set = None, window_size_seconds = None, stride_seconds = None, 
                   sampling_rate = None, label_name = None, labeling = 'last_label'):
    """Create sliding windows and create labels for each window

    Parameters
    ----------
    data_set : pandas DataFrame
        Data set that will be used as input

    window_size_seconds : int
        Value specifying the window size

    stride_seconds : int
        Value specifying the overlap between windows

    sampling_rate : int
        Sampling rate in Hz of data contained in data_set

    label_name : str
        Name of label in data_set

    labeling : str
        Method to use for extracting labels in sliding window

    Returns
    -------
    windows : list
        List of sliding windows

    labels : 1d array-like
        Labels for each sliding window in windows list
    """
    # Get stride factor and window size
    total_frames = int(data_set.shape[0] - 1)
    stride_factor = int(np.ceil(stride_seconds * sampling_rate))
    window_size = int(np.ceil(window_size_seconds * sampling_rate))

    # Error checking
    assert total_frames > stride_factor and total_frames > window_size, "Strides and/or window size too large"

    # Number of windows
    condition = (window_size >= stride_factor)
    n_windows = int(np.ceil(total_frames/stride_factor) - np.floor(window_size/stride_factor) + condition)

    # Get indices
    start_idx = stride_factor*(np.arange(0, n_windows - 1) + 1).astype('int')
    start_idx = np.insert(start_idx, 0, 0).astype('int') 
    end_idx = (start_idx[1:] + window_size - 1).astype('int')
    end_idx = np.insert(end_idx, 0, start_idx[0] + window_size - 1).astype('int')

    # Check to make sure ending index is equal to the number of rows in data set
    if end_idx[-1] > total_frames:
        end_idx[-1] = total_frames

    # Create sliding windows
    windows = [data_set.iloc[start_idx[i]:end_idx[i]+1] for i in xrange(n_windows)]
    
    # Make sure last window is same size as other windows, if not, remove it
    if len(windows[-1]) != window_size:
        windows.pop(-1)

    # Get labels if label variable provided (for example with training data)
    # If no labels, only return sliding windows for data_set (for example with testing data)
    if labels is not None:
        # Create labels for each window
        labels = np.zeros(len(windows))
        if labeling == 'last_label':
            # Create labels for each window by taking the last Event label in a window as the overall window label
            for i in xrange(labels.shape[0]):
                labels[i] = windows[i][label_name][-1]

        elif labeling == 'majority_label':
            from scipy.stats import mode
            # Create labels for each window by taking the mode of the class labels in the window as the overall window label; if tie choose 
            # among class labels randomly
            for i in xrange(labels.shape[0]):
                labels[i] = float(mode(windows[i][label_name])[0])
        else:
            raise ValueError('% not a valid labeling method. Valid methods are last_label and majority_label')

        return windows, labels

    else:
        return windows