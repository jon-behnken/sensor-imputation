from functools import cached_property
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygrinder import block_missing
from pypots.imputation.lerp import Lerp
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score

class Imputer:
    def __init__(self, data, independent, features, neighbors):
        self.data = data
        self.independent = independent
        self.features = features
        
        self.features.append(independent)
        
        self.neighbors = neighbors
        self.imputer = KNNImputer(n_neighbors=neighbors, missing_values=np.nan)

        # block length corresponds to the interval of data.
        # i.e. if 1 row = 1 hour, then 24 blocks = 24 hours.
        self.block_length = 24

        self.knn_predictions_df = self.transform_to_df(self.knn_predictions)
        self.lerp_predictions_df = self.transform_to_df(self.lerp_predictions)


    @cached_property
    def numpy_data(self):
        self.data['datetime'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour']])
        self.data['datetime_int'] = self.data['datetime'].astype(int)
        # Remove rows with any NaN values
        self.data.dropna(inplace=True)
        self.data.set_index('datetime', inplace=True)

        filtered = self.data[['datetime_int', *self.features]]
                
        numpy_data = filtered.to_numpy()
        
        # (samples, rows, cols)
        return numpy_data[np.newaxis, :, :]
    
    def transform_to_df(self, arr):
        df = pd.DataFrame(arr.squeeze(), columns=['datetime', *self.features])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
        
    @cached_property
    def masked_data(self):
        # only mask the independent variable column 
        masked_data = block_missing(self.numpy_data[:,:,-1:], factor=.05, block_width=1, block_len=self.block_length)
        
        # we now have masked values
        assert np.isnan(masked_data).any()
        
        # insert them back into the full dataset
        copy = np.copy(self.numpy_data)
        copy[:,:,-1:] = masked_data[:,:,:]

        self.masked_data_df = self.transform_to_df(copy)
        
        return copy

    @cached_property
    def knn_predictions(self):
        self.knn_start_time = time.time()
        
        predictions = self.imputer.fit_transform(np.squeeze(self.masked_data))
        predictions = predictions[np.newaxis, :, :]
        
        self.knn_end_time = time.time()
        self.knn_runtime = (self.knn_end_time - self.knn_start_time)

        return predictions
        
    @cached_property
    def lerp_predictions(self):
        raw_predictions = Lerp().impute(dict(X=self.masked_data))
        
        # insert them back into the full dataset
        predictions = np.copy(self.numpy_data)
        predictions[:,:,:] = raw_predictions[:,:,:]
        
        return predictions

    @cached_property
    def null_indices(self):
        # where data is missing
        return np.argwhere(np.isnan(self.masked_data[:,:,-1:]))[:,1]
        
    def knn_r_score(self):
        truth_values = self.numpy_data[:,:,-1].flatten()[self.null_indices]
        predictions = self.knn_predictions[:,:,-1].flatten()[self.null_indices]
        return r2_score(truth_values, predictions)

    def lerp_r_score(self):
        truth_values = self.numpy_data[:,:,-1].flatten()[self.null_indices]
        predictions = self.lerp_predictions[:,:,-1].flatten()[self.null_indices]
        return r2_score(truth_values, predictions)