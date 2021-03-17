
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import numpy as np

class FeatureSelectHelper(BaseEstimator, TransformerMixin):
  def __init__(self, selected_features):
    self.selected_features = selected_features
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X_temp = X.copy()
    return X_temp[self.selected_features]
class FeatureEngineering(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def extract_past_y(self):
    start = self.new_features.index[0]
    last = self.new_features.index[-1] - timedelta(hours=1)
    values = np.concatenate(
        (self.new_features.loc[:start, 'cnt'].values,
        self.new_features.loc[:last, 'cnt'].values)
    )
    self.new_features['past_y'] = values
  def rolling_mean(self):
    self.new_features['SMA_cnt_3'] = self.new_features.cnt.rolling(3,center=True).mean()
    self.new_features['SMA_cnt_4'] = self.new_features.cnt.rolling(4,center=True).mean()
  def transform(self, X):
    self.new_features = X.copy()
    self.extract_past_y()
    self.rolling_mean()
    return self.new_features