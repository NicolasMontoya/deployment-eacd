
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
    self.new_features['SMA_cnt_30'] = self.new_features.cnt.rolling(30,center=True,min_periods=1).mean()
    self.new_features['SMA_cnt_60'] = self.new_features.cnt.rolling(60,center=True,min_periods=1).mean()
  def rolling_median(self):
    self.new_features['median_cnt_30'] = self.new_features.cnt.rolling(10,center=True,min_periods=1).median()
    self.new_features['median_cnt_15'] = self.new_features.cnt.rolling(20,center=True,min_periods=1).median()
  def last_day(self):
    second_day_start = self.new_features.index[0] + timedelta(hours=23)
    last_day_start = self.new_features.index[-1] - timedelta(days=1)
    values = np.concatenate(
        (self.new_features.loc[:second_day_start,'cnt'].values,
        self.new_features.loc[:last_day_start,'cnt'].values)
    )
    self.new_features['last_day_same_hour'] = values
  def transform(self, X):
    self.new_features = X.copy()
    self.extract_past_y()
    return self.new_features