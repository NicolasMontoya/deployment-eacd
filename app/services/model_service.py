from datetime import date, datetime
from typing import Dict
from faunadb.errors import HttpError
import requests, os, logging
from joblib import dump
import pandas as pd
from io import StringIO
from random import randint
from .helpers import FeatureEngineering, FeatureSelectHelper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
import numpy as np
import holidays


logger = logging.getLogger(__name__) 
REQUIRED_COLUMNS = list(['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','cnt'])
PROCESS_COLUMNS = list(['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed'])
class TrainerModel:

  def __init__(self, model: Dict, dataset_url: str, hiper_params=None, params=None) -> None:
    self.model_definition = model
    self.dataset_url = dataset_url
    self.define_model(model, hiper_params, params)
    download_dataset(dataset_url, model['dataset'])

  def define_model(self, model, hiper_params, params):
    if model['type_model'] == 'RF':
      if params == None:
        model_ = RandomForestRegressor()
      else:
        model_ = RandomForestRegressor(**params)
    else:
      if params == None:
        model_ = LinearRegression()
      else:
        model_ = LinearRegression(**params)
    model_ = Pipeline([
      ("extractor", FeatureEngineering()),
      ("helper", FeatureSelectHelper(PROCESS_COLUMNS) ),
      ("scaler", StandardScaler()),
      ("model", model_)
    ])
    if model['grid_search'] == True and model['hiper_params'] is not None:
      self.model = GridSearchCV(
        estimator=model_, param_grid={**hiper_params}, scoring=self.get_eval()
      )
    elif model['grid_search'] == True and model['hiper_params'] is None:
      raise ValueError('You need to include hiper_params, if you select grid_search')
    else:
      self.model = model_

  def get_eval(self):
    if self.model_definition['eval_metric'] == 'DEFAULT':
      return 'mean_absolute_error'
    else:
      return make_scorer(self.bike_number_error)
  
  def bike_number_error(y_true, y_pred, understock_price=0.3, overstock_price=0.7):
    error = y_true - y_pred
    return sum(map(lambda err: overstock_price*err*(-1) if err < 0 else understock_price*err , error)) / len(error)

  def train(self):
    df = self.load_dataset()
    y = df['cnt']
    X = df

    train_indices = X["yr"] == 0
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[~train_indices], y[~train_indices]

    self.model.fit(X_train, y_train)
    dump(self.model, f"{os.getcwd()}/app/models/ml/{self.model_definition['id']}")
    if self.model_definition['eval_metric'] == 'CUSTOM':
      return self.bike_number_error(y_test, self.model.predict(X_test))
    return mean_absolute_error(y_test, self.model.predict(X_test))

  def load_dataset(self):
    url = f"{os.getcwd()}/app/datasets/{self.model_definition['dataset']}.csv"
    if not os.path.exists(url):
      download_dataset(self.dataset_url, self.model_definition['dataset'])
    df = pd.read_csv(url)
    df['datetime'] =  pd.to_datetime(df['dteday']) + df['hr'].apply(pd.Timedelta, unit='h')
    df_time = df.set_index('datetime')
    df_time = df_time.reindex(pd.date_range(
            start=df_time.index[0],
            end=df_time.index[-1],
            freq='1H'
        )
    )
    df_time = clean_dataframe(df_time)
    return df_time

def download_dataset(url, dataset_id):
    if not os.path.exists(f"{os.getcwd()}/app/datasets/{dataset_id}.csv"):
      logger.info("Replicating database resources...")
      try:
          req = requests.get(url)
          url_content = req.content
          df = pd.read_csv(StringIO(url_content.decode("utf-8")))
          for field in REQUIRED_COLUMNS:
            if field not in df.columns:
              raise ValueError("Dataset is not allowed")
          df.to_csv(f'{os.getcwd()}/app/datasets/{dataset_id}.csv', index=False)
      except requests.exceptions.HTTPError as err:
          raise HttpError("Download, please verify the URL")
    logger.info("Resources ready...")

def clean_dataframe(df):
  df_imp = df.drop(columns=['instant'])
  null_indexes = df_imp[df_imp.isna().any(axis=1)].index
  Y = 2000
  seasons = [
            (1, (date(Y,  1, 1),  date(Y,  6, 20))),
            (2, (date(Y,  6, 21),  date(Y,  9, 22))),
            (3, (date(Y,  9, 23),  date(Y, 12, 20))),
            (4, (date(Y, 12, 21),  date(Y, 12, 31)))]

  def get_season(now):
      if isinstance(now, datetime):
          now = now.date()
      now = now.replace(year=Y)
      return next(season for season, (start, end) in seasons
                  if start <= now <= end)

  for index in null_indexes:
    df_imp.loc[index, ['hr', 'yr', 'mnth']] = (index.hour, int(index.year == 2012), index.month)
    df_imp.holiday.loc[index] = int(index in holidays.UnitedStates())
    df_imp.weekday.loc[index]  = index.dayofweek
    df_imp.workingday.loc[index]  = int(index.dayofweek in [0,1,2,3,4])
    df_imp.season.loc[index]  = get_season(index)
    df_imp.weathersit.loc[index]  = randint(1, 4)

  df_imp.temp = df_imp.temp.interpolate(method='time').round(2)
  df_imp.atemp = df_imp.atemp.interpolate(method='time').round(2)
  df_imp.hum = df_imp.hum.interpolate(method='time').round(2)
  df_imp.windspeed = df_imp.windspeed.interpolate(method='time').round(2)
  df_imp.casual = df_imp.casual.interpolate(method='time').apply(np.floor)
  df_imp.registered = df_imp.registered.interpolate(method='time').apply(np.floor)

  df_imp.season = df_imp.season.astype(np.int64)
  df_imp.yr = df_imp.yr.astype(np.int64)
  df_imp.mnth = df_imp.mnth.astype(np.int64)
  df_imp.hr = df_imp.hr.astype(np.int64)
  df_imp.holiday = df_imp.holiday.astype(np.int64)
  df_imp.weekday = df_imp.weekday.astype(np.int64)
  df_imp.workingday = df_imp.workingday.astype(np.int64)
  df_imp.weathersit = df_imp.weathersit.astype(np.int64)

  df_imp.cnt =  df_imp.casual + df_imp.registered
  return df_imp