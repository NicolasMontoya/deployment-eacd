from typing import Dict
from faunadb.errors import HttpError
from app.models.schema import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import requests, os, logging
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

logger = logging.getLogger(__name__) 

class TrainerModel:
  def __init__(self, model: Dict, dataset_url: str, params=None) -> None:
    logger.info("starting trainer...")
    self.model_definition = model
    self.dataset_url = dataset_url
    if model['type_model'] == 'RF':
      if params == None:
        self.model = RandomForestRegressor()
      else:
        self.model = RandomForestRegressor(**params)
    else:
      if params == None:
        self.model = LinearRegression()
      else:
        self.model = LinearRegression(**params)
    download_dataset(dataset_url, model['dataset'])
  def train(self, target: str, special: bool=True, test_size=0.3, random_state=17):
    df = self.load_dataset()
    y = df[target]
    X = df.drop(columns=[target, 'dteday'])
    if special:
      train_indices = X["yr"] == 0
      X_train, y_train = X[train_indices], y[train_indices]
      X_test, y_test = X[~train_indices], y[~train_indices]
    else:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    self.model.fit(X_train, y_train)
    dump(self.model, f"{os.getcwd()}/app/models/ml/{self.model_definition['id']}")
    return self.model.score(X_test, y_test)
  def load_dataset(self):
    url = f"{os.getcwd()}/app/datasets/{self.model_definition['dataset']}.csv"
    if not os.path.exists(url):
      download_dataset(self.dataset_url, self.model_definition['dataset'])
    return pd.read_csv(url)

def download_dataset(url, dataset_id):
    if not os.path.exists(f"{os.getcwd()}/app/datasets/{dataset_id}.csv"):
      logger.info("Replicating database resources...")
      try:
          req = requests.get(url)
          url_content = req.content
          csv_file = open(f'{os.getcwd()}/app/datasets/{dataset_id}.csv', 'wb')
          csv_file.write(url_content)
          csv_file.close()
      except requests.exceptions.HTTPError as err:
          raise HttpError("Download, please verify the URL")
    logger.info("Resources ready...")