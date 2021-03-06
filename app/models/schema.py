'''Schema of all data received and sent back to the user'''
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, time, timedelta
from typing import Any

from pydantic.types import PositiveInt
 
class ModelType(str, Enum):
     RF = 'RandomForest'
     LR = 'LinearRegression'

class EvalMetric(str, Enum):
     Q2 = 'DEFAULT'
     CT = 'CUSTOM'

class User(BaseModel):
    '''Base User schema'''
    email: str = Field(
        None, title="The email of the user", max_length=256, 
        regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    )
    password: str = Field(
        None, title="The password of the user", max_length=20,
        regex=r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[a-zA-Z]).{8,}$"
    )

class Dataset(BaseModel):
    '''
    Schema of dataset
    '''
    name: str = Field(..., title='The name of the todo', max_length=20)
    url: str = Field(..., title='Dataset location')

class DatasetOutput(Dataset):
    '''
    Schema of dataset
    '''
    id: str = Field(None, title='Unique identifier')
    created_date: Any = Field(None, title="Creation date")
    last_updated_date: Any = Field(None, title="Modification date")

class Model(BaseModel):
    '''
    Schema of the definition model
    '''
    name: str = Field(..., title="Name of the model", max_length=14, min_length=6)
    type_model: ModelType = Field(..., title="Model type")
    version: int = Field(..., title="Version of the model", gt=0)
    params: Any = Field(None, title="Parameters models")
    hiper_params: Any = Field(None, title="Hiper parameters models")
    eval_metric: EvalMetric = Field(..., title="Default metric")
    grid_search: bool = Field(False, title="Search best parameters")
    dataset: str = Field(None, title="dataset")
class ModelOutput(Model):
    '''
    Schema of model
    '''
    id: str = Field(None, title='Unique identifier')
    state: str = Field(None, title="State of the model")
    created_date: Any = Field(None, title="Creation date")
    last_updated_date: Any = Field(None, title="Modification date")
    score: float = Field(None, title="Model score")

class PredictOutput(BaseModel):
    '''
    Schema of model
    '''
    eval_metric: EvalMetric = Field(..., title="Model metric")
    score: float = Field(None, title="Model score")
    datetime: str = Field(None, title="Date input")
    predict: float = Field(None, title="Prediction")

