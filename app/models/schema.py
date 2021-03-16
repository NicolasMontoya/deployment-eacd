'''Schema of all data received and sent back to the user'''
from pydantic import BaseModel, Field
from datetime import datetime, time, timedelta
from typing import Any

from pydantic.types import PositiveInt
 

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
    name: str = Field(None, title='The name of the todo', max_length=20)
    is_completed: bool = Field(False, title='Determines if the todo is completed or not defaults to False')
    dataset_type: str = Field(None, title='Dataset type')
    url: str = Field(None, title='Dataset location')

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
    name: str = Field(None, title="Name of the model", max_length=14, min_length=6)
    type_model: str = Field(None, title="Model type")
    version: PositiveInt = Field(None, title="Version of the model")
    hiper_params: Any = Field(None, title="Hiper parameters model")
    library: str = Field(None, title="Library model")
    model_library_name: str = Field( None, title="Library model")
    url: datetime = Field(None, title="Ubication url")
    state: str = Field(None, title="State of the model")
    created_date: datetime = Field(None, title="Creation date")

class Prediction(BaseModel):
    '''
    Schema of data returned when a new user is created. Contains name, email, and id
    '''
    y_pred: Any = Field(None, title='Response of the model')

class BikeData(Prediction):
    '''
    BikeData
    '''
    date: datetime = Field(None, title='Date of the prediction')
