from typing import List
from fastapi.params import Body

from pydantic.fields import Field

from app.models.schema import Model, ModelOutput
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import Response
from ..models import database

router = APIRouter()

base_url = "/models"

@router.get(base_url, tags=["models"], response_model=List[ModelOutput], responses={204: {"description": "No content"} })
async def read_datasets():
    try:
        models = database.Model().get() 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not models:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return models

@router.get(base_url + "/{dataset_id}", tags=["models"], response_model=ModelOutput, responses={204: {"description": "No content"} } )
async def read_dataset_by_id(dataset_id: str):
    try:
        model = database.Model().get_by_id(dataset_id) 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not model:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return  model

@router.post(base_url, tags=["models"], response_model=ModelOutput, status_code=status.HTTP_201_CREATED)
async def create_dataset(model: Model):
    try:
        dataset = database.Model().create(data=model.dict())
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    return dataset

@router.put(base_url + "/{model_id}", tags=["models"])
async def update_dataset(model_id: str =Field(..., description='Id of the model'), data: Model =  Body(...)):
    try:
        dataset = database.Model().update(model_id, data.dict())
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    return  dataset