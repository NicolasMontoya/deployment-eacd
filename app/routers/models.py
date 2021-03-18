from datetime import datetime
from typing import List
from fastapi.params import Body

from pydantic.fields import Field

from app.models.schema import Model, ModelOutput
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import Response, JSONResponse
from ..models import database
from ..services.model_service import TrainerModel


router = APIRouter()

base_url = "/models"


@router.get(base_url, tags=["models"], response_model=List[ModelOutput], responses={204: {"description": "No content"} })
async def read_models():
    try:
        models = database.Model().get() 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not models:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return models

@router.get(base_url + "/{model_id}", tags=["models"], response_model=ModelOutput, responses={204: {"description": "No content"} } )
async def read_model_by_id(model_id: str):
    try:
        model = database.Model().get_by_id(model_id) 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not model:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return  model

@router.post(base_url, tags=["models"], response_model=ModelOutput, status_code=status.HTTP_201_CREATED)
async def create_model(model: Model):
    try:
        dataset = database.Dataset().get_by_id(model.dataset)
        if dataset is None:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content="Dataset is not allowed")
        dict_ = {**model.dict(), 'state': 'PROGRESS'}
        model_response = database.Model().create(data=dict_)
        model_ = TrainerModel(model_response, dataset['url'])
        presition = model_.train()
        dict_ = {**model.dict(), 'state': 'READY', 'score': presition}
        database.Model().update(model_response['id'], dict_)
        model_response['score'] = presition
    except Exception as e:
        database.Model().delete(model_response['id'])
        raise HTTPException(400, detail=str(e))
    return model_response

@router.get(base_url + "/{model_id}", tags=["models"], response_model=ModelOutput, responses={204: {"description": "No content"} } )
async def model_predict(model_id: str, date: datetime):
    try:
        model = database.Model().get_by_id(model_id) 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not model:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return  model

@router.delete(base_url + "/{model_id}", tags=["models"])
async def update_model(model_id: str =Field(..., description='Id of the model'), data: Model =  Body(...)):
    try:
        model = database.Model().delete(model_id)
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    return  model