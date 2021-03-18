from typing import Dict, List
from fastapi.params import Body

from pydantic.fields import Field

from app.models.schema import Dataset, DatasetOutput
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import Response
from ..models import database
from ..services.model_service import download_dataset

router = APIRouter()

base_url = "/datasets"

@router.get(base_url, tags=["datasets"], response_model=List[DatasetOutput], responses={204: {"description": "No content"} })
async def read_datasets():
    try:
        datasets = database.Dataset().get() 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not datasets:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return datasets

@router.get(base_url + "/{dataset_id}", tags=["datasets"], response_model=DatasetOutput, responses={204: {"description": "No content"} } )
async def read_dataset_by_id(dataset_id: str):
    try:
        dataset = database.Dataset().get_by_id(dataset_id) 
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    if not dataset:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return  dataset

@router.post(base_url, tags=["datasets"], response_model=DatasetOutput, status_code=status.HTTP_201_CREATED)
async def create_dataset(dataset: Dataset):
    try:
        dataset_response: Dict = database.Dataset().create(data=dataset.dict())
        download_dataset(dataset_response['url'], dataset_response['id'])
    except Exception as e:
        database.Dataset().delete(dataset_response['id'])
        raise HTTPException(400, detail=str(e))
    return dataset_response

@router.delete(base_url + "/{dataset_id}", tags=["datasets"])
async def update_dataset(dataset_id: str =Field(..., description='Id of the todo')):
    try:
        dataset = database.Dataset().delete(dataset_id)
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    return  dataset