from fastapi import APIRouter
from ..models import database

router = APIRouter()

base_url = "/models"

@router.get(base_url, tags=["models"])
async def read_models():
    return [{"username": "Rick"}, {"username": "Morty"}]

@router.get(base_url + "/{dataset_id}", tags=["models"], description="Creation endpoint datasets")
async def read_model_by_id(dataset_id: str):
    return {"username": dataset_id}

@router.post(base_url, tags=["models"])
async def create_model():
    return {"username": "fakecurrentuser"}

@router.put(base_url + "/{dataset_id}", tags=["models"])
async def update_model(dataset_id: str):
    return {"username": dataset_id}