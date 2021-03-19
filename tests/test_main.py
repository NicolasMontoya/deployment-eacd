import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
id_dataset = None
id_model = None

@pytest.mark.order1
def test_create_dataset():
    dataset = {
      'name': 'Testing',
      'url': 'https://raw.githubusercontent.com/UDEA-Esp-Analitica-y-Ciencia-de-Datos/EACD-06-MACHINE-LEARNING-II/master/local/data/timeseries.csv'
    }
    response = client.post("/datasets", json=dataset)
    content = response.json()
    assert response.status_code == 201
    pytest.id_dataset = content['id']
    assert content['id'] != None
@pytest.mark.order2
def test_read_dataset():
    response = client.get("/datasets")
    assert response.status_code == 200 or response.status_code == 204
    assert len(response.json()) > 0

@pytest.mark.order3
def test_get_by_id__datase():
    response = client.get("/datasets/" + pytest.id_dataset)
    content = response.json()
    assert response.status_code == 200
    assert content['id'] != None

@pytest.mark.order4
def test_create_model():
    model = {
      'name': 'Testing model',
      'type_model': 'RandomForest',
      'version': 1,
      'eval_metric': 'DEFAULT',
      'grid_search': False,
      'dataset': pytest.id_dataset
    }
    response = client.post("/models", json=model)
    content = response.json()
    assert response.status_code == 201
    pytest.id_model = content['id']
    assert content['id'] != None

@pytest.mark.order5
def test_read_main():
    response = client.get("/models")
    assert response.status_code == 200 or response.status_code == 204
    assert len(response.json()) > 0
@pytest.mark.order6
def test_get_by_id():
    response = client.get("/models/" + pytest.id_model)
    content = response.json()
    assert response.status_code == 200
    assert content['id'] != None

@pytest.mark.order7
def test_delete_dataset():
    response = client.delete("/datasets/" + pytest.id_dataset)
    assert response.status_code == 200

@pytest.mark.order8
def test_delete_model():
    response = client.delete("/models/" + pytest.id_model)
    assert response.status_code == 200
