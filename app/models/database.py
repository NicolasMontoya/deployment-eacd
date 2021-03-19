from datetime import datetime
from faunadb import query as q
from faunadb.client import FaunaClient
from faunadb.errors import NotFound
from dotenv import load_dotenv
from typing import Dict
import os, secrets
import pytz

load_dotenv()

client = FaunaClient(secret=os.getenv('FAUNA_SECRET'))
indexes = client.query(q.paginate(q.indexes()))

print(indexes) # Returns an array of all index created for the database.

class BaseConnection:

    def __init__(self, collection) -> None:
        self.collection_name = collection
        self.collection = q.collection(collection)

    def create(self, data) -> Dict[str, str]:
        new_data = client.query(
            q.create(
                self.collection,
                {'data': {**data, 'created_date': datetime.now(pytz.utc), 'last_updated_date': datetime.now(pytz.utc), 'id': secrets.token_hex(12)}}
            )
        ) 
        return new_data['data']

    def update(self, id, data):
        try:
            return client.query(
                q.update(
                    q.select("ref", q.get(
                        q.match(q.index(self.collection_name + "_by_id"), id)
                    )),
                    {'data': {**data, 'last_updated_date': datetime.now(pytz.utc)}}
                )
            )['data']
        except NotFound:  
            return 'Not found'
    
    def delete(self, id):
        try:
            return client.query(
                q.delete(
                    q.select("ref", q.get(
                        q.match(q.index(self.collection_name + "_by_id"), id)
                    ))
                )
            )['data']
        except NotFound:  
            return 'Not found'

    def get(self):
        try:
            datasets = client.query(
                q.paginate(
                    q.match(q.index("all_"+ self.collection_name ))
                )
            )
            return [
                        client.query(
                            q.get(q.ref(self.collection, dataset.id()))
                        )['data']  
                        for dataset in datasets['data']
                    ] 
        except NotFound:
            return None

    def get_by_id(self, id):
        try:
            dataset = client.query(
                q.get(q.match(q.index(self.collection_name +"_by_id"), id))
            )
        except NotFound:
            return None
        return None if dataset.get('errors') else dataset['data']

class Dataset(BaseConnection):
    def __init__(self):
        super().__init__('datasets')

class Model(BaseConnection):
    def __init__(self):
        super().__init__('models')

class Prediction(BaseConnection):
    def __init__(self):
        super().__init__('predictions')