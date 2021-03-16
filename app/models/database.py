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

class Dataset:

    def __init__(self) -> None:
        self.collection = q.collection('datasets')

    def create_dataset(self, data) -> Dict[str, str]:
        new_data = client.query(
            q.create(
                self.collection,
                {'data': {**data, 'created_date': datetime.now(pytz.utc), 'last_updated_date': datetime.now(pytz.utc), 'id': secrets.token_hex(12)}}
            )
        ) 
        return new_data['data']

    def update_dataset(self, id, data):
        try:
            return client.query(
                q.update(
                    q.select("ref", q.get(
                        q.match(q.index("dataset_by_id"), id)
                    )),
                    {'data': {**data, 'last_updated_date': datetime.now(pytz.utc)}}
                )
            )['data']
        except NotFound:  
            return 'Not found'

    def get_datasets(self):
        try:
            datasets = client.query(
                q.paginate(
                    q.match(q.index("all_datasets"))
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

    def get_dataset(self, id):
        try:
            dataset = client.query(
                q.get(q.match(q.index("dataset_by_id"), id))
            )
        except NotFound:
            return None
        return None if dataset.get('errors') else dataset['data']
