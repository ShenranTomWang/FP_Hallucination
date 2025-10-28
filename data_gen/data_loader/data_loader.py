import json, importlib
from abc import ABC, abstractmethod

datasetname2loader = {
    'movies': 'WikidataLoader',
    'CREPE': 'CREPELoader'
}

def instantiate_dataloader(dataset_name: str, file_dir: str = 'dataset'):
    loader_class_name = datasetname2loader[dataset_name]
    module = importlib.import_module('data_gen.data_loader')
    cls = getattr(module, loader_class_name)
    return cls(dataset_name, file_dir)

class DataLoader(ABC):
    datasetname2path = {
        'movies': '{}/toy_dataset/movies/wikidata_movies.json',
        'CREPE': {
            'train': '{}/CREPE/train.jsonl',
            'dev': '{}/CREPE/dev.jsonl',
            'test': '{}/CREPE/test.jsonl',
        }
    }
    
    def __init__(self, dataset_name: str, file_dir: str = 'dataset'):
        self.dataset_name = dataset_name
        self.file_dir = file_dir

    def load_data(self, split: str = None) -> list[dict]:
        data = []
        if split:
            split_path = self.datasetname2path[self.dataset_name][split].format(self.file_dir)
            with open(split_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            path = self.datasetname2path[self.dataset_name].format(self.file_dir)
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        return data

    @abstractmethod
    def get_question(self, dp: dict, **kwargs) -> str:
        pass
    
    @abstractmethod
    def save_data(self, data: list[dict], out_path: str = None, **kwargs):
        pass
        