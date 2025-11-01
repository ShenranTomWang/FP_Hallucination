import json, importlib
from abc import ABC, abstractmethod

datasetname2loader = {
    'movies': 'WikidataLoader',
    'CREPE': 'CREPELoader'
}

def instantiate_dataloader(dataset_name: str, file_dir: str = 'dataset', model_name: str = None) -> 'DataLoader':
    loader_class_name = datasetname2loader[dataset_name]
    module = importlib.import_module('data_gen.data_loader')
    cls = getattr(module, loader_class_name)
    return cls(dataset_name, file_dir, model_name=model_name)

class DataLoader(ABC):
    datasetname2path = {
        'movies': '{}/toy_dataset/movies/wikidata_movies.json',
        'CREPE': {
            'train': '{}/CREPE/train.jsonl',
            'dev': '{}/CREPE/dev.jsonl',
            'test': '{}/CREPE/test.jsonl',
            'CREPE_Presupposition_Extraction': '{}/curated_dataset_{}_CREPE_Presupposition_Extraction.jsonl',
        }
    }
    
    def __init__(self, dataset_name: str, file_dir: str = 'dataset', model_name: str = None):
        self.dataset_name = dataset_name
        self.file_dir = file_dir
        self.model_name = model_name

    def load_data(self, split: str = None) -> list[dict]:
        data = []
        if split and split != 'CREPE_Presupposition_Extraction':
            split_path = self.datasetname2path[self.dataset_name][split].format(self.file_dir)
            with open(split_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif split == 'CREPE_Presupposition_Extraction':
            split_path = self.datasetname2path[self.dataset_name][split].format(self.file_dir, self.model_name)
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
    def save_data(self, data: list[dict], out_path: str = None, **kwargs):
        pass
        