import json, importlib

class DataLoader:
    datasetname2path = {
        'movies': 'dataset/toy_dataset/movies/wikidata_movies.json'
    }
    
    datasetname2correct_key = {
        'movies': 'time'
    }
    
    datasetname2fp_key = {
        'movies': 'time_fp'
    }
    
    def __init__(self, dataset_name: str, file_path: str):
        self.dataset_name = dataset_name
        self.file_path = file_path

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_templates(self, templates_type: str):
        templates = getattr(importlib.import_module(f"data_gen.{self.dataset_name}.templates"), templates_type)
        return templates

    def get_correct_key(self):
        return self.datasetname2correct_key.get(self.dataset_name)

    def get_fp_key(self):
        return self.datasetname2fp_key.get(self.dataset_name)