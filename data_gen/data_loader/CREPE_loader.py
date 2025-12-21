from .data_loader import DataLoader
import json
from typing import List

class CREPELoader(DataLoader):
    def save_data(self, data: List[dict], split: str, out_path: str = None, model_name: str = None, **kwargs):
        if out_path is None or model_name is None:
            with open(self.datasetname2path[self.dataset_name][split].format(self.file_dir), 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp) + '\n')
        elif model_name:
            with open(self.datasetname2path[self.dataset_name][split].format(self.file_dir, model_name), 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp) + '\n')
        else:
            with open(out_path, 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp) + '\n')