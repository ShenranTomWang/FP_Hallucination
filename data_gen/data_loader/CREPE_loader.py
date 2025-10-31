from .data_loader import DataLoader
from .template import Template
from typing import Type
import json

class CREPELoader(DataLoader):
    def get_question(self, dp: dict, template: Type[Template], **kwargs) -> str:
        return template(**dp).generate(**kwargs)
    
    def save_data(self, data: list[dict], split: str, out_path: str = None):
        if out_path is None:
            with open(self.datasetname2path[self.dataset_name][split].format(self.file_dir), 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp) + '\n')
        else:
            with open(out_path, 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp) + '\n')