import importlib
from ..template import Template

class WikidataLoader:
    datasetname2correct_key = {
        'movies': 'time'
    }
    
    datasetname2fp_key = {
        'movies': 'time_fp'
    }

    def get_templates(self, templates_type: str) -> list[Template]:
        try:
            templates = getattr(importlib.import_module(f"data_gen.{self.dataset_name}.templates"), templates_type)
            return templates
        except ImportError:
            print(f"No templates found for dataset {self.dataset_name}.")
            return []

    def get_correct_key(self) -> str:
        return self.datasetname2correct_key.get(self.dataset_name)

    def get_fp_key(self) -> str:
        return self.datasetname2fp_key.get(self.dataset_name)
    
    def get_question(self, dp: dict, template: Template, **kwargs) -> str:
        return template(**dp).generate()