from .data_loader import DataLoader
from ..template import Template

class CREPELoader(DataLoader):
    def get_question(self, dp: dict, template: Template, **kwargs) -> str:
        return template.generate()
