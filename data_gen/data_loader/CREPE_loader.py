from .data_loader import DataLoader

class CREPELoader(DataLoader):
    def get_question(self, dp: dict, **kwargs) -> str:
        return dp['question']
