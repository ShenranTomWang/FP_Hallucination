from .data_loader import DataLoader, instantiate_dataloader
from .wikidata_loader import WikidataLoader
from .CREPE_loader import CREPELoader

__all__ = ['DataLoader', 'WikidataLoader', 'CREPELoader', 'instantiate_dataloader']