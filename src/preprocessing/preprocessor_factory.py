from typing import Any
from .fugaku_preprocessor import FugakuPreprocessor
from .eagle_preprocessor import EaglePreprocessor
from .sandia_preprocessor import SandiaPreprocessor
from .bu_preprocessor import BUPreprocessor
from .m100_preprocessor import M100Preprocessor


class PreprocessorFactory:
    """Factory class to create the appropriate preprocessor for a given dataset."""
    
    @staticmethod
    def get_preprocessor(dataset_name: str) -> Any:
        """
        Get the preprocessor for a specific dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to process
            
        Returns:
        --------
        preprocessor : Preprocessor instance
            The appropriate preprocessor for the dataset
        """
        preprocessors = {
            'Fugaku': FugakuPreprocessor,
            'Eagle': EaglePreprocessor,
            'Sandia': SandiaPreprocessor,
            'BU': BUPreprocessor,
            'M100': M100Preprocessor
        }
        
        if dataset_name not in preprocessors:
            available_datasets = ', '.join(preprocessors.keys())
            raise ValueError(f"No preprocessor available for dataset: {dataset_name}. Available options are: {available_datasets}")
        
        return preprocessors[dataset_name]()