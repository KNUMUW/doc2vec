import os
import logging
import pandas as pd
import multiprocessing as mp
from abc import ABC, abstractmethod
from data_loader.util import get_files_from_dir


class Dataset(ABC):
    """An abstract base class for dataset access endpoint objects.

    Attributes:
        _data_path (str): path to the directory containing dataset files.    

    """    

    def __init__(self, data_path):
        if os.path.exists(data_path):
            self._data_path = data_path
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
        else:
            raise FileNotFoundError('Data directory not found.')

    def _get_file_paths(self):
        """Returns paths to files that make up training and test set, respectively.
        
        This method is provided here only as a reference, because it is called in get_dataset method below. 
        It is not mandatory if your get_dataset method does not need it. However, when it does (e.g. because 
        you call super method in get_dataset), then this method must be implemented in your derived class 
        in accordance with the dataset structure.

        Returns:
            train_set_paths (list(str)): paths to training data files.
            test_set_paths (list(str)): paths to test data files.

        """       
        raise NotImplementedError('_get_file_paths method must be implemented!')        

    @staticmethod
    def _load_single_file(path):
        """Loads a single file from given dataset.

        This method is provided here only as a reference, because it is called in _build_dataframe method below. 
        It is not mandatory unless you use some of below implementations (e.g. you call super method in your
        get_dataset method). In that case this method must be implemented in your derived class in accordance with 
        the dataset structure.
        
        Args:
            path (str): path to the file to be loaded.       
        Returns:
            content (str): content of the document.
            label (type): label of the document.
 
        """
        raise NotImplementedError('_load_single_file method must be implemented!')
 
    @classmethod
    def _build_dataframe(cls, file_paths):
        """Builds appropriate dataframe from all given data files.

        This implementation is provided for datasets that are composed of many individual data files and 
        train/test split is performed at file level. If data level split is necessary, then dataframe build 
        should be performed differently.  

        Args:
            file_paths (list(str)): paths to all data files we want to wrap as a dataframe.
            workers (int): number of processes to be spawned to build the dataframe.            
        Returns:
            dataframe (pandas.DataFrame): the resulting dataframe.            
       
         """
        # Get file contents and labels in parallel.
        chunksize = int(len(file_paths) / mp.cpu_count())

        pool = mp.Pool(processes=mp.cpu_count())              
        results = pool.map(cls._load_single_file, file_paths, chunksize)

        pool.close()
        pool.join()
   
        # Build the dataframe.           
        data_dict = {}
        data_dict['document'] = [res[0] for res in results]         
        data_dict['label'] = [res[1] for res in results]       

        return pd.DataFrame.from_dict(data_dict)

    @abstractmethod 
    def get_dataset(self):
        """Returns given dataset. 
        
        This implementation is provided for datasets that are composed of many individual data files and 
        train/test split can be performed at file level. If a dataset is composed of huge files and 
        fine grained split is necessary, then it should be performed at data level and implemented differently.
        Subclasses must implement this method.
 
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        # Get lists of file paths for training and test sets, respectively.
        train_set_paths, test_set_paths = self._get_file_paths()        

        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_set_paths)
        test_set = self._build_dataframe(test_set_paths) 

        return train_set, test_set            

