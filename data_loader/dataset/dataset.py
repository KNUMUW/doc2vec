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
        self._data_path = data_path


    @classmethod
    def _build_dataframe(cls, file_paths):
        """Builds appropriate dataframe from all given data files.

        This implementation is provided for datasets that are composed of many individual data files and train/test split
        is performed at file level. If data level split is necessary, then it dataframe building should be performed differently.  

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
        
        This implementation is provided for datasets that are composed of many individual data files and train/test split 
        can be performed at file level. If a dataset is composed of huge files and fine grained split is necessary, then it 
        should be performed at data level and implemented differently.
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

