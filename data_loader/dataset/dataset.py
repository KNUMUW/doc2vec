import pandas as pd 
import multiprocessing as mp 
from data_loader.file_util import FileUtil


class Dataset:
    """A base class for dataset access endpoint objects.

    Attributes:
        _data_path (str): path to the directory containing dataset files.
        _file_util (data_loader.file_util.FileUtil): utility object dealing with loading files.
    
    """

    
    def __init__(self, data_path):
        self._data_path = data_path
        self._file_util = FileUtil()

     
    def _build_dataframe(self, file_paths, workers=mp.cpu_count()):
        """Builds appropriate dataframe from all given data files.

        Args:
            file_paths (list(str)): paths to all data files we want to wrap as a dataframe.
            workers (int): number of processes to be spawned to build the dataframe.            
        Returns:
            dataframe (pandas.DataFrame): the resulting dataframe.            

        """
        
        try:
            chunksize = int(len(file_paths) / workers)
            pool = mp.Pool(processes=workers)              
            results = pool.map(self._file_util.load_single_file, file_paths, chunksize)
            pool.close()
            pool.join()
        except NotImplementedError:
            raise NotImplementedError('Cannot call this method on base class instance!')
        
        # Flatten resulting lists if necessary.
        contents = [res[0] if isinstance(res[0], list) else [res[0]] for res in results] 
        contents = [el for sub_list in contents for el in sub_list]
        
        labels = [res[1] if isinstance(res[1], list) else [res[1]] for res in results]       
        labels = [el for sub_list in labels for el in sub_list]
        
        data_dict = {}
        data_dict['document'] = contents     
        data_dict['label'] = labels

        return pd.DataFrame.from_dict(data_dict)


    def get_dataset(self):
        """Returns particular dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        
        raise NotImplementedError('Cannot call this method on base class instance!')

