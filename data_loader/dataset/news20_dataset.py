import os
import pandas as pd
from data_loader.dataset.dataset import Dataset
from data_loader.file_util import FileUtil


class News20FileUtil(FileUtil):
    """Utility class for accessing files in IMDB movie reviews dataset."""    


    @staticmethod
    def load_single_file(path):
        """Loads a single file from 20 Newsgroup dataset.

        Args:
            path (str): path to the file to be loaded. 
        Returns:
            content (str): content(s) of the document.
            label (str): label (topic) of the document.
 
        """

        label = os.path.basename(os.path.dirname(path))     
         
        with open(path, 'r', encoding='utf8', errors='ignore') as doc_file:
            content = doc_file.read()            
            content = content.split('\n\n', maxsplit=1)[1]
        return content, label


class News20Dataset(Dataset):
    """A wrapper class for 20 Newsgroup dataset."""


    def __init__(self, data_path):
        super(News20Dataset, self).__init__(data_path)
        self._file_util = News20FileUtil()

    
    def get_dataset(self):
        """Returns 20 Newsgroup dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        
        files_list = []        
        for directory in os.scandir(self._data_path):
            files_list.extend(self._file_util.get_files(directory.path))

        train_test_ratio = 0.5
        breakpoint = int(train_test_ratio * len(files_list))
        
        train_set_paths = files_list[:breakpoint]
        test_set_paths = files_list[breakpoint:]

        assert train_set_paths and test_set_paths
        
        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_set_paths)
        test_set = self._build_dataframe(test_set_paths) 

        return train_set, test_set            

