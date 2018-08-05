import os
import pandas as pd
import multiprocessing as mp
from data_loader.dataset.dataset import Dataset
from data_loader.file_util import FileUtil


class LingFileUtil(FileUtil):
    """Utility class for accessing files in Ling-Spam dataset."""    
    

    @staticmethod
    def load_single_file(path):
        """Loads a single file from Ling-Spam dataset.
        
        The file is added to docs_list as an appropriate tuple (described below).

        Args:
            path (str): path to the file to be loaded.       
        Returns:
            content (str): content of the document.
            label (int): label of the document.
 
        """
    
        name = os.path.basename(path)
        
        if 'spmsg' in name:
            label = 1
        else:
            label = 0
 
        with open(path, 'r') as doc_file:
            content = doc_file.read()
        
        return content, label


class LingspamDataset(Dataset):
    """A wrapper class for Ling-Spam dataset."""
    

    def __init__(self, data_path):
        super(LingspamDataset, self).__init__(data_path)
        self._file_util = LingFileUtil()

    
    def get_dataset(self):
        """Returns Ling-Spam dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                

        # Get lists of file paths for training and test sets, respectively.
        files_list = []
        for i in range(1, 11):
            dir_path = os.path.join(self._data_path, 'part%d' % i)            
            files_list.extend(self._file_util.get_files(dir_path))

        train_test_ratio = 0.5
        breakpoint = int(train_test_ratio * len(files_list))
        
        train_set_paths = files_list[:breakpoint]
        test_set_paths = files_list[breakpoint:]
        
        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_set_paths)
        test_set = self._build_dataframe(test_set_paths) 
        
        return train_set, test_set            
 
