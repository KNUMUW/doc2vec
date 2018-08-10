import os
import pandas as pd
import multiprocessing as mp 
from data_loader.dataset.dataset import Dataset
from data_loader.file_util import FileUtil


class IMDBFileUtil(FileUtil):
    """Utility class for accessing files in IMDB movie reviews dataset."""    


    @staticmethod
    def load_single_file(path):
        """Loads a single file from IMDB dataset.
        
        Args:
            path (str): path to the file to be loaded.       
        Returns:
            content (str): content of the document.
            label (int): label of the document.
 
        """

        name = os.path.basename(path)
        
        number, rating = name.split('_')
        label = int(rating[:-4])
               
        with open(path, 'r') as doc_file:
            content = doc_file.read()

        return content, label 


class IMDBDataset(Dataset):
    """A wrapper class for IMDB movie reviews dataset."""


    def __init__(self, data_path):
        super(IMDBDataset, self).__init__(data_path)
        self._file_util = IMDBFileUtil()

        
    def get_dataset(self):
        """Returns IMDB movie reviews dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        
        # Get lists of file paths for training and test sets, respectively.
        files_lists = []
        for dir_1 in ['train', 'test']:
            for dir_2 in ['pos', 'neg']:
                dir_path = os.path.join(self._data_path, dir_1, dir_2) 
                files_lists.append(self._file_util.get_files(dir_path))

        train_set_paths = files_lists[0] + files_lists[1]
        test_set_paths = files_lists[2] + files_lists[3] 
                 
        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_set_paths)
        test_set = self._build_dataframe(test_set_paths) 

        return train_set, test_set            

