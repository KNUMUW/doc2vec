import os
from data_loader.dataset.dataset import Dataset
from data_loader.util import get_files_from_dir


class News20Dataset(Dataset):
    """A wrapper class for 20 Newsgroup dataset."""

    def __init__(self, data_path):
        super().__init__(data_path)

    def _get_file_paths(self, train_test_ratio=0.5):
        """Returns paths to files that make up training and test set, respectively.
        
        Args:
            train_test_ratio (float): split ratio for available data files
        Returns:
            train_set_paths (list(str)): paths to training data files.
            test_set_paths (list(str)): paths to test data files.

        """                
        files_list = []
        
        for directory in os.scandir(self._data_path):
            files_list.extend(get_files_from_dir(directory.path))

        breakpoint = int(train_test_ratio * len(files_list))
        
        train_set_paths = files_list[:breakpoint]
        test_set_paths = files_list[breakpoint:]

        return train_set_paths, test_set_paths
 
    @staticmethod
    def _load_single_file(path):
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
        
    @classmethod
    def _build_dataframe(cls, file_paths):
        """Builds appropriate dataframe from all given data files.

        Args:
            file_paths (list(str)): paths to all data files we want to wrap as a dataframe.
        Returns:
            dataframe (pandas.DataFrame): the resulting dataframe.            

        """
        return super()._build_dataframe(file_paths)

    def get_dataset(self):
        """Returns 20 Newsgroup dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        return super().get_dataset()
 
