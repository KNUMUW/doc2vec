import os
from data_loader.dataset.dataset import Dataset
from data_loader.util import get_files_from_dir


class IMDBDataset(Dataset):
    """A wrapper class for IMDB movie reviews dataset."""

    def __init__(self, data_path):
        super().__init__(data_path)

    def _get_file_paths(self):
        """Returns paths to files that make up training and test set, respectively.

        Returns:
            train_set_paths (list(str)): paths to training data files.
            test_set_paths (list(str)): paths to test data files.

        """        
        files_list = []

        for dir_up in ['train', 'test']:
            for dir_down in ['pos', 'neg']:
                dir_path = os.path.join(self._data_path, dir_up, dir_down) 
                files_list.append(get_files_from_dir(dir_path))

        train_set_paths = files_list[0] + files_list[1]
        test_set_paths = files_list[2] + files_list[3] 
        
        return train_set_paths, test_set_paths

    @staticmethod
    def _load_single_file(path):
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
               
        with open(path, 'r', encoding='utf8', errors='ignore') as doc_file:
            content = doc_file.read()

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
        """Returns IMDB movie reviews dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                
        return super().get_dataset()
 
