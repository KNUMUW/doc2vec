import os
import pandas as pd


class FileUtil:
    """Utility base class for accessing files in some dataset (specified via inheritance)."""    


    @staticmethod
    def get_files(dir_path):
        """Returns all data file paths from given directory.

        Args:
            dir_path (str): the directory we want files from. 
        Returns:
            file_paths (list(str)): paths to data files in given directory.   

        """ 
        
        file_paths = []

        for entry in os.scandir(dir_path):
            if entry.is_file():
                file_paths.append(entry.path)

        return file_paths


    @staticmethod
    def load_single_file(path):
        """Loads a single file from a dataset.
        
        This method is implemented differently depending on the dataset it is intended to use on. 
        The file is returned as an appropriate tuple (described below).

        Args:
            path (str): path to the file to be loaded.
        Returns:
            content (list(str)/str): content(s) of document(s) (depending on whether single file contains just one document or not).
            label (list(str)/str): label(s) of document(s).
 
        """

        raise NotImplementedError('method load_single_file must be implemented')

