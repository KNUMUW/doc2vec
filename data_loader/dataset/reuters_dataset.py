import os
import pandas as pd
import multiprocessing as mp
from bs4 import BeautifulSoup as BS
from data_loader.dataset.dataset import Dataset
from data_loader.util import get_files_from_dir


class FileUtil:
    """The idea so far is to have one producer process that would read the whole file and push <REUTERS> blocks to an mp.Queue 
    (the point is: producer must be fast; gotta check the time to yield all the blocks e.g. with re package)
    We would also have consumer processes that would consume blocks from queue - filter them (are topics and contents valid) and
    then push content with label to a single (thread-safe) result list. Finally we construct a dataframe from that list.
    """
    pass


class ReutersFileUtil(FileUtil):
    """Utility class for accessing files in Reuters-21578 dataset."""    


    @staticmethod
    def load_single_file(path):
        """Loads a single file from Reuters dataset.
        
        Args:
            path (str): path to the file to be loaded.    
        Returns:
            content (list(str)): contents of documents (there are multiple ones in a single file).
            label (list(str)): labels of documents.
 
        """
        
        # re.findall(r'<REUTERS.*?<\/REUTERS>', data, re.DOTALL) - gets you list of REUTERS blocks
 
        with open(path, 'r', encoding='utf8', errors='ignore') as doc_file:
            file_content = doc_file.read()
       
            # That's necessary for BeautifulSoup to parse the file.
            file_content = file_content.replace('<BODY>', '<CONTENT>').replace('</BODY>', '</CONTENT>')
             
        soup = BS(file_content, 'lxml')
        
        articles = soup.find_all('reuters')
        
        # Find all articles that have non empty content and some topic.
        valid_articles = []    
        
        for art in articles:
            if art.content and art.content.text and art.topics.text:
                valid_articles.append(art)

        # TODO: we would like to parallelize dataframe building for this dataset.
        content = [art.content.text for art in valid_articles]
        label = [art.topics.text for art in valid_articles]

        return content, label  


class ReutersDataset(Dataset):
    """A wrapper class for Reuters-21578 dataset."""


    def __init__(self, data_path):
        super(ReutersDataset, self).__init__(data_path) 


    def _get_file_paths(self):
        """Returns paths to all available data files."""   
        file_paths = get_files_from_dir(self._data_path)
        file_paths = [path for path in file_paths if os.path.basename(path).startswith('reut')]                  
        
        return file_paths


    def get_dataset(self):
        """Returns Reuters dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                        
        file_paths = self._get_file_paths() 

        # TODO: employ producer-consumer pattern. 
       
        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_set_paths)
        test_set = self._build_dataframe(test_set_paths) 

        return train_set, test_set            
 
