import os 
import json
import pandas as pd
from data_loader.dataset import IMDBDataset, LingspamDataset, News20Dataset, ReutersDataset


class DataLoader:
    """A composition class for datasets that allows to load them into memory.

    Attributes:
        _data_path (str): path to directory containing all datasets.
        _datasets (dict): map between dataset names and data_loader.dataset.Dataset objects.        

    """

    # Set default data directory path.
    _package_absolute_path = os.path.abspath(os.path.dirname(__file__))
    _default_data_dir = os.path.join(_package_absolute_path, '../data/')

    def __init__(self, data_path=_default_data_dir): 
        self._data_path = data_path
        self._datasets = {}
        
        # IDEA: datasets could be initialized via a configuration file instead of having these values hardcoded.
        imdb_dataset_path = os.path.join(data_path, 'imdb_reviews', 'aclImdb')
        lingspam_dataset_path = os.path.join(data_path, 'lingspam_public', 'bare')        
        news20_dataset_path = os.path.join(data_path, 'news20', '20_newsgroup')
        reuters_dataset_path = os.path.join(data_path, 'reuters21578')

        self._datasets['imdb_reviews'] = IMDBDataset(imdb_dataset_path)
        self._datasets['lingspam_public'] = LingspamDataset(lingspam_dataset_path)
        self._datasets['news20'] = News20Dataset(news20_dataset_path) 
        self._datasets['reuters21578'] = ReutersDataset(reuters_dataset_path)        
   
    def load_dataset(self, settings):
        """Returns  given dataset split between training and test set.
        
        Args:
            settings (dict): experiment description. 
        Returns:
            train_set (pd.DataFrame): dataframe containing individual labeled documents from dataset 
            test_set (pd.DataFrame): as above
        
        Example:
            >>> from data_loader import DataLoader
            >>> loader = DataLoader()
            >>> with open('experiment_1.txt', 'r') as settings:
            ...     settings = json.load(settings)
            ...     train_set, test_set = loader.load_dataset(settings)        

        """       
        dataset_name = settings['dataset']

        if dataset_name in self._datasets.keys():
            return self._datasets[dataset_name].get_dataset()
        else:
            dataset_names = self._datasets.keys()
            dataset_names = ['\'' + name + '\'' for name in dataset_names]
            err_msg = 'Dataset \'{}\' could not be found. Available datasets:\n'.format(dataset_name) 
            err_msg += '\n'.join(dataset_names)
            raise NameError(err_msg)

