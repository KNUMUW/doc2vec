import os
import re
import pandas as pd
import multiprocessing as mp
from lxml import etree
from operator import itemgetter
from data_loader.dataset.dataset import Dataset
from data_loader.util import get_files_from_dir

# TODO: IDEA: actually BlockProducer and ResultReceiver functionality could be located directly in the main thread.

class BlockProducer(mp.Process):
    """A producer class for single <REUTERS> blocks.
   
    BlockProducer reads input files and pushes subsequent <REUTERS> blocks to an output queue.
 
    Attributes:
        _num_consumer (int): how many BlockConsumer objects are spawned.
        _file_paths (list(str)): paths to data files to process.
        _task_queue (mp.Queue): the output queue containing <REUTERS> blocks.
     
    """

    def __init__(self, num_consumers, file_paths, task_queue):
        super().__init__()
        self._num_consumer = num_consumers
        self._file_paths = file_paths
        self._task_queue = task_queue

    def run(self):
        # Indices are necessary for train/test split to be consistent from run to run.
        idx = 0
        for path in self._file_paths:
            # Get <REUTERS> blocks from given file.
            with open(path, 'r', encoding='utf8', errors='ignore') as doc_file:
                file_content = doc_file.read()
            blocks = re.findall(r'<REUTERS.*?<\/REUTERS>', file_content, re.DOTALL)
            
            # Push the blocks to output queue.
            for block in blocks:
                self._task_queue.put((idx, block))
                idx += 1

        # Send poison pills to BlockConsumers.
        for _ in range(self._num_consumer):
            self._task_queue.put((None, None))
        

class BlockConsumer(mp.Process):
    """A consumer class for single <REUTERS> blocks.

    BlockConsumer gets <REUTERS> blocks from an input queue, parses them and pushes results - 
    - (document, labels) pairs, if they were found within a block - to an output queue for 
    further processing.
    
    Attributes:
        _task_queue (mp.Queue): the input queue containing <REUTERS> blocks.
        _result_queue (mp.Queue): the output queue containing (document, labels) pairs. 

    """

    def __init__(self, task_queue, result_queue):
        super().__init__()
        self._task_queue = task_queue
        self._result_queue = result_queue

    def run(self):
        while True:
            # Get a message.
            idx, task = self._task_queue.get()
            # Check if poison pill.
            if task is None:
                break
            
            # Parse the block for content and labels.
            document, labels = self._process_data(task)      
            
            # If the block contains valid data - push it to results.
            if document and all(labels):
                self._result_queue.put((idx, (document, labels)))
        
        # Send poison pill to the ResultConsumer.
        self._result_queue.put((None, None))

    def _process_data(self, task):
        """Returns document contents and labels (if found) from a <REUTERS> block.

        Args:
            task (str): a <REUTERS> block from data file.
        Returns:
            document (str): contents of a <BODY> block (if found, else '').
            labels (tuple(str)): contents of <TOPICS> block (if found, else ('',)).
        
        """
        parser = etree.HTMLParser()

        # Get the document contents.
        doc = re.search(r'<BODY>.*<\/BODY>', task, re.DOTALL) 
        if doc:
            doc = etree.fromstring(doc.group(0), parser)
            doc = doc.xpath('//body/text()')[0]
        else:
            doc = ''

        # Get the topics (labels).
        labels = re.search(r'<TOPICS>.*<\/TOPICS>', task, re.DOTALL)
        if labels:
            labels = etree.fromstring(labels.group(0), parser)
            labels = labels.xpath('//d/text()')
            if labels:
                labels = tuple(labels)
            else:
                labels = ('', )             
        else:
            labels = ('', )

        return doc, labels


class ResultConsumer(mp.Process):
    """A consumer class for (document, labels) pairs.
    
    ResultConsumer gets (document, labels) pairs from an input queue and appends them to a 
    results list. 
     
    Attributes:
        _num_consumer (int): how many BlockConsumer objects are spawned.
        _result_queue (mp.Queue): the input queue containing (document, labels) pairs. 
        _results_list (list): the list of (document, labels) pairs to be returned 
        _conn (mp.connection.Connection): one end of a pipe to send results to main thread.

    """

    def __init__(self, num_consumer, result_queue, conn):
        super().__init__()
        self._num_consumer = num_consumer
        self._result_queue = result_queue
        self._results_list = []
        self._conn = conn

    def run(self):
        poison_count = 0
        while poison_count < self._num_consumer:
            # Get the message.
            idx, result = self._result_queue.get()        
            # Check if poison pill.
            if result is None:
                poison_count += 1
                continue

            # Add the result to the results list.
            self._results_list.append((idx, result))

        # Restore the order and send the results back to the main thread.
        self._results_list.sort(key=itemgetter(0))
        self._results_list = [item[1] for item in self._results_list]
        self._conn.send(self._results_list)
        self._conn.close()


class ReutersDataset(Dataset):
    """A wrapper class for Reuters-21578 dataset."""

    def __init__(self, data_path):
        super().__init__(data_path) 

    def _get_file_paths(self):
        """Returns paths to all available data files."""   
        file_paths = get_files_from_dir(self._data_path)
        file_paths = [path for path in file_paths if os.path.basename(path).startswith('reut')]                  
        
        return file_paths

    def _get_results(self, file_paths):
        """Returns all found documents with corresponding labels.

        Args:
            file_paths (list(str)): paths to data files.
        Returns:
            results (list(tuple(str, tuple(str))): pairs (document, document labels) obtained 
                from given data.

        """        
        # Initialize communication objects.
        parent_conn, child_conn = mp.Pipe()
        tasks = mp.Queue()
        results = mp.Queue()
           
        # Initialize and start all workers.
        num_consumers = mp.cpu_count() 
        block_producer = BlockProducer(num_consumers, file_paths, tasks)
        block_consumers = [BlockConsumer(tasks, results) for _ in range(num_consumers)]
        result_consumer = ResultConsumer(num_consumers, results, child_conn)

        result_consumer.start()
        for consumer in block_consumers:
            consumer.start()
        block_producer.start()

        # Wait for initial workers to finish.
        block_producer.join()
        for consumer in block_consumers:
            consumer.join()
        
        # Get the results from ResultsConsumer.
        results = parent_conn.recv()
        
        # Join the final worker.
        result_consumer.join()

        return results

    @staticmethod
    def _split(data, train_test_ratio=0.5):
        """Returns data list split in two in given ratio."""
        
        breakpoint = int(train_test_ratio * len(data))        
        train_data = data[:breakpoint]
        test_data = data[breakpoint:]

        return train_data, test_data
 
    @staticmethod
    def _build_dataframe(results):
        """Builds appropriate dataframe from given (document, labels) pairs.

        Args:
            results (list): (document, labels) pairs obtained from data files.
        Returns:
            dataframe (pandas.DataFrame): the resulting dataframe.            

        """
 
        data_dict = {}
        data_dict['document'] = [res[0] for res in results]         
        data_dict['label'] = [res[1] for res in results]       

        return pd.DataFrame.from_dict(data_dict)

    def get_dataset(self):
        """Returns Reuters dataset.
       
        Returns:
            train_set (pandas.DataFrame): training set dataframe.
            test_set (pandas.DataFrame): test set dataframe.

        """                        
        # Get list of all data file paths. 
        file_paths = self._get_file_paths() 

        # Get list of all documents with corresponding labels.
        results = self._get_results(file_paths)  

        # Split the results between training and test sets.
        train_results, test_results = self._split(results)

        # Build appropriate dataframes. 
        train_set = self._build_dataframe(train_results)
        test_set = self._build_dataframe(test_results) 

        return train_set, test_set            
 
