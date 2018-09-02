import multiprocessing as mp
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class TaggedDocumentIterator:
    """An iterable object that yields TaggedDocument instances for gensim doc2vec model training.

    Attributes:
        _documents (list(list(str))): a list of tokenized documents.
        _labels (list): a list of document labels (commonly just subsequent integers).
        size (int): number of stored documents. 
    """

    def __init__(self, documents, labels):
        self._documents = documents
        self._labels = labels
        self.size = len(documents)

    def __iter__(self):
        for  doc, label in zip(self._documents, self._labels):
            yield TaggedDocument(words=doc, tags=[label])


class Doc2VecWrapper:
    """A class that allows to obtain vector representations for given documents."""

    @staticmethod
    def _get_representations(doc_iterator, vector_length):
        """Trains doc2vec model on given data and returns vector representations.

        Args:
            doc_iterator (TaggedDocumentIterator): yields TaggedDocument instances for training.
            vector_length (int): length of returned vector representations.
        Returns:
            representations (pandas.Series): NumPy arrays of vector representations for input documents.

        """
        # Define models.
        cores = mp.cpu_count()
        dbow_model = Doc2Vec(dm=0, vector_size=vector_length, negative=5, hs=0, min_count=2, sample=0, 
            epochs=20, workers=cores)
        dm_model = Doc2Vec(dm=1, vector_size=vector_length, window=10, negative=5, hs=0, min_count=2, 
            sample=0, epochs=20, workers=cores, alpha=0.05)
    
        # Build the vocabulary.
        dbow_model.build_vocab(doc_iterator)
        dm_model.build_vocab(doc_iterator)

        # Train models.
        dbow_model.train(doc_iterator, total_examples=dbow_model.corpus_count, epochs=20)
        dm_model.train(doc_iterator, total_examples=dm_model.corpus_count, epochs=20)
        
        # Get concatenated vector representations.
        representations = []
        for i in range(doc_iterator.size): 
            dbow_vector = dbow_model.docvecs[i]
            dm_vector = dm_model.docvecs[i]
            representations.append(np.concatenate((dbow_vector, dm_vector)))

        return pd.Series(representations)

    @classmethod
    def doc2vec_features(cls, prep_dataset, settings):
        """Returns doc2vec vector representations for given documents.
 
        Args:
            prep_dataset (pandas.Series(list(str))): documents after preprocessing. They must already be tokenized.
            settings (dict): experiment description.
        Returns:
            representations (pandas.Series): NumPy arrays of vector representations for input documents.

        """
        # Some preliminaries.
        labels = list(prep_dataset.index)
        vector_length = settings['vector_length']
        doc_iterator = TaggedDocumentIterator(prep_dataset, labels)
        
        # Get vector representations.
        representations = cls._get_representations(doc_iterator, vector_length)

        return representations

