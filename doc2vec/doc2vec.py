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
        for doc, label in zip(self._documents, self._labels):
            yield TaggedDocument(words=doc, tags=[label])


class Doc2VecWrapper:
    """A class that allows to obtain vector representations for given documents."""

    def train(self, prep_dataset, settings):
        """Trains doc2vec model on given data.

        Args:
            prep_dataset (pandas.Series(list(str))): documents after preprocessing. They must already be tokenized.
            settings (dict): experiment description.

        """
        # Preliminaries.
        vector_length = settings['vector_length']        
        labels = list(prep_dataset.index)
        doc_iterator = TaggedDocumentIterator(prep_dataset, labels)
 
        # Define models.
        cores = mp.cpu_count()
        self._dbow_model = Doc2Vec(dm=0, vector_size=vector_length, negative=5, hs=0, min_count=2, sample=0, 
            epochs=20, workers=cores)
        self._dm_model = Doc2Vec(dm=1, vector_size=vector_length, window=10, negative=5, hs=0, min_count=2, 
            sample=0, epochs=20, workers=cores, alpha=0.05)
    
        # Build the vocabulary.
        self._dbow_model.build_vocab(doc_iterator)
        self._dm_model.build_vocab(doc_iterator)

        # Train models.
        self._dbow_model.train(doc_iterator, total_examples=self._dbow_model.corpus_count, epochs=20)
        self._dm_model.train(doc_iterator, total_examples=self._dm_model.corpus_count, epochs=20)

        # Discard unnecessary parameters.
        self._dbow_model.delete_temporary_training_data(keep_doctags_vectors=False)
        self._dm_model.delete_temporary_training_data(keep_doctags_vectors=False)

    def doc2vec_features(self, prep_dataset):
        """Returns doc2vec vector representations for given documents.
 
        Args:
            prep_dataset (pandas.Series(list(str))): documents after preprocessing. They must already be tokenized.
        Returns:
            representations (pandas.Series): NumPy arrays of vector representations for input documents.

        """
        # Get vector representations from both models.
        # BTW inference parameters (alpha, number of epochs and so on) can be changed here.
        dbow_rep = prep_dataset.apply(self._dbow_model.infer_vector)
        dm_rep = prep_dataset.apply(self._dm_model.infer_vector)
    
        # Concatenate obtained representations.
        representations = dbow_rep.combine(dm_rep, lambda r1, r2: np.concatenate((r1, r2)))

        return representations

