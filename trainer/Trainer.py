import importlib
import logging
import collections
import numpy as np
Model = collections.namedtuple('Model', 'name trainer')
Trained = collections.namedtuple('Trained', 'name model')
Predicted = collections.namedtuple('Predicted', 'name predicted')

class Trainer:
    def __init__(self, settings): 
        modelsSettings = settings["models"]
        self.models = []
        for model in modelsSettings:
            self.models.append(Model(model["name"],self._buildTrainer(model)))
    def _buildTrainer(self,model):
        module = importlib.import_module(model["module"])
        trainer = getattr(module, model["type"])
        my_instance = trainer(**model["params"])
        return my_instance
    def _train(self,model,X,y):
        logging.info('Training '+ model.name)
        return model.trainer.fit(X,y)
    def fit(self,vector,set):
        X = np.vstack(vector.values)
        y = set.label
        self.traineds = [Trained(model.name,self._train(model,X,y)) for model in self.models]
    def predict(self, vector):
        X = np.vstack(vector.values)
        return [Predicted(trained.name,trained.model.predict(X)) for trained in self.traineds]
     