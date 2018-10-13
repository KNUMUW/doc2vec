import importlib
import logging
import collections
import numpy as np
import sklearn.metrics
class MetricEvaluator:
    def __init__(self, settings): 
        self.metricSettings = settings["metric"]
        self.metric = getattr(sklearn.metrics, self.metricSettings["name"])
    def _evaluate(self,params, y_pred):
        logging.info('Evaluating '+ y_pred.name)
        metric = self.metric(**{'y_pred':y_pred.predicted,**params})
        return y_pred.name,metric
    def evaluate(self,test_set, y_preds):
        logging.info('Using '+ self.metricSettings["name"] + ' metric')
        y_true = test_set.label
        params= {'y_true':y_true,**self.metricSettings["params"]}
        return [self._evaluate(params,y_pred) for y_pred in y_preds]
