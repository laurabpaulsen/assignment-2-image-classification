"""
Holds the ImageClassifier class, which is used to classify images for this assignment.

The class is initialized with the model type (logistic or mlp) and the parameters for the model. 
Any parameters allowed for the scikit learn model can be specified as keyword arguments when initializing the class. 
The parameters are then updated with the ones specified in the class attributes if they exist.

Author: Laura Bock Paulsen (202005791)
"""

import os
import numpy as np

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# metrics
from sklearn.metrics import classification_report



class ImageClassifier:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        # check if model is valid
        if self.model_type not in ["logistic", "mlp"]:
            raise ValueError("Only 'logistic' and 'mlp' are implemented for now.")

        # add attributes to the class based on the keyword arguments (if they exist)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.clf = self.get_classifier()

    def update_params(self, params):
        """
        Updates the parameters with ones specified in the class attributes if they exist.

        Parameters
        ----------
        params : dict
            Dictionary with default parameters for the model.

        Returns
        -------
        None.
        """
        for key, value in params.items(): # loop over the default parameters
            if hasattr(self, key): # check if the attribute exists in the class
                try: 
                    params[key] = getattr(self, key)
                except AttributeError: # if the key does not exist, do nothing and continue
                    pass

    def get_classifier(self):
        if self.model_type == "logistic":
            # default parameters for logistic regression
            params = {"penalty": "none",
                        "tol": 0.1,
                        "verbose": True,
                        "solver": "saga",
                        "multi_class": "multinomial",
                        "C": 1.0,
                        "fit_intercept": True,
                        "random_state": 42,
                        "max_iter": 1000,
                        "warm_start": False,
                        "n_jobs": None,
                        "l1_ratio": None,
                        "intercept_scaling": 1,
                        "class_weight": None,
                        "dual": False}
        else:
            # default parameters for MLP
            params = {"random_state": 42,
                      "hidden_layer_sizes": (64, 10),
                      "learning_rate":"adaptive",
                      "early_stopping": True,
                      "verbose": True,
                      "activation": "relu",
                      "solver": "adam",
                      "batch_size": "auto",
                      "learning_rate_init": 0.001,
                      "learning_rate": "constant",
                      "power_t": 0.5,
                      "max_iter": 200,
                      "shuffle": True,
                      "tol": 0.0001,
                      "early_stopping": False,
                      "warm_start": False,
                      "momentum": 0.9,
                      "nesterovs_momentum": True,
                      "validation_fraction": 0.1,
                      "beta_1": 0.9,
                      "beta_2": 0.999,
                      "epsilon": 1e-08,
                      "n_iter_no_change": 10,
                      "max_fun": 15000
                      }
        # update the parameters with the ones specified in the class attributes if they exist
        self.update_params(params) 
        
        if self.model_type == "logistic":
            return LogisticRegression(**params)
        
        else:
            return MLPClassifier(**params)
        
    
    def train(self, X_train, y_train):
        """
        Trains the classifier on the training data.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.

        Returns
        -------
        None.
        """
        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Predicts the labels for the test data.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test data.

        Returns
        -------
        y_pred : numpy.ndarray 
            Predicted labels for the test data.
        """
        return self.clf.predict(X_test)
    

    def evaluate(self, y_test, y_pred, **kwargs):
        """
        Returns the classification report

        Parameters
        ----------
        y_test : numpy.ndarray
            True labels for the test data.
        y_pred : numpy.ndarray
            Predicted labels for the test data.
        **kwargs : dict
            Keyword arguments to be passed to the classification_report function.
        
        Returns
        -------
        report : str
            Classification report.
        """
        metrics = classification_report(y_test, y_pred, **kwargs)
        params = self.clf.get_params()

        report = f"Classification report for {self.model_type} classifier with parameters set to {params}:\n\n {metrics}"

        return report