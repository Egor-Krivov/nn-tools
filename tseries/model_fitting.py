import numpy as np
from abc import abstractmethod, ABCMeta


class BaseModelFitter(metaclass=ABCMeta):
    """Base model fitter class for ValController."""
    @abstractmethod
    def __init__(self):
        """Instantiates model for fitting."""
        pass

    @abstractmethod
    def fit(self):
        """Performs one fit and returns epoch's record."""
        pass

    @property
    @abstractmethod
    def make_model_builder(self):
        """Returns method that cam build current model copy."""


class BestValController:
    def __init__(self, *, patience=None, max_iter=10, tol=0.01,
                 key=lambda x: x.val_loss, verbose=0):
        self.patience = patience
        self.max_iter = max_iter
        self.key = key
        self.tol = tol
        self.verbose = verbose

        self.best_model_builder = None
        self.best_score = np.inf
        self.history = None

    def fit_model(self, model_fitter: BaseModelFitter):
        """Performs complete model fitting"""
        self.best_model = None
        self.best_score = np.inf
        self.history = []

        best_epoch = 0
        for epoch in range(self.max_iter):
            record = model_fitter.fit()
            self.history.append(record)

            val = self.key(record)
            if val < self.best_score*(1-self.tol):
                best_epoch = epoch
                self.best_score = val
                self.best_model_builder = model_fitter.make_model_builder()
                if self.verbose:
                    print('new best score {}'.format(val))

            # Early stopping
            if self.patience and epoch - best_epoch > self.patience:
                break
             
    def get_best_model(self):
        return self.best_model_builder()
