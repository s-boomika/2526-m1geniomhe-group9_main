from ..base import BaseEstimator

class UnsupervisedModel(BaseEstimator):
    """
    base class for all unsupervised learning models.
    """
    def __init__(self):
        super().__init__()
