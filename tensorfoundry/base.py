class BaseEstimator:
    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict()")