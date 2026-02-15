from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error

class TernaryHyperparameterOptimizer:
    """
    A custom optimization class that searches for the optimal hyperparameters of a 
    machine learning model using a combination of coordinate descent and ternary search. 
    It isolates and optimizes one hyperparameter at a time based on a predefined order 
    of importance.
    """

    def __init__(self, model, X, y, hyperparameters_ranges_dict: dict[str, tuple[int]], hyperparameter_importance_order:list[str], split = 0.2, error_function:callable=mean_absolute_error):
        """
        Initializes the optimizer with the model, dataset, search space, and evaluation criteria.

        Args:
            model (class): The uninstantiated machine learning model class. The class must 
                accept hyperparameters as keyword arguments and support a random_state parameter.
            X (array-like or DataFrame): The feature dataset.
            y (array-like or Series): The target variable.
            hyperparameters_ranges_dict (dict[str, tuple]): A dictionary where keys are the 
                hyperparameter names and values are tuples defining the search boundaries (low, high). 
            hyperparameter_importance_order (list[str]): A list of hyperparameter names defining 
                the exact sequence in which they should be optimized.
            split (float, optional): The proportion of the dataset to include in the validation 
                split if cross-validation is not used. Defaults to 0.2.
            error_function (callable, optional): The metric used to evaluate model performance. 
                Must follow the (y_true, y_pred) signature. Defaults to mean_absolute_error.
        """
        
        self.model = model

        self.X = X
        self.y = y
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=split)
        
        self.ranges = hyperparameters_ranges_dict
        self.importance_order = hyperparameter_importance_order

        self.error = error_function
        self.scorer = make_scorer(error_function, greater_is_better=False)

    def optimize(self, steps=5, cross_validate=False, cross_validation_folds=3):
        """
        Executes the ternary search optimization loop across all specified hyperparameters.

        Args:
            steps (int, optional): The number of ternary search iterations to perform 
                per hyperparameter. Defaults to 5.
            cross_validate (bool, optional): If True, evaluates the error using cross-validation 
                on the full datasets. If False, uses the standard train/validation split. 
                Defaults to False.
            cross_validation_folds (int, optional): The number of folds to use if 
                cross_validate is True. Defaults to 3.

        Returns:
            dict: A dictionary containing the final optimized hyperparameter configuration.
        """

        hyperparameters = {hp: self.ranges[hp][0] for hp in self.importance_order}

        for hp in self.importance_order:
            low, high = self.ranges[hp][0], self.ranges[hp][1]
            min_hp = low
            min_error = 0
            first = True

            for _ in range(steps):
                mid1 = low + (high - low) / 3
                mid2 = high - (high - low) / 3

                if isinstance(low, int):
                    mid1 = int(mid1)
                    mid2 = int(mid2)

                hyperparameters[hp] = mid1
                error1 = self.get_error(hyperparameters, cross_validate, cross_validation_folds)
                
                hyperparameters[hp] = mid2
                error2 = self.get_error(hyperparameters, cross_validate, cross_validation_folds)

                if error1 <= error2:
                    high = mid2
                    error = error1
                    hp_val = mid1
                elif error1 >= error2:
                    low = mid1
                    error = error2
                    hp_val = mid2

                if first or error < min_error:
                    min_error = error
                    min_hp = hp_val
                    first = False

                if mid1 == mid2: break

            hyperparameters[hp] = min_hp

        return hyperparameters
    
    def get_error(self, hyperparameters, cv, folds):
        """
        Fits the model with the current hyperparameter configuration and returns the calculated error.

        Args:
            hyperparameters (dict): The current state of the hyperparameters being tested.
            cv (bool): Flag indicating whether to use cross-validation.
            folds (int): The number of cross-validation folds.

        Returns:
            float: The calculated error score for the given hyperparameter configuration. 
                Lower values indicate better performance.
        """

        model = self.model(**hyperparameters, random_state=0)

        if cv:
            errors = -1 * cross_val_score(model, self.X, self.y, cv=folds, scoring=self.scorer)
            error = errors.mean()
        else:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_valid)
            error = self.error(self.y_valid, y_pred)

        return error