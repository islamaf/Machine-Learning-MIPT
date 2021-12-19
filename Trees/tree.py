from os import XATTR_REPLACE
import numpy as np
from sklearn.base import BaseEstimator
import warnings


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      y_mean = np.mean(y, axis=0)
      entropy_result = -1*np.sum(y_mean * np.log(y_mean + EPS))

    return entropy_result
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      y_mean = np.mean(y, axis=0)
      gini_impurity = 1 - np.sum(y_mean ** 2)

    return gini_impurity
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    y_variance = np.var(y)

    return y_variance

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    y_mad_median = np.mean(np.abs(y - np.median(y)))

    return y_mad_median


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=None):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        slice_X_subset = X_subset[:, feature_index]
        X_left = X_subset[slice_X_subset < threshold]
        X_right = X_subset[slice_X_subset >= threshold]

        y_left = y_subset[slice_X_subset < threshold]
        y_right = y_subset[slice_X_subset >= threshold]

        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold
        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        slice_X_subset = X_subset[:, feature_index]
        y_left = y_subset[slice_X_subset < threshold]
        y_right = y_subset[slice_X_subset >= threshold]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        """
        # YOUR CODE HERE
        feature_index, threshold = 0, 0
        best_value = np.inf
        number_of_objects, number_of_features = X_subset.shape

        for feature_idx in range(number_of_features):
          for threshold_idx in set(X_subset[:, feature_idx]):
            y_left, y_right = self.make_split_only_y(feature_idx, threshold_idx, X_subset, y_subset)
            n = len(y_left) / len(X_subset) * self.criterion(y_left) + len(y_right) / len(X_subset) * self.criterion(y_right)

            if n < best_value:
              best_value = n
              feature_index = feature_idx
              threshold = threshold_idx

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if self.max_depth == 0 or all([np.all(y == y_subset[0]) for y in y_subset]):
          if self.classification:
            proba = np.sum(y_subset, axis=0)
            proba /= np.sum(proba)
          else:
            proba = np.mean(y_subset)
          return Node(None, None, proba)

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)

        self.max_depth -= 1
        new_node.left_child = self.make_tree(X_left, y_left)
        new_node.right_child = self.make_tree(X_right, y_right)
        self.max_depth += 1

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on
        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        y_predicted = []
        for x in X:
          current_node = self.root
          while current_node.proba is None:
            if x[current_node.feature_index] < current_node.value:
              current_node = current_node.left_child
            else:
              current_node = current_node.right_child
          
          if self.classification:
            y_predicted.append(np.argmax(current_node.proba))
          else:
            y_predicted.append(current_node.proba)

        y_predicted = np.array(y_predicted).reshape((-1, 1))
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        for i in range(X.shape[0]):
          current_node = self.root
          while current_node.proba is None:
            if X[i][current_node.feature_index] < current_node.value:
              current_node = current_node.left_child
            else:
              current_node = current_node.right_child
          y_predicted_probs[i] = current_node.proba

        return np.array(y_predicted_probs)
