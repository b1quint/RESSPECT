# Copyright 2020 resspect software
# Author: The RESSPECT team
#
# created on 14 April 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

__all__ = ['sklearn_classifiers', 'bootstrap_clf']


def bootstrap_clf(clf_function, n_ensembles, train_features,
                  train_labels, test_features, **kwargs):
    """
    Train an ensemble of classifiers using bootstrap.

    Parameters
    ----------
    clf_function: function
        function to train classifier
    n_ensembles: int
        number of classifiers in the ensemble
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Prediction of the ensemble
    class_prob: np.array
        Average distribution of ensemble members
    ensemble_probs: np.array
        Probability output of each member of the ensemble
    """
    n_labels = np.unique(train_labels).size
    num_test_data = test_features.shape[0]
    ensemble_probs = np.zeros((num_test_data, n_ensembles, n_labels))

    for i in range(n_ensembles):
        x_train, y_train = resample(train_features, train_labels)
        predicted_class, class_prob = clf_function(x_train,
                                                   y_train,
                                                   test_features,
                                                   **kwargs)
        ensemble_probs[:, i, :] = class_prob

    class_prob = ensemble_probs.mean(axis=1)
    predictions = np.argmax(class_prob, axis=1)

    return predictions, class_prob, ensemble_probs


def sklearn_classifiers(classifier: str,
                        train_features:  np.array, 
                        train_labels: np.array,
                        test_features: np.array, **kwargs):
    """Random Forest classifier.

    Parameters
    ----------
    classifier: str
        Classifier. Options are: 'RandomForest', 
        'GradientBoostedTrees', 'KNearestNeighbor', 
        'MultiLayerPerceptron', 'SupportVectorMachine'
        or 'NaiveBayes'.
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Predicted classes for test sample.
    prob: np.array
        Classification probability for test sample [pIa, pnon-Ia].
    clf: classifier object
        The trained classifier.
    """

    # create classifier instance
    if classifier == 'RandomForest':
        clf = RandomForestClassifier(**kwargs)
    elif classifier == 'GradientBoostedTrees':
        clf = XGBClassifier(**kwargs)
    elif classifier == 'KNearestNeighbor':
        clf = KNeighborsClassifier(**kwargs)
    elif classifier == 'MultiLayerPerceptron':
        clf = MLPClassifier(**kwargs)
    elif classifier == 'SupportVectorMachine':
        clf = SVC(probability=True, **kwargs)
    elif classifier == 'NaiveBayes':
        clf=GaussianNB(**kwargs)

    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob, clf


def main():
    return None


if __name__ == '__main__':
    main()
