# Copyright 2020 resspect software
# Author: The RESSPECT team
#         Initial skeleton taken from ActSNClass
#
# created on 02 March 2020
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


__all__ = ['uncertainty_sampling', 'random_sampling', 'percentile_sampling']


def uncertainty_sampling(class_prob: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty in predicted class.
    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.
    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """

    if class_prob.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # calculate distance to the decision boundary - only binary classification
    dist = abs(class_prob[:, 1] - 0.5)

    # get indexes in increasing order
    order = dist.argsort()

    # only allow objects in the query sample to be chosen
    flag = []
    for item in order:
        if test_ids[item] in queryable_ids:
            flag.append(True)
        else:
            flag.append(False)

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(class_prob[order[0]], '-- > ', class_prob[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])
    
def random_sampling(test_ids: np.array, queryable_ids: np.array,
                    batch=1, queryable=False, query_thre=1.0, seed=42) -> list:
    """Randomly choose an object from the test sample.
    Parameters
    ----------
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    queryable: bool (optional)
        If True, check if randomly chosen object is queryable.
        Default is False.
    query_thre: float (optinal)
        Threshold where a query is considered worth it.
        Default is 1.0 (no limit).
    seed: int (optional)
        Seed for random number generator. Default is 42.
    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried. If there are less queryable objects than the
            required batch it will return only the available objects
            -- so the list of objects to query can be smaller than 'batch'.
    """

    # randomly select indexes to be queried
    np.random.seed(seed)
    indx = np.random.choice(np.arange(0, len(test_ids)), 
                            size=len(test_ids),
                            replace=False)

    if queryable:
        # flag only the queryable objects
        flag = []
        for item in indx:
            if test_ids[item] in queryable_ids:
                flag.append(True)
            else:
                flag.append(False)

        flag = np.array(flag)

        # check if there are queryable objects within threshold
        indx_query = int(len(flag) * query_thre)

        if sum(flag[:indx_query]) > 0:
            # return the corresponding batch size
            return list(indx[flag])[:batch]
        else:
            # return empty list
            return list([])

    else:
        return list(indx)[:batch]
    
def percentile_sampling(class_prob: np.array, test_ids: np.array,
                  queryable_ids: np.array, perc: float) -> list:
    """Search for the sample at a specific percentile of uncertainty.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    perc: float in [0,1]
        Percentile used to identify obj to be queried.
  
    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
    """

    # calculate distance to the decision boundary - only binary classification
    dist = abs(class_prob[:, 1] - 0.5)

    # get indexes in increasing order
    order = dist.argsort()

    # get index of wished percentile
    perc_index = int(order.shape[0] * perc)

    # return the index of the highest object at the requested percentile
    return list([perc_index])


def main():
    return None


if __name__ == '__main__':
    main()
