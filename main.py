import time
import json
import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.metrics import accuracy_score
from typing import List

# === helper functions below ===

def get_candidate_radiuses(distances, epsilon=0.01):
    '''
    param distances: distance matrix
    param epsilon: search resolution. epsilon=0.01 means about 100 candidates
    return: candidate radiuses chosen from sorted unique distances
    '''
    candidate_radiuses = np.unique(distances)                        # sort and remove duplicates
    candidate_radiuses = candidate_radiuses[candidate_radiuses > 0]  # keep only positive values (without zeros)

    n = len(candidate_radiuses)
    num_candidates = int(1 / epsilon)

    # ---------- small amount of candidates cases ----------
    # edge case: no candidates
    if n == 0:
        return candidate_radiuses

    # if the number of candidates is small enough, test all of them
    if n <= num_candidates:
        return candidate_radiuses

    # ---------- large amount of candidates case ----------
    # remove extreme radiuses: below the 5th percentile and above the 95th percentile
    if n > num_candidates:
        low_index = int(0.05 * n)
        high_index = int(0.95 * n)
        candidate_radiuses = candidate_radiuses[low_index:high_index]
        n = len(candidate_radiuses)  # update n after removing extreme values

    # choose evenly spaced representative radiuses
    indexes = np.linspace(0, n - 1, num_candidates, dtype=int)
    candidate_radiuses = candidate_radiuses[indexes]

    return candidate_radiuses

def get_predictions_for_a_given_radius(radius, distances, y_train):
    '''
    param radius: selected radius
    param distances: distance matrix where rows are validation/test instances and columns are train instances
    param y_train: train labels
    return: predictions
    '''
    predictions = []

    for x_index in range(len(distances)):
        # boolean mask: True for train instances inside the radius
        neighbors_mask = distances[x_index] <= radius
        neighbor_labels = y_train[neighbors_mask]

        # majority vote among neighbors
        if len(neighbor_labels) > 0:
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            most_freq_label = labels[np.argmax(counts)]

        # edge case: no neighbors inside the radius
        else:
            nearest_index = np.argmin(distances[x_index])
            most_freq_label = y_train[nearest_index]

        predictions.append(most_freq_label)

    return predictions


def get_distance_matrix(X_train, X_val):
    """
    param X_train
    param X_val
    return: distance matrix
    """
    distances_rows = []

    for x in X_val:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        distances_rows.append(distances)

    return np.array(distances_rows)


def normalize_X_array(X_array, mean, std):
    '''
    param: X_array
    return: normalized_X_array
    '''
    normalized_X_array = (X_array - mean) / std
    return normalized_X_array

# === helper functions above ===


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:

    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # read train and validation csv files
    df_train = pd.read_csv(data_trn)
    df_val = pd.read_csv(data_vld)

    # X:
    X_train = df_train.drop(['class'], axis=1)
    X_val = df_val.drop(['class'], axis=1)
    X_tst = df_tst  # according to main

    # y:
    y_train = df_train['class']
    y_val = df_val['class']

    # use as numPy arrays:
    X_train, X_val, X_tst = X_train.to_numpy(dtype=float), X_val.to_numpy(dtype=float), X_tst.to_numpy(dtype=float)
    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()

    # normalize using helper function:
    # calc mean and std by X_train:
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train , axis=0)
    std[std == 0] = 1  # fallback for edge case, can not divide by zero

    # normalize:
    X_train = normalize_X_array(X_train, mean, std)
    X_val   = normalize_X_array(X_val  , mean, std)
    X_tst   = normalize_X_array(X_tst  , mean, std)

    # distance matrix between validation and train
    distances = get_distance_matrix(X_train, X_val)

    # -----------------------------------------------------------------------------------------------------
    # finding the best radius:
    # -----------------------------------------------------------------------------------------------------

    # choose candidate radiuses using a helper function:
    candidate_radiuses = get_candidate_radiuses(distances, 0.02)

    # edge case when there are no radiuses
    if len(candidate_radiuses) == 0:
        best_radius = 0

    else:
        best_radius, best_score = None, -1

        for curr_radius in candidate_radiuses:
            # getting val predictions using a helper function:
            val_predictions = get_predictions_for_a_given_radius(curr_radius, distances, y_train)

            # check the score:
            curr_score = accuracy_score(y_val, val_predictions)

            # update best score and radius:
            if best_score < curr_score:
                best_radius, best_score = curr_radius, curr_score

    # -----------------------------------------------------------------------------------------------------
    # using the same NNR logic to predict labels for df_tst:
    # -----------------------------------------------------------------------------------------------------

    tst_distances = get_distance_matrix(X_train, X_tst)
    tst_predictions = get_predictions_for_a_given_radius(best_radius, tst_distances, y_train)

    return list(tst_predictions)


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty predictions, should not happen
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'classification accuracy on the test set: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
