from utils.feature_utils import *
import numpy as np
from stqdm import stqdm


def bsoid_predict_numba(feats, clf):
    """
    :param scaler:
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        # scaled_feats = scaler.transform(feats[i])
        labels = clf.predict(np.nan_to_num(feats[i]))
        labels_fslow.append(labels)
    return labels_fslow


def frameshift_predict(data_test, num_test, rf_model, framerate=30):
    labels_fs = []
    new_predictions = []
    for i in stqdm(range(num_test), desc="Predicting behaviors from files"):
        feats_new = bsoid_extract_numba([data_test[i]], framerate)
        labels = bsoid_predict_numba(feats_new, rf_model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(framerate / 10)):
            labels_fs2.append(labels_fs[k][l])
        new_predictions.append(np.array(labels_fs2).flatten('F'))
    new_predictions_pad = []
    for i in range(0, len(new_predictions)):
        new_predictions_pad.append(np.pad(new_predictions[i], (len(data_test[i]) -
                                                               len(new_predictions[i]), 0), 'edge'))
    return np.hstack(new_predictions_pad)