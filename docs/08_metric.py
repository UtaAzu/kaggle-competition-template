import numpy as np
import numba
import json

def _rle_encode_jit(mask):
    mask = mask.T.flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

@numba.njit
def _rle_decode_jit(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    idx = 0
    for i in range(0, len(rle), 2):
        start = rle[i] - 1
        length = rle[i + 1]
        mask[start:start + length] = 1
    return mask.reshape((shape[1], shape[0])).T

def rle_encode(mask):
    return json.dumps(_rle_encode_jit(mask))

def rle_decode(rle_str, shape):
    rle = json.loads(rle_str)
    return _rle_decode_jit(np.array(rle, dtype=np.int32), shape)

def oF1_score(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)

def score(submission, ground_truth, shape):
    f1_scores = []
    for case_id in ground_truth:
        gt_mask = rle_decode(ground_truth[case_id], shape)
        pred_mask = rle_decode(submission.get(case_id, "[]"), shape)
        f1_scores.append(oF1_score(gt_mask, pred_mask))
    return np.mean(f1_scores)
