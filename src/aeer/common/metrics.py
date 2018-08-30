'''
Implementation of the metrics
'''
import numpy as np


def precision_at_k(predictions, actuals, k):
    """
    Computes the precision at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns precision: float, the precision score at k
    """
    N = len(actuals)
    hits = len(set(predictions[-k:]).intersection(set(actuals)))
    precision = hits / min(N, k)
    return precision


def recall_at_k(predictions, actuals, k):
    """
    Computes the recall at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns recall: float, the recall score at k
    """
    N = len(actuals)
    hits = len(set(predictions[-k:]).intersection(set(actuals)))
    recall = hits / N
    return recall


def map_at_k(predictions, actuals, k):
    """
    Computes the MAP at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns MAP: float, the score at k
    """
    avg_prec = []
    for i in range(1, k + 1):
        prec = precision_at_k(predictions, actuals, i)
        avg_prec.append(prec)
    return np.mean(avg_prec)


def ndcg_at_k(predictions, actuals, k):
    """
    Computes the NDCG at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns NDCG: float, the score at k
    """
    N = min(len(actuals), k)
    cum_gain = 0
    ideal_gain = 0
    topk = predictions[-N:]
    hits = 0
    # calculate the ideal gain at k
    for i in range(0, N):
        if topk[i] in actuals:
            cum_gain += 1 / np.log2(i + 2)
            hits = hits + 1

    for i in range(0, hits):
        ideal_gain += 1 / np.log2(i + 2)
    if ideal_gain != 0:
        ndcg = cum_gain / ideal_gain
    else:
        ndcg = 0
    return ndcg


if __name__ == '__main__':
    pass
