import numpy as np

def logsoftmax(x: np.ndarray, axis: int = -1):
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return (x - x_max) - np.log(s)

def get_top_k_logits(scores, top_k):
    """
    perform top-k sampling

    Parameters:
      scores - model output logits
      top_k - number of elements with highest probability to select
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def create_sinusoidal_positions(num_pos: int, dim: int, base: float):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos, dtype=np.float32), inv_freq).astype("float32")
    sinusoid_inp = np.concatenate((sinusoid_inp, sinusoid_inp), axis=-1)
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

