import numpy as np
import torchvision.ops


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious
    ious = torchvision.ops.box_iou(atlbrs, btlbrs)
    return ious


def crop_player_photo(image, bboxes, index):
    x1, y1, x2, y2, _ = bboxes[index]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    return image[int(y1):int(y2), int(x1):int(x2)]


def load_player_profiles(player_profiles_file):
    player_profiles = {}
    import pickle
    with open(player_profiles_file, "rb") as f:
        player_profiles = pickle.load(f)
    return player_profiles
