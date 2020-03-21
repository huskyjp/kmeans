from analysis import update_assignment, majority_count, accuracy
from kmeans_tests import assert_dict_eq


# helper functions
def setup():
    data = [
            [-1.01714716,  0.95954521,  1.20493919,  0.34804443],
            [-1.36639346, -0.38664658, -1.02232584, -1.05902604],
            [1.13659605, -2.47109085, -0.83996912, -0.24579457],
            [-1.48090019, -1.47491857, -0.6221167,  1.79055006],
            [-0.31237952,  0.73762417,  0.39042814, -1.1308523],
            [-0.83095884, -1.73002213, -0.01361636, -0.32652741],
            [-0.78645408,  1.98342914,  0.31944446, -0.41656898],
            [-1.06190687,  0.34481172, -0.70359847, -0.27828666],
            [-2.01157677,  2.93965872,  0.32334723, -0.1659333],
            [-0.56669023, -0.06943413,  1.46053764,  0.01723844]
        ]
    labels = [0, 1, 0, 2, 1, 2, 1, 2, 0, 0]
    random_centroids = {
            "centroid1": [0.1839742, -0.45809263, -1.91311585, -1.48341843],
            "centroid2": [-0.71767545, 1.2309971, -1.00348728, -0.38204247],
            "centroid3": [-1.71767545, 0.29971, 0.00328728, -0.38204247],
        }
    bad_centroids = {
            "centroid1": [0.1839742, -0.45809263, -1.91311585, -1.48341843],
            "centroid2": [10, 10, 10, 10],
            "centroid3": [-10, 1, -10, 10],
        }
    return data, labels, random_centroids, bad_centroids


# test begins
def test_update_assignment():
    # set up
    data, labels, random_centroids, bad_centroids = setup()

    # random
    answer = {'centroid3': [0, 1, 2, 1, 2, 2, 0], 'centroid1': [0],
              'centroid2': [1, 0]}
    assert_dict_eq(update_assignment(data, labels, random_centroids), answer)

    # bad
    answer = {'centroid1': [0, 1, 0, 2, 1, 2, 1, 2, 0, 0]}
    assert_dict_eq(update_assignment(data, labels, bad_centroids), answer)
    print("test_update_assignment passed")


def test_majority_count():
    # single
    assert majority_count([0, 0, 0, 0, 0, 0]) == 6

    # mixed
    assert majority_count([0, 0, 1, 1, 0, 0]) == 4
    assert majority_count([0, 2, 2, 2, 3, 3, 0, 1, 1, 0, 0]) == 4

    # tied max count
    assert majority_count([0, 2, 2, 2, 0, 2, 0, 0]) == 4
    print("test_majority_count passed")


def test_accuracy():
    # set up
    data, labels, random_centroids, bad_centroids = setup()

    # random
    answer = 0.5
    assert abs(accuracy(data, labels, random_centroids) - answer) < 1e-5

    # bad
    answer = 0.4
    assert abs(accuracy(data, labels, bad_centroids) - answer) < 1e-5
    print("test_accuracy passed")


if __name__ == '__main__':
    test_update_assignment()
    test_majority_count()
    test_accuracy()
    print("all tests passed.")
