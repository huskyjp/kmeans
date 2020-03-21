from kmeans import assign_data_to_closest_centroid
from utils import load_centroids, read_data


def update_assignment(data, labels, centroids):
    """Assign all data points to the closest centroids and keep track of their
    labels. The i-th point in "data" corresponds to the i-th label in "label".

    Arguments:
        data: a list of lists representing all data points
        labels: a list of ints representing all data labels
        ex.) labels = [0, 1, 0, 2, 1, 2, 1, 2, 0, 0]
        centroids: the centroid dictionary
        ex.) "centroid1": [0.1839742, -0.45809263, -1.91311585, -1.48341843],

    Returns: a new dictionary whose keys are the centroids' key names and
             values are a list of labels of the data points that are assigned
             to that centroid.
    """

    new_dict = {}
    min_string_list = []

    for i in range(0, len(data)):
        curr_list = data[i]
        curr_index_label = labels[i]

        # get key that has minimum distance between them as String
        # Assign that String to new_dict
        closest_centroids_key_string = assign_data_to_closest_centroid(
            curr_list, centroids)
        min_string_list.append(closest_centroids_key_string)

        # assining this string list as Key and
        # its corresponding labels as Value
        # data[1] is correspond to labels[1] so
        # current <i> is current int for both

        if closest_centroids_key_string not in new_dict:
            new_dict[closest_centroids_key_string] = []

        new_dict[closest_centroids_key_string].append(curr_index_label)
    return new_dict


def majority_count(labels):
    """Return the count of the majority labels in the label list

    Arguments:
        labels: a list of labels

    Returns: the count of the majority labels in the list
    """
    # user mylist.count(x) to get a number of specific character
    # update when it is MAX

    majority_label_count = 0
    count = 0

    for i in range(0, len(labels)):
        curr_label = labels[i]
        temp_majority_label = labels.count(curr_label)

        if (temp_majority_label >= majority_label_count):
            count += 1
            majority_label_count = temp_majority_label
        else:
            majority_label_count = majority_label_count

    return majority_label_count


def accuracy(data, labels, centroids):
    """Calculate the accuracy of the algorithm. You should use
    update_assignment and majority_count (that you previously implemented)

    Arguments:
        data: a list of lists representing all data points
        labels: a list of ints representing all data labels
        centroids: the centroid dictionary

    Returns: a float representing the accuracy of the algorithm
    """

    new_dict = update_assignment(data, labels, centroids)
    max_appeard = 0
    elements = 0
    total_accuracy = 0.0

    # value is list of label
    for element_value_list in new_dict.values():
        max_appeard += majority_count(element_value_list)
        elements += len(element_value_list)

    total_accuracy = (float)((max_appeard) / (elements))
    return total_accuracy


if __name__ == '__main__':
    centroids = load_centroids("mnist_final_centroids.csv")
    data, label = read_data("data/mnist.csv")
    print(accuracy(data, label, centroids))


# LEAVE YOUR ANSWERS HERE...
# 1. What happened to the centroids? Why are there fewer than 10?
# The centroids converted ino new visual and now it looks like actual
# digits rather previous
# gaussian noised appearance, stored in MNIST/init file.
# The centroid4 did not show up since it wasn't assigned to
# any data point when we run
# def update_assignment(data, labels, centroids):. Because this function
# returns a new
# dictionary whose keys are the centroids' key names and values are a list
# of labels which are related to
# each data points, if the centroid is not assigned to any of data
# points as closest centroids at that moment,
# we are not able to convert it at all, i.e. no process will be done by python.

# 2. What's the accuracy of the algorithm on MNIST? By looking at the
# centroids, which digits are easier to be distinguished by the algorithm,
# and which are harder?
# ...
# Since we have accuray result as 0.582, in other words, our algorithm is
# 58.2% correct and 41.8% contains an error,
# we have only liitle bit above 1/2 probability success rate of getting
# the corret result.
# Centroid easier to understand: Centroid8, Centroid6,
#                                Centroid5, Centroid2,
#                                Centroid1
# Centroid harder to understand: Centroid9, Centroid7,
#                                Centroid3, Centroid0
# This results is understandable since our algorithm has
# 55-60% (58.2%) range success rate,
# hale of the images should be readable and the others should be not.
# The centroids that classified into harder (to understand),
# it contains error data so irregular color was applied,
# then as a result, it became harder to understand.
