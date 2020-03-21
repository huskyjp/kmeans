import os
import math
from utils import converged, plot_2d, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt


# problem for students
def euclidean_distance_between_data(dp1, dp2):
    """Calculate the Euclidean distance between two data points.

    Arguments:
        dp1: a list of floats representing a data point
        dp2: a list of floats representing a data point

    Returns: the Euclidean distance between two data points
    """
    distance = 0
    for i in range(len(dp1)):
        distance += (dp1[i] - dp2[i])**2
    distance = math.sqrt(distance)
    return distance


def assign_data_to_closest_centroid(data_point, centroids):
    """Assign a single data point to the closest centroid. You should use
    euclidean_distance_between_data function (that you previously implemented).

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a string as the key name of the closest centroid to the data point
    """

    # for initial closest distance
    val_list = list(centroids.values())
    curr_closest_distance = euclidean_distance_between_data(data_point,
                                                            val_list[0])

    # for loop to find minimum distance
    # return that minimum value's key
    for element_key, element_value_list in centroids.items():
        temp_closest_distance = euclidean_distance_between_data(
                                data_point, element_value_list)
    # if this is minimum, return that list's key!
        if temp_closest_distance <= curr_closest_distance:
            closest_distance_key = element_key
            curr_closest_distance = temp_closest_distance
        else:
            curr_closest_distance = curr_closest_distance

    return closest_distance_key


# problem for students
def update_assignment(data, centroids):
    """Assign all data points to the closest centroids. You should use
    assign_data_to_closest_centroid fucntion (that you previously
    implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a new dictionary whose keys are the centroids' key names and
             values are lists of points that belong to the centroid
    """

    new_dict = {}
    min_string_list = []

    for i in range(0, len(data)):
        curr_list = data[i]
    # get key that has minimum distance between them as String
    # Assign that String to new_dict
        closest_centroids_key_string = assign_data_to_closest_centroid(
            curr_list, centroids)
        min_string_list.append(closest_centroids_key_string)
    # assining this string list as Key and its corresponding Value
    # but it is not able to assigning by using .append with empty dictonary so
    # first step is ADD key into dict by if it is not exsists
        if closest_centroids_key_string not in new_dict:
            new_dict[closest_centroids_key_string] = []
        new_dict[closest_centroids_key_string].append(curr_list)
    return new_dict


# problem for students
def mean_of_points(data):
    """Calculate the mean of a given group of data points. You should NOT hard
    -code the dimensionality of the data points).

    Arguments:
        data: a list of lists representing a group of data points

    Returns: a list of floats as the mean of the given data points
    """

    mean_list = []
    for i in range(0, len(data[0])):
        mean = 0
        for j in range(0, len(data)):
            add_point = ((data[j][i]) / len(data))
            mean = add_point + mean
        mean_list.append(mean)
    return mean_list


# problem for students
def update_centroids(assignment_dict):
    """Update centroid locations as the mean of all data points that belong
    to the cluster. You should use mean_of_points function (that you previously
    implemented).

    Arguments:
        assignment_dict: the dictionary returned by update_assignment function

    Returns: A new dictionary representing the updated centroids
    """

    new_dict = {}

    for element_key, element_value_list in assignment_dict.items():
        new_val = mean_of_points(element_value_list)
        for i in range(0, len(new_val)):
            if element_key not in new_dict:
                new_dict[element_key] = []
            new_dict[element_key].append(new_val[i])
    return new_dict


def main_2d(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        # plot centroid
        fig = plot_2d(assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D", f"step{step}.png"))
        plt.clf()
        step += 1
    print(f"K-means converged after {step} steps.")
    return centroids


def main_mnist(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    # plot initial centroids
    plot_centroids(centroids, "init")
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        step += 1
    print(f"K-means converged after {step} steps.")
    # plot final centroids
    plot_centroids(centroids, "final")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/mnist.csv")
    init_c = load_centroids("data/mnist_init_centroids.csv")
    final_c = main_mnist(data, init_c)
    write_centroids_tofile("mnist_final_centroids.csv", final_c)
