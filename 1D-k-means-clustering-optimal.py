import numpy as np
import matplotlib.pyplot as plt
import random

TOTAL_ITERATIONS: int = 10
MAX_CENTRES: int = 8
COLOURS_LIST: list[str] = ["red", "yellow", "green", "blue", "orange", "purple", "cyan", "brown"]


def graph_data(input_data: list[float], centres: list[float], colours: list[str], number_of_centres: int, number_of_inputs: int) -> None:
    plt.scatter(x = input_data, y = np.zeros(number_of_inputs), c = colours)
    plt.scatter(x = centres, y = np.divide(np.ones(number_of_centres), 20), c = COLOURS_LIST[:number_of_centres], marker = "v")
    plt.ylim([-0.05, 1])
    plt.savefig(f"clustering_{number_of_centres}.jpg")
    plt.clf()

def get_assignments(input_data: list[float], centres: list[float]) -> np.array:
    assignments: list[int] = []
    for current_input in input_data:
        best_distance: float = None
        best_centre: int = None
        for current_centre in centres:
            distance: float = abs(current_input - current_centre)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_centre = list(centres).index(current_centre)
        assignments.append(best_centre)
    return np.array(assignments)

def get_colours(assignments: np.array, number_of_inputs: int) -> list[str]:
    colours: list[str] = []
    for current_input in range(number_of_inputs):
        if assignments[current_input] == 0:
            colours.append("firebrick")
        elif assignments[current_input] == 1:
            colours.append("khaki")
        elif assignments[current_input] == 2:
            colours.append("forestgreen")
        elif assignments[current_input] == 3:
            colours.append("cornflowerblue")
        elif assignments[current_input] == 4:
            colours.append("moccasin")
        elif assignments[current_input] == 5:
            colours.append("violet")
        elif assignments[current_input] == 6:
            colours.append("paleturquoise")
        else:
            colours.append("chocolate")
    return colours

def update_centres(input_data: list[float], assignments: np.array) -> np.array:
    new_centres: list[int] = []
    number_of_centres: int = max(assignments) + 1
    for current_centre_index in range(number_of_centres):
        inputs_in_cluster: list[float] = [input_data[index] for index in np.nditer(np.asarray(np.equal(assignments, current_centre_index)).nonzero())]
        cluster_min, cluster_max = min(inputs_in_cluster), max(inputs_in_cluster)
        centre: float = (cluster_min + cluster_max) / 2
        new_centres.append(centre)
    return np.array(new_centres)

def get_total_variance(input_data: list[float], assignments: np.array, centres: np.array) -> float:
    number_of_clusters: int = max(assignments) + 1
    total_variance: float = 0.0
    for current_cluster in range(number_of_clusters):
        # Get all inputs where they are assigned to the current cluster
        inputs_in_cluster: list[float] = [input_data[index] for index in np.nditer(
                                            np.asarray(np.equal(assignments, current_cluster)).nonzero())]
        
        variance: float = 0.0
        number_of_inputs_in_cluster: int = len(inputs_in_cluster)
        for current_input_from_cluster in inputs_in_cluster:
            variance += abs(current_input_from_cluster - centres[current_cluster])
        total_variance += variance / number_of_inputs_in_cluster
    return total_variance

def graph_variance_for_optimatal_centres_by_number(optimal_centres_variances: list[int]) -> None:
    plt.plot(np.arange(1, MAX_CENTRES + 1), optimal_centres_variances)
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of cluster spread")
    plt.ylim(bottom = 0)
    plt.savefig("optimal_variance_by_centre_num.jpg")

def main():
    # List of data points
    input_data = [-10, -9.25, -9, -6.25, -6, -5.75, -5.5, -5, -2, -1, 0, 0.5, 0.75, 1, 1.5, 6, 7, 7.5, 8, 9]
    random.shuffle(input_data)
    number_of_inputs = len(input_data)

    optimal_centres_variances: list[float] = []

    for number_of_centres in range(1, MAX_CENTRES + 1):
        centres_matrix: np.ndarray = np.empty((TOTAL_ITERATIONS, number_of_centres))
        total_variances = np.empty((TOTAL_ITERATIONS,))
        for iteration_index in range(TOTAL_ITERATIONS):
            # Initialize random centres from the data points given
            centres = np.sort(random.sample(input_data, number_of_centres))

            # Assign the points to clusters
            assign = get_assignments(input_data, centres)

            current_assignments = None
            new_assignments = assign
            while current_assignments is None or not np.all(current_assignments == new_assignments):
                current_assignments = new_assignments
                centres = update_centres(input_data, current_assignments)
                new_assignments = get_assignments(input_data, centres)

            total_variance = get_total_variance(input_data, new_assignments, centres)

            centres_matrix[iteration_index] = centres
            total_variances[iteration_index] = total_variance

        optimal_variance_index: int = np.argmin(total_variances)
        optimal_variance: float = min(total_variances)
        optimal_centres: list[float] = centres_matrix[optimal_variance_index]
        optimal_assignments: np.array = get_assignments(input_data, optimal_centres)
        optimal_colours: list[str] = get_colours(optimal_assignments, number_of_inputs)
        graph_data(input_data, optimal_centres, optimal_colours, number_of_centres, number_of_inputs)
        optimal_centres_variances.append(optimal_variance)

    graph_variance_for_optimatal_centres_by_number(optimal_centres_variances)

if __name__ == "__main__":
    main()
