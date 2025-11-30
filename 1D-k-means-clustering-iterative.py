import numpy as np
import matplotlib.pyplot as plt
import random

COLOURS_LIST:list[str] = ["red", "yellow", "green", "blue", "orange", "purple", "cyan", "brown"]

def graph_data(input_data: list[float], centres: list[float], colours: list[str], number_of_centres: int, number_of_inputs: int) -> None:
    plt.scatter(x = input_data, y = np.zeros(number_of_inputs), c = colours)
    plt.scatter(x = centres, y = np.divide(np.ones(number_of_centres), 20),
                c = COLOURS_LIST[:number_of_centres], marker = "v")
    plt.ylim([-0.05, 1])
    plt.show()

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
    assignments = np.array(assignments)
    return assignments

def get_colours(assign: np.array, number_of_inputs: int) -> list[str]:
    colours: list[str] = []
    for current_input in range(number_of_inputs):
        if assign[current_input] == 0:
            colours.append("firebrick")
        elif assign[current_input] == 1:
            colours.append("khaki")
        elif assign[current_input] == 2:
            colours.append("forestgreen")
        elif assign[current_input] == 3:
            colours.append("cornflowerblue")
        elif assign[current_input] == 4:
            colours.append("moccasin")
        elif assign[current_input] == 5:
            colours.append("violet")
        elif assign[current_input] == 6:
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
        centre = (cluster_min + cluster_max) / 2
        new_centres.append(centre)
    return np.array(new_centres)


def main():
    # List of data points
    input_data: list[float] = [-10, -9.25, -9, -6.25, -6, -5.75, -5.5, -5, -2, -1, 0, 0.5, 0.75, 1, 1.5, 6, 7, 7.5, 8, 9]
    random.shuffle(input_data)
    number_of_inputs: int = len(input_data)

    number_of_centres: int = 4

    # Initialize random centres from the data points given
    centres: list[float] = np.sort(random.sample(input_data, number_of_centres))
    # Assigns the points to clusters
    assignments: list[int] = get_assignments(input_data, centres)
    # Colour the points based on their cluster
    colours: list[str] = get_colours(assignments, number_of_inputs)

    current_assignments: list[int] = None
    new_assignments: list[int] = assignments

    while current_assignments is None or not np.all(current_assignments == new_assignments):
        current_assignments = new_assignments
        centres = update_centres(input_data, current_assignments)
        new_assignment = get_assignments(input_data, centres)
        colours = get_colours(new_assignment, number_of_inputs)
        graph_data(input_data, centres, colours, number_of_centres, number_of_inputs)

if __name__ == "__main__":
    main()
