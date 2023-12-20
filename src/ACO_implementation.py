# ---------------------- IMPORT LIBRARIES -------------------------

import numpy as np
import xml.etree.ElementTree as ET


# --------------------- PARSING OPERATIONS ------------------------


def parse_XML(filename):
    '''
    Parse an XML file containing input data for TSP
    args
        | filename - name of the file to parse
    return
        | tree - an ElementTree formed from the input data
    '''
    # Parse the XML file
    tree = ET.parse(filename)
    return tree


# ---------------------- DISTANCE MATRIX --------------------------

def build_distance_matrix(tree):
    '''
    Construct a distance matrix containing distance values between each pair of cities
    args
        | tree - an ElementTree containing (cost) data for each path
    return
        | distance - a 2D array (distance matrix) containing distance (costs) for each path in the graph
    '''
    # Get the root element
    root = tree.getroot()
    # Count number of vertices
    for child in root:
        if child.tag == "graph":
            for vertex in child:
                i = 0
                for edge in vertex:
                    i += 1
                break
            break
    vertices = i + 1
    # Construct distance matrix
    distances = []
    for child in root:
        if child.tag == "graph":
            for vertex in child:
                row = [0] * vertices
                for edge in vertex:
                    row[int(edge.text)] = float(edge.attrib["cost"])
                distances.append(row)
    return distances


# --------------------- ACO IMPLEMENTATIONS ---------------------



# Standard ACO algorithm for TSP
def ant_colony_optimization(input_data, num_ants, num_iterations, alpha, beta, rho, Q):
    '''
    Runs the standard version of ACO for TSP on a given dataset
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d distance heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                # Add cost of travelling to selected city to total distance
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of returning to start city
            distance += distance_matrix[visited[-1]][visited[0]]
            paths.append(visited)
            distances.append(distance)
            # Evaluate solution fitness (total cost/distance)
            if distance < best_distance:
                best_distance = distance
                best_path = visited.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Deposit pheromone with given Q parameter
        for i, path in enumerate(paths):
            for j in range(num_cities - 1):
                pheromone_matrix[path[j]][path[j + 1]] += (Q / distances[i])
    return best_path, best_distance


# Max-Min Ant System ACO algorithm for TSP
def MMAS_ant_colony_optimization(input_data, num_ants, num_iterations, alpha, beta, rho, Q, max_p, min_p):
    '''
    Runs the max-min ant system variant of ACO for TSP on a given dataset
    The amount of pheromone is limited between a maximum and minimum value
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
        | max_p - maximum limit to amount of pheromone
        | min_p - minimum limit to amount of pheromone
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse the input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize maximum and minimum pheromone limits with given parameters
    max_pheromone = max_p
    min_pheromone = min_p
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                # Add cost of travelling to selected city to total distance
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of returning to start city
            distance += distance_matrix[visited[-1]][visited[0]]
            paths.append(visited)
            distances.append(distance)
            # Evaluate solution fitness (total cost/distance)
            if distance < best_distance:
                best_distance = distance
                best_path = visited.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Deposit pheromone with given Q parameter
        for i, path in enumerate(paths):
            for j in range(num_cities - 1):
                pheromone_matrix[path[j]][path[j + 1]] += (Q / distances[i])
        # Apply maximum and minimum pheromone limits
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone_matrix[i][j] = max(min_pheromone, min(max_pheromone, pheromone_matrix[i][j]))
    return best_path, best_distance


# Elitist Ant System ACO algorithm for TSP
def elitist_ant_colony_optimization(input_data, num_ants, num_iterations, alpha, beta, rho, Q):
    '''
    Runs the elitist ant system variant of ACO for TSP on a given dataset
    Only the best performing ant is allowed to deposit pheromone
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse the input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                # Add cost of travelling to selected city to total distance
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of returning to start city
            distance += distance_matrix[visited[-1]][visited[0]]
            paths.append(visited)
            distances.append(distance)
            # Evaluate solution fitness (total cost/distance)
            if distance < best_distance:
                best_distance = distance
                best_path = visited.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Only allow best ant to deposit pheromone with given Q parameter
        best_ant_path = paths[np.argmin(distances)]
        for j in range(num_cities - 1):
            pheromone_matrix[best_ant_path[j]][best_ant_path[j + 1]] += Q / best_distance
    return best_path, best_distance


# Rank Based Ant System algorithm for TSP
def rank_based_ant_colony_optimization(input_data, num_ants, num_iterations, alpha, beta, rho, Q):
    '''
    Runs the rank-based ant system variant of ACO for TSP on a given dataset
    Ants are ranked based on path fitness and deposit pheromone depending on their rank
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse the input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                # Add cost of travelling to selected city to total distance
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of returning to start city
            distance += distance_matrix[visited[-1]][visited[0]]
            paths.append(visited)
            distances.append(distance)
            # Evaluate solution fitness (total cost/distance)
            if distance < best_distance:
                best_distance = distance
                best_path = visited.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Get indices of sorted paths by fitness to rank ants
        sorted_indices = np.argsort(distances)
        # Update pheromone trails based on ant ranks with given Q parameter
        for rank, ant_index in enumerate(sorted_indices):
            p = paths[ant_index]
            for i in range(num_cities - 1):
                pheromone_matrix[p[i]][p[i + 1]] += Q / (rank + 1)  # Update based on rank
    return best_path, best_distance



# Ant Colony Optimization algorithm for TSP with Tabu Search local search heuristic
def ant_colony_optimization_tabu_search(input_data, num_ants, num_iterations, alpha, beta, rho, Q, tabu_iterations, tabu_size):
    '''
    Runs the standard version of ACO for TSP on a given dataset, with tabu search as a local search heuristic
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
        | tabu_iterations - number of iterations for tabu search
        | tabu_size - size of tabu list
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d distance heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of travelling to selected city to total distance
            distance += distance_matrix[visited[-1]][visited[0]]
            # Apply Tabu Search local search heuristic to the ant's path
            best_p, best_d = tabu_search(visited, distance_matrix, tabu_iterations, tabu_size)
            paths.append(best_p)
            distances.append(best_d)
            # Evaluate solution fitness (total cost/distance)
            if best_d < best_distance:
                best_distance = best_d
                best_path = best_p.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Deposit pheromone with given Q parameter
        for i, path in enumerate(paths):
            for j in range(num_cities - 1):
                pheromone_matrix[path[j]][path[j + 1]] += (Q / distances[i])
    return best_path, best_distance



# Ant Colony Optimization algorithm for TSP with Hill Climbing local search heuristic
def ant_colony_optimization_hill_climbing(input_data, num_ants, num_iterations, alpha, beta, rho, Q):
    '''
    Runs the standard version of ACO for TSP on a given dataset, with hill climbing as a local search heuristic
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d distance heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of travelling to selected city to total distance
            distance += distance_matrix[visited[-1]][visited[0]]
            # Apply Hill Climbing local search heuristic to the ant's path
            best_p, best_d = hill_climbing(visited, distance_matrix)
            paths.append(best_p)
            distances.append(best_d)
            # Evaluate solution fitness (total cost/distance)
            if best_d < best_distance:
                best_distance = best_d
                best_path = best_p.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Deposit pheromone with given Q parameter
        for i, path in enumerate(paths):
            for j in range(num_cities - 1):
                pheromone_matrix[path[j]][path[j + 1]] += (Q / distances[i])
    return best_path, best_distance


# Ant Colony Optimization algorithm for TSP with 2-opt local search heuristic
def ant_colony_optimization_two_opt(input_data, num_ants, num_iterations, alpha, beta, rho, Q):
    '''
    Runs the standard version of ACO for TSP on a given dataset, with 2-opt as a local search heuristic
    args
        | input_data - filename of an XML file containing input data to form a graph
        | num_ants - number of generated ants
        | num_iterations - number of iterations of the algorithm
        | alpha - transition probability parameter 1
        | beta - transition probability parameter 2
        | rho - pheromone evaporation rate
        | Q - pheromone deposit factor
    return
        | best_path - best path obtained from running ACO using the given parameters
        | best_distance - shortest distance obtained from running ACO using the given parameters
    '''
    # Parse input file and build distance matrix
    tree = parse_XML(input_data)
    distance_matrix = build_distance_matrix(tree)
    num_cities = len(distance_matrix)
    # Initialize pheromone matrix
    pheromone_matrix = np.random.rand(num_cities, num_cities)
    np.fill_diagonal(pheromone_matrix, 0.0)
    # Initialize distance and path variables
    best_distance = float('inf')
    best_path = []
    # Run ACO for given number of iterations
    for x in range(num_iterations):
        paths = []
        distances = []
        # Generate given number of ants
        for ant in range(num_ants):
            # Start at arbitrary city
            current = 0
            visited = [current]
            distance = 0
            # For each unvisitied city, calculate ant transition probabilities
            for y in range(num_cities - 1):
                probs = np.zeros(num_cities)
                unvisited = list(set(range(num_cities)) - set(visited))
                for city in unvisited:
                    # Transition rule formula (with 1/d distance heuristic) with given alpha and beta parameters
                    probs[city] = (pheromone_matrix[current][city] ** alpha) * (1.0 / distance_matrix[current][city]) ** beta
                # Select the next city based on calculated transition probabilities
                selected = np.random.choice(range(num_cities), p=probs / np.sum(probs))
                visited.append(selected)
                distance += distance_matrix[current][selected]
                current = selected
            # Add cost of travelling to selected city to total distance
            distance += distance_matrix[visited[-1]][visited[0]]
            # Apply 2-opt local search heuristic to the ant's path
            best_p, best_d = two_opt(visited, distance_matrix)
            paths.append(best_p)
            distances.append(best_d)
            # Evaluate solution fitness (total cost/distance)
            if best_d < best_distance:
                best_distance = best_d
                best_path = best_p.copy()
        # Evaporate pheromone with given rho parameter
        pheromone_matrix *= (1 - rho)
        # Deposit pheromone with given Q parameter
        for i, path in enumerate(paths):
            for j in range(num_cities - 1):
                pheromone_matrix[path[j]][path[j + 1]] += (Q / distances[i])
    return best_path, best_distance




# ---------------------- LOCAL SEARCH HEURISTICS FOR ACO -----------------------

# Tabu Search local search heuristic
def tabu_search(path, distance_matrix, tabu_iterations=20, tabu_size=5):
    '''
    Runs the Tabu local search method
    Tabu Search attempts to improve a solution by exploring possible modifications to it
    Also attempts to avoid previously obtained solutions to escape local optima
    args
        | path - given ant path
        | distance matrix - 2D array containing travel distance information between each pair of cities
        | tabu_iterations - given number of iterations for tabu search (default = 20)
        | tabu_size - capacity of the tabu list (default = 5)
    return
        | path - modified path 
        | best distance - best distance of the new path
    '''
    num_cities = len(path)
    # Calculate path distance
    best_distance = sum(distance_matrix[path[i - 1]][path[i]] for i in range(num_cities))
    tabu_list = []
    # Run Tabu local search for given number of iterations
    for x in range(tabu_iterations):
        best_path = path.copy()
        # Generate new paths
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue  
                new_path = path.copy()
                # Swap two cities
                new_path[i], new_path[j] = new_path[j], new_path[i]
                # Calculate new path distance
                new_distance = sum(distance_matrix[new_path[k - 1]][new_path[k]] for k in range(num_cities))
                # Compare new path distance with best distance and check tabu conditions
                if new_distance < best_distance and [i, j] not in tabu_list:
                    best_path = new_path
                    best_distance = new_distance
                    tabu_list.append([i, j])
                    # Make sure tabu list does not exceed given size
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)
        path = best_path
    return path, best_distance


def hill_climbing(path, distance_matrix):
    '''
    Runs the Hill climbing local search method
    Similarly to Tabu Search, Hill Climbing attempts to improve a solution by exploring
    possible modifications to it (i.e., its neighbours) but without using a tabu list
    args
        | path - given ant path
        | distance matrix - 2D array containing travel distance information between each pair of cities
    return
        | path - modified path 
        | best distance - best distance of the new path
    '''
    num_cities = len(path)
    # Calculate path distance
    best_distance = sum(distance_matrix[path[i - 1]][path[i]] for i in range(num_cities))
    improved = True
    # Loop until the solution is improved
    while improved:
        improved = False
        # Generate new paths
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue 
                new_path = path.copy()
                # Swap two cities
                new_path[i], new_path[j] = new_path[j], new_path[i]
                # Calculate new path distance
                new_distance = sum(distance_matrix[new_path[k - 1]][new_path[k]] for k in range(num_cities))
                # Compare new path distance with best distance                
                if new_distance < best_distance:
                    path = new_path
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return path, best_distance


def two_opt(path, distance_matrix):
    '''
    Runs the 2-opt local search method
    2-opt also attempts to improve a solution by by iteratively swapping edges to find shorter paths
    args
        | path - given ant path
        | distance matrix - 2D array containing travel distance information between each pair of cities
    return
        | path - modified path 
        | best distance - best distance of the new path
    '''
    num_cities = len(path)
    # Calculate path distance
    best_distance = sum(distance_matrix[path[i - 1]][path[i]] for i in range(num_cities))
    improved = True
    # Loop until the solution is improved
    while improved:
        improved = False
        # Generate new paths
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_path = path.copy()
                # Swap edges
                new_path[i:j] = path[j - 1:i - 1:-1]
                # Calculate new path distance
                new_distance = sum(distance_matrix[new_path[k - 1]][new_path[k]] for k in range(num_cities))
                # Compare new path distance with best distance 
                if new_distance < best_distance:
                    path = new_path
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return path, best_distance



# --------------------- MAIN PROGRAM LOOP --------------------

accepted_variants = ['s', 'm', 'e', 'r']
welcome_str = "\nWelcome to this Ant Colony Optimisation (ACO) algorithm implementation!\n\
This program allows you to run several variants of ACO on test data using your own parameters.\n\n\
The available variants of ACO available are the following, with their associated code in parentheses.\n"
choice_str = "- Standard ACO (s)\n- Max-Min Ant System ACO (m)\n- Elitist Ant System ACO (e)\n- Rank-Based ACO (r)\n\n"
heuristic_str = "Valid local search heuristics are Tabu Search (t), Hill Climbing (h), 2-opt (2), or no heuristic (none)\n"
accepted_heuristics = ['t', 'h', '2', 'none']

def main():
    alive = True
    print(welcome_str)
    while alive:
        print(choice_str)
        code = input("Enter the letter code for the ACO variant you want to choose (enter q to exit)  ")
        if code == "q":
            alive = False
            break
        elif code not in accepted_variants:
            print("Invalid code. Please try again. \n")
            continue
        else:
            input_data = input("Choose a dataset (enter 'brazil58.xml' or 'burma14.xml')  ")
            num_ants = input("Enter the number of ants to generate  ")
            num_iterations = input("Enter the number of iterations  ")
            alpha = input("Enter the value for the alpha transition rule parameter  ")
            beta = input("Enter the value for the beta transition rule parameter  ")
            rho = input("Enter the value for the pheromone evaporation (rho) parameter  ")
            Q = input("Enter the value for the pheromone deposit factor (Q)  ")
            if code == "s":
                heuristic_valid = True
                print(heuristic_str)
                while heuristic_valid:
                    heuristic = input("Enter the letter code for the local search heuristic you want to choose (enter 'none' for no heuristic)  ")
                    if heuristic not in accepted_heuristics:
                        print("Invalid code. Please try again. \n")
                        continue
                    else:
                        break
                if heuristic == "t":
                    tabu_iterations = input("Choose the number of tabu iterations (WARNING - ACO with Tabu Search is very slow, recommend a low number of iterations)  ")
                    tabu_size = input("Enter the size of the tabu list  ")
                    try:
                        print("Running standard ACO with Tabu Search local search heuristic! Please wait...")
                        best_path, best_distance = ant_colony_optimization_tabu_search(input_data=input_data, 
                                                                                    num_ants=int(num_ants), 
                                                                                    num_iterations=int(num_iterations),
                                                                                    alpha=float(alpha), 
                                                                                    beta=float(beta), 
                                                                                    rho=float(rho), 
                                                                                    Q=float(Q),
                                                                                    tabu_iterations=int(tabu_iterations),
                                                                                    tabu_size=int(tabu_size)
                        )
                        print(f'Best path found: {best_path}')
                        print(f'Path distance: {best_distance}\n')
                        continue
                    except:
                        print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                        continue
                elif heuristic == "h":
                    try:
                        print("Running standard ACO with Hill Climbing local search heuristic! Please wait...")
                        best_path, best_distance = ant_colony_optimization_hill_climbing(input_data=input_data, 
                                                                                    num_ants=int(num_ants), 
                                                                                    num_iterations=int(num_iterations),
                                                                                    alpha=float(alpha), 
                                                                                    beta=float(beta), 
                                                                                    rho=float(rho), 
                                                                                    Q=float(Q)                                                                                   
                        )
                        print(f'Best path found: {best_path}')
                        print(f'Path distance: {best_distance}\n')
                        continue
                    except:
                        print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                        continue
                elif heuristic == "2":
                    try:
                        print("Running standard ACO with 2-opt local search heuristic! Please wait...")
                        best_path, best_distance = ant_colony_optimization_two_opt(input_data=input_data, 
                                                                                    num_ants=int(num_ants), 
                                                                                    num_iterations=int(num_iterations),
                                                                                    alpha=float(alpha), 
                                                                                    beta=float(beta), 
                                                                                    rho=float(rho), 
                                                                                    Q=float(Q)                                                                                   
                        )
                        print(f'Best path found: {best_path}')
                        print(f'Path distance: {best_distance}\n')
                        continue
                    except:
                        print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                        continue
                elif heuristic == "none":
                    try:
                        print("Running standard ACO without a local search heuristic! Please wait...")
                        best_path, best_distance = ant_colony_optimization(input_data=input_data, 
                                                                                num_ants=int(num_ants), 
                                                                                num_iterations=int(num_iterations),
                                                                                alpha=float(alpha), 
                                                                                beta=float(beta), 
                                                                                rho=float(rho), 
                                                                                Q=float(Q)                                                                                   
                        )
                        print(f'Best path found: {best_path}')
                        print(f'Path distance: {best_distance}\n')
                        continue
                    except:
                        print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                        continue
            elif code == 'm':
                max_p = input("Enter the maximum pheromone value limit  ")
                min_p = input("Enter the minimum pheromone value limit  ")
                try:
                    print("Running Max-Min Ant System ACO! Please wait...")
                    best_path, best_distance = MMAS_ant_colony_optimization(input_data=input_data, 
                                                                            num_ants=int(num_ants), 
                                                                            num_iterations=int(num_iterations),
                                                                            alpha=float(alpha), 
                                                                            beta=float(beta), 
                                                                            rho=float(rho), 
                                                                            Q=float(Q),
                                                                            max_p=float(max_p),
                                                                            min_p=float(min_p)                                                                                   
                    )
                    print(f'Best path found: {best_path}')
                    print(f'Path distance: {best_distance}\n')
                    continue
                except:
                    print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                    continue
            elif code == 'e':
                try:
                    print("Running Elitist Ant System ACO! Please wait...")
                    best_path, best_distance = elitist_ant_colony_optimization(input_data=input_data, 
                                                                            num_ants=int(num_ants), 
                                                                            num_iterations=int(num_iterations),
                                                                            alpha=float(alpha), 
                                                                            beta=float(beta), 
                                                                            rho=float(rho), 
                                                                            Q=float(Q)                                                                                   
                    )
                    print(f'Best path found: {best_path}')
                    print(f'Path distance: {best_distance}\n')
                    continue
                except:
                    print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                    continue
            elif code == 'r':
                try:
                    print("Running Rank-Based Ant System ACO! Please wait...")
                    best_path, best_distance = elitist_ant_colony_optimization(input_data=input_data, 
                                                                            num_ants=int(num_ants), 
                                                                            num_iterations=int(num_iterations),
                                                                            alpha=float(alpha), 
                                                                            beta=float(beta), 
                                                                            rho=float(rho), 
                                                                            Q=float(Q)                                                                                   
                    )
                    print(f'Best path found: {best_path}')
                    print(f'Path distance: {best_distance}\n')
                    continue
                except:
                    print("An error occured, make sure the name of the input file is spelt correctly and that you have entered parameters correctly\n")
                    continue



if __name__ == "__main__":
    main()