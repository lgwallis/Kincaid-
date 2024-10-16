"""
Simple greedy add for the first run, subsequent runs have an 80% chance of a greedy add using the probability matrix, and a 20% chance of generating a random solution 
Probability vector: When a location is used in a solution, it's "probability" is increased so that it has a lower chance of being chosen during greedy add...if a randomly generated number between 0 and 1 is lower than the probability, that location cannot be chosen 
Probability smoothing to prevent a probability from getting larger than 0.9

This code ran pretty slow (didn't use numpy arrays) and has some serious issues with its local search procedure, 
which is fine because this was my firt attempt. What helped me fix these issues was rewriting the local search in terms 
of candidate lists/candidate moves, and making sure that only one actual swap could be made for each iteration. The formatting 
of the output in the version also left a lot to be desired. However, its accuracy when given many runs wasn't actually too bad. 


"""

import random

def solve_pMedian(filename, nLOC, maxIT, tenure, runs, no_improvement_limit, seed):
    """
    Main function to solve the p-Median problem using a Tabu Search algorithm.

    Parameters:
    - filename: Path to the file containing the cost matrix.
    - nLOC: Number of locations to select.
    - maxIT: Maximum number of iterations for the Tabu search.
    - tenure: Duration for which a move remains tabu.
    - runs: Number of independent runs to execute.
    - no_improvement_limit: Limit for consecutive iterations without improvement before stopping.
    - seed: Seed for random number generation to ensure reproducibility.
    """

    def read_costMatrix():
        """
        Reads the cost matrix from the specified file.

        Returns:
        - costMatrix: A 2D list representing the cost matrix.
        """
        costMatrix = []
        # Open the file and read line by line
        with open(filename, 'r') as data:
            for line in data:
                if len(line) > 1:  # Skip empty lines
                    # Convert each line into a list of floats and append to costMatrix
                    costMatrix.append(list(map(float, line.split())))  
        return costMatrix

    def compute_objective(solution, costMatrix):
        """
        Computes the objective value for a given solution based on the cost matrix.

        Parameters:
        - solution: A list of selected locations.
        - costMatrix: A 2D list representing the cost matrix.

        Returns:
        - objVal: The total objective value for the solution.
        """
        objVal = 0  # Initialize objective value
        for i in range(len(costMatrix)):
            min_dist = float('inf')  # Start with infinite distance for each location
            # Find the minimum distance to the selected locations
            for j in solution:
                if costMatrix[i][j] < min_dist:
                    min_dist = costMatrix[i][j]
            objVal += min_dist  # Accumulate the minimum distance
        return objVal

    def normalize_probability_matrix(probabilityMatrix):
        """
        Normalizes the probability matrix such that each row sums to 1.

        Parameters:
        - probabilityMatrix: A 2D list representing the probability of selecting each location.

        Returns:
        - probabilityMatrix: The normalized probability matrix.
        """
        for i in range(len(probabilityMatrix)):
            row_sum = sum(probabilityMatrix[i])  # Calculate the sum of the row
            if row_sum != 0:  # Avoid division by zero
                for j in range(len(probabilityMatrix[i])):
                    # Normalize each element in the row
                    probabilityMatrix[i][j] /= row_sum  
        return probabilityMatrix

    def greedy_add(costMatrix, probabilityMatrix, nLOC):
        """
        Constructs an initial solution using a greedy addition method based on the cost matrix and probabilities.

        Parameters:
        - costMatrix: A 2D list representing the cost matrix.
        - probabilityMatrix: A 2D list representing the probability of selecting each location.
        - nLOC: Number of locations to select.

        Returns:
        - selected_locations: A list of selected locations.
        """
        selected_locations = []  # List to hold selected locations
        nRow = len(costMatrix)  # Number of locations in the cost matrix

        while len(selected_locations) < nLOC:  # Continue until we have selected nLOC locations
            best_addition = None  # Variable to track the best location to add
            best_objective = float('inf')  # Start with an infinitely large objective value

            for loc in range(nRow):
                if loc not in selected_locations:  # Only consider unselected locations
                    num = random.random()  # Generate a random number between 0 and 1
                    # Check if the random number is less than the probability of selecting the location
                    if probabilityMatrix[0][loc] < num:  
                        new_solution = selected_locations + [loc]  # Create a new solution with the added location
                        new_objective = compute_objective(new_solution, costMatrix)  # Compute its objective value

                        # Update best addition if the new objective is better
                        if new_objective < best_objective:
                            best_objective = new_objective
                            best_addition = loc  # Update the best location to add

            if best_addition is not None:
                selected_locations.append(best_addition)  # Add the best location to the selected list
            else:
                # If no location was added, select a random one to avoid infinite loops
                remaining_locations = [loc for loc in range(nRow) if loc not in selected_locations]
                selected_locations.append(random.choice(remaining_locations))

        print('The starting solution is: ' + str(selected_locations) + 
              ' and its objective value is ' + str(compute_objective(selected_locations, costMatrix)))
        return selected_locations

    def construct_starting_solution(costMatrix, probabilityMatrix, nLOC, run):
        """
        Constructs the initial solution based on the run number and probability matrix.

        Parameters:
        - costMatrix: A 2D list representing the cost matrix.
        - probabilityMatrix: A 2D list representing the probability of selecting each location.
        - nLOC: Number of locations to select.
        - run: Current run number.

        Returns:
        - An initial solution as a list of selected locations.
        """
        nRow = len(costMatrix)
        if run == 0:
            return greedy_add(costMatrix, probabilityMatrix, nLOC)  # For the first run, use greedy addition
        else:
            num = random.random()  # Generate a random number
            if num >= 0.2:  # 80% chance of using greedy add for intensification
                print('greedy add solution')
                return greedy_add(costMatrix, probabilityMatrix, nLOC)
            else:  # 20% chance of generating a random solution for diversification
                print('random solution')
                return [random.randrange(0, nRow) for i in range(nLOC)]

    def tabu_search(costMatrix, probabilityMatrix, run, tabu_list, best_evaluation_location, best_evaluation, best_solution):
        """
        Executes the Tabu Search algorithm to find the optimal solution.

        Parameters:
        - costMatrix: A 2D list representing the cost matrix.
        - probabilityMatrix: A 2D list representing the probability of selecting each location.
        - run: Current run number.
        - tabu_list: List of tabu moves.
        - best_evaluation_location: Tuple indicating the best evaluation location.
        - best_evaluation: Current best evaluation.
        - best_solution: Current best solution.

        Returns:
        - iteration_result: A dictionary containing results of the iteration.
        """
        nRow = len(costMatrix)  # Number of locations
        locations = list(range(nRow))  # All potential locations
        inList = construct_starting_solution(costMatrix, probabilityMatrix, nLOC, run)  # Initial solution
        outList = [i for i in locations if i not in inList]  # Locations not in the current solution
        best_local_solution = inList.copy()  # Track the best local solution
        best_local_obj = compute_objective(best_local_solution, costMatrix)  # Objective of the local solution
        iterations = []  # Track iterations
        evaluations = []  # Track evaluations
        aspiration_count = 0  # Count of aspiration moves
        no_improvement_iterations = 0  # Counter for iterations without improvement

        best_neighbor = None  # Best neighboring solution found
        best_neighbor_obj = float('inf')  # Initialize neighbor objective

        for iteration in range(maxIT):  # Iterate for a maximum number of iterations
            improved = False  # Track if an improvement is found
            for i in range(nLOC):
                for j in outList:  # Explore all swaps between inList and outList
                    neighbor_solution = inList.copy()  # Create a copy of the current solution
                    neighbor_solution[i] = j  # Make a swap
                    neighbor_obj = compute_objective(neighbor_solution, costMatrix)  # Evaluate the new solution
                    swap = (inList[i], j)  # Define the move as a swap

                    # Check if the swap is tabu but improves the best known evaluation
                    if swap in tabu_list and neighbor_obj < best_evaluation:
                        best_neighbor = neighbor_solution
                        best_local_obj = neighbor_obj
                        best_evaluation = neighbor_obj  # Update the best evaluation
                        aspiration_count += 1  # Increment aspiration count
                        tabu_list.remove(swap)  # Remove the swap from the tabu list
                        improved = True

                    # If the swap is not tabu and improves the local solution
                    elif swap not in tabu_list and neighbor_obj < best_local_obj:
                        best_neighbor = neighbor_solution
                        best_local_obj = neighbor_obj  # Update the best local objective
                        tabu_list.append(swap)  # Add the swap to the tabu list
                        improved = True

            # Update the current solution if a better neighbor is found
            if best_neighbor is not None:
                inList = best_neighbor
                outList = [i for i in locations if i not in inList]  # Update outList

            # Update the best solution found if improved
            if best_local_obj < best_evaluation:
                best_solution = best_neighbor
                best_evaluation = best_local_obj
                best_evaluation_location = (iteration, run)  # Update the best evaluation location

            # Limit the size of the tabu list based on the tenure
            if len(tabu_list) > tenure:
                tabu_list.pop(0)

            # If no improvement was found, increment the counter
            if not improved:
                no_improvement_iterations += 1
                if no_improvement_iterations >= no_improvement_limit:  # Stop if limit exceeded
                    print('Long-term memory triggered after ' + str(no_improvement_iterations) + ' iterations without improvement.')
                    break
            else:
                no_improvement_iterations = 0  # Reset counter if improvement found

            if best_solution is None:  # Initialize best solution if it's not set
                best_solution = best_local_solution

            iterations.append(iteration)  # Record the iteration
            evaluations.append(best_evaluation)  # Record the best evaluation
            print('Iteration: ' + str(iteration) +
                  ' | Best local objective: ' + str(best_local_obj) +
                  ' | Best overall objective value: ' + str(best_evaluation) +
                  ' | Aspiration count for current run: ' + str(aspiration_count))

            # Update the probability matrix based on the locations in the best local solution
            for location in best_local_solution:
                probabilityMatrix[0][location] += 0.1  # Increase probability for selected locations
                # Smooth the probability if it exceeds a threshold
                if probabilityMatrix[0][location] >= 0.9:
                    probabilityMatrix[0][location] = 0.8

            # Normalize the probability matrix after updates
            probabilityMatrix = normalize_probability_matrix(probabilityMatrix)

        # Prepare the result for this iteration
        iteration_result = {
            'best solution': best_solution,
            'best evaluation': best_evaluation,
            'iterations': iterations,
            'evaluations': evaluations,
            'aspiration count': aspiration_count,
            'starting solution': best_solution,
            'tabu_list': tabu_list,
            'best_evaluation_location': best_evaluation_location
        }

        run += 1  # Increment run counter

        return iteration_result

    random.seed(seed)  # Set the random seed for reproducibility
    costMatrix = read_costMatrix()  # Read the cost matrix from the file
    # Initialize the probability matrix with uniform probabilities
    probabilityMatrix = [[(1/nLOC) for _ in range(len(costMatrix))]]  
    probabilityMatrix = normalize_probability_matrix(probabilityMatrix)  # Normalize the probability matrix
    tabu_list = []  # Initialize the tabu list
    best_solution = [None] * nLOC  # Best solution placeholder
    best_evaluation_location = (0, 0)  # Track the best evaluation location
    best_evaluation = float('inf')  # Initialize the best evaluation to infinity

    for run in range(runs):  # Execute multiple runs of the Tabu Search
        print('-------------------- Run #' + str(run) + ' --------------------')
        # Execute the Tabu Search for the current run
        iteration_result = tabu_search(costMatrix, probabilityMatrix, run, tabu_list, best_evaluation_location, best_evaluation, best_solution)
        # Update the global best solution if the current run yields a better solution
        if iteration_result['best evaluation'] < best_evaluation:
            best_evaluation = iteration_result['best evaluation']
            best_solution = iteration_result['best solution']
            best_evaluation_location = iteration_result['best_evaluation_location']
        # Output the best solution found in this run
        print('The best solution is ' + str(best_solution) + '\n' 
              + 'its objective value is ' + str(best_evaluation) + '\n'
              + 'its length is ' + str(len(best_solution)) + '\n'
              + 'and it was found in iteration ' + str(best_evaluation_location[0]) + 
              ' of run ' + str(best_evaluation_location[1]))

# Entry point of the program
if __name__ == '__main__':
    solve_pMedian(filename='spd100b(1).dat', nLOC=12, maxIT=50, tenure=35, runs=50, no_improvement_limit=100, seed=42)
