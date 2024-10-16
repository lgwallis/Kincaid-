"""
Issues that I fixed from V1:
• Took out the probability matrix and probability smoothing
• Changed the logic of the local tabu search, made it more clear that
the algorithm was choosing the best move from the candidate list at
every iteration
• Caught a few errors that I fixed: making swaps that were on the
tabu list (it helped to print out the swaps), making several moves per
iteration

This version is basically an improved V1. However, it still runs really slow which makes it difficult to test. 

"""

import random

def solve_pMedian(filename, nLOC, maxIT, tenure, runs, no_improvement_limit, seed):
    """
    Main function to solve the p-Median problem using Tabu Search.
    
    Parameters:
    filename (str): The file containing the cost matrix.
    nLOC (int): Number of locations to be selected.
    maxIT (int): Maximum number of iterations for the Tabu Search.
    tenure (int): Tabu tenure, i.e., how long a move remains tabu.
    runs (int): Number of independent runs of the Tabu Search.
    no_improvement_limit (int): Number of iterations with no improvement before triggering long-term memory.
    seed (int): Random seed for reproducibility.
    """

    def read_costMatrix():
        """
        Reads the cost matrix from a file and returns it as a 2D list.
        
        Returns:
        costMatrix (list of lists): The cost matrix.
        """
        costMatrix = []
        with open(filename, 'r') as data:
            for line in data:
                if len(line.strip()) > 0:
                    costMatrix.append(list(map(float, line.split())))  # Convert each line into a list of floats
        return costMatrix

    def compute_objective(solution, costMatrix):
        """
        Computes the objective value (total distance) of a given solution.
        
        Parameters:
        solution (list of int): List of selected locations.
        costMatrix (list of lists): The cost matrix.
        
        Returns:
        objVal (float): The objective value of the solution.
        """
        objVal = 0
        for i in range(len(costMatrix)):
            min_dist = float('inf')
            for j in solution:
                if costMatrix[i][j] < min_dist:
                    min_dist = costMatrix[i][j]
            objVal += min_dist
        return objVal

    def greedy_add(costMatrix, nLOC):
        """
        Constructs a starting solution using a greedy algorithm.
        
        Parameters:
        costMatrix (list of lists): The cost matrix.
        nLOC (int): Number of locations to be selected.
        
        Returns:
        selected_locations (list of int): List of selected locations.
        """
        selected_locations = []
        nRow = len(costMatrix)

        while len(selected_locations) < nLOC:
            best_addition = None
            best_objective = float('inf')

            for loc in range(nRow):
                if loc not in selected_locations:
                    new_solution = selected_locations + [loc]
                    new_objective = compute_objective(new_solution, costMatrix)

                    if new_objective < best_objective:
                        best_objective = new_objective
                        best_addition = loc

            if best_addition is not None:
                selected_locations.append(best_addition)
            else:
                # If no location was added, select a random one to avoid infinite loops
                remaining_locations = [loc for loc in range(nRow) if loc not in selected_locations]
                selected_locations.append(random.choice(remaining_locations))

        print('The starting solution is: ' + str(selected_locations) + ' and its objective value is ' + str(compute_objective(selected_locations, costMatrix)))
        return selected_locations

    def construct_starting_solution(costMatrix, nLOC, run):
        """
        Constructs the starting solution for the given run.
        
        Parameters:
        costMatrix (list of lists): The cost matrix.
        nLOC (int): Number of locations to be selected.
        run (int): The current run number.
        
        Returns:
        (list of int): The starting solution.
        """
        nRow = len(costMatrix)
        if run == 0:
            return greedy_add(costMatrix, nLOC)  # For the first run, use greedy add
        else:
            num = random.random()
            if num >= 0.4:  # 60% chance to use greedy add (intensification)
                print('Greedy add solution')
                return greedy_add(costMatrix, nLOC)
            else:  # 40% chance to restart with a random solution (diversification)
                print('Random solution')
                randSolution = [random.randrange(0, nRow) for i in range(nLOC)]
                print('The starting solution is: ' + str(randSolution) + ' and its objective value is ' + str(compute_objective(randSolution, costMatrix)))
                return randSolution

    def make_tabu(stmMatrix, swap, iteration):
        """
        Marks a swap move as tabu by updating the tabu matrix.
        
        Parameters:
        stmMatrix (list of lists): The tabu matrix.
        swap (tuple of int): The locations swapped (out, in).
        iteration (int): The current iteration number.
        """
        i, j = swap
        stmMatrix[i][j] = iteration + tenure
        stmMatrix[j][i] = iteration + tenure

    def is_tabu(stmMatrix, swap, iteration):
        """
        Checks if a swap move is tabu.
        
        Parameters:
        stmMatrix (list of lists): The tabu matrix.
        swap (tuple of int): The locations swapped (out, in).
        iteration (int): The current iteration number.
        
        Returns:
        bool: True if the move is tabu, False otherwise.
        """
        i, j = swap
        return stmMatrix[i][j] > iteration or stmMatrix[j][i] > iteration

    def tabu_search(costMatrix, run, best_evaluation_location, best_evaluation, best_solution, stmMatrix):
        """
        Performs the Tabu Search algorithm.
        
        Parameters:
        costMatrix (list of lists): The cost matrix.
        run (int): The current run number.
        best_evaluation_location (tuple of int): Location where the best evaluation was found.
        best_evaluation (float): The best evaluation found so far.
        best_solution (list of int): The best solution found so far.
        stmMatrix (list of lists): The tabu matrix.
        
        Returns:
        dict: Results of the tabu search including the best solution, evaluation, aspiration count, and evaluation location.
        """
        nRow = len(costMatrix)
        locations = list(range(nRow))
        inList = construct_starting_solution(costMatrix, nLOC, run)
        outList = [i for i in locations if i not in inList]
        best_local_solution = inList.copy()
        best_local_obj = compute_objective(best_local_solution, costMatrix)
        aspiration_count = 0
        no_improvement_iterations = 0
        last_improvement_obj = best_local_obj

        for iteration in range(maxIT):
            candidate_moves = []
            candidate_objs = []

            for i in range(nLOC):
                for j in outList:
                    neighbor_solution = inList.copy()
                    neighbor_solution[i] = j  # Make a swap
                    neighbor_obj = compute_objective(neighbor_solution, costMatrix)
                    swap = (inList[i], j)

                    if not is_tabu(stmMatrix, swap, iteration) or (is_tabu(stmMatrix, swap, iteration) and neighbor_obj < best_evaluation):
                        candidate_moves.append((neighbor_solution, swap, neighbor_obj))
                        candidate_objs.append(neighbor_obj)

            if not candidate_moves:
                print("No non-tabu moves available. Terminating search.")
                break

            best_move = min(candidate_moves, key=lambda x: x[2])
            best_neighbor_solution, best_swap, best_neighbor_obj = best_move

            # Aspiration criterion: accept a tabu move only if it's better than the best known solution
            if best_neighbor_obj < best_evaluation:
                aspiration_count += 1

            make_tabu(stmMatrix, best_swap, iteration)

            inList = best_neighbor_solution
            outList = [i for i in locations if i not in inList]

            if best_neighbor_obj < best_local_obj:
                best_local_obj = best_neighbor_obj
                no_improvement_iterations = 0  # Reset counter when there's an improvement
                last_improvement_obj = best_local_obj
            else:
                no_improvement_iterations += 1

            if best_local_obj < best_evaluation:
                best_solution = inList.copy()
                best_evaluation = best_local_obj
                best_evaluation_location = (iteration, run)

            print('Iteration: ' + str(iteration) +
                  ' | Swap: ' + str(best_swap) +
                  ' | Iteration objective: ' + str(best_neighbor_obj) +
                  ' | Best local objective: ' + str(best_local_obj) +
                  ' | Best global objective: ' + str(best_evaluation))

            if no_improvement_iterations >= no_improvement_limit:
                print('Long-term memory triggered after ' + str(no_improvement_iterations) + ' iterations without improvement.')
                print('Last improvement objective: ' + str(last_improvement_obj))
                break

        return {
            'best solution': best_solution,
            'best evaluation': best_evaluation,
            'aspiration count': aspiration_count,
            'best_evaluation_location': best_evaluation_location
        }

    # Set the random seed for reproducibility
    random.seed(seed)
    # Read the cost matrix from the file
    costMatrix = read_costMatrix()
    # Initialize the tabu matrix with zeros
    stmMatrix = [[0]*len(costMatrix) for _ in range(len(costMatrix))]
    best_solution = None
    best_evaluation_location = (0, 0)
    best_evaluation = float('inf')

    for run in range(runs):
        print('-------------------- Run #' + str(run) + ' --------------------')
        # Perform tabu search for each run
        iteration_result = tabu_search(costMatrix, run, best_evaluation_location, best_evaluation, best_solution, stmMatrix)
        
        # Check if the current run found a better evaluation than previously known
        if iteration_result['best evaluation'] < best_evaluation:
            best_evaluation = iteration_result['best evaluation']  # Update the best evaluation
            best_solution = iteration_result['best solution']  # Update the best solution
            best_evaluation_location = iteration_result['best_evaluation_location']  # Update the location of the best evaluation
        
        # Print the best solution found in the current run
        print('The best solution is ' + str(best_solution) + '\n' 
              + 'its objective value is ' + str(best_evaluation) + '\n'
              + 'and it was found in iteration ' + str(best_evaluation_location[0]) + 
              ' of run ' + str(best_evaluation_location[1]))

if __name__ == '__main__':
    # Call the solve_pMedian function with parameters for the problem
    solve_pMedian(filename='spd19.dat', nLOC=80, maxIT=30, tenure=35, runs=5, no_improvement_limit=100, seed=123)

