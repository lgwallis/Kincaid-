"""
V4 has a lot of promise that I didn't fully explore this summer. It was a bit of an afterthough because by the time 
I finished V3 I was getting a bit bored of the p-median problem and wanted to start writing code for the p-dispersion 
problem (V5). As such, this is kind of a half-baked attempt. My intention with this version was to more deeply explore 
the tradeoffs between intensification and diversification, and implement different ways of going about those two search modes. 
Really the only new strategy I implemented here was choosing the locations with the highest values in the probabiity matrix for 
intensify mode and the lowest ones for diversify mode. I did not play around with this version enough to see if I implemented 
this idea rightly but I think this version could lead to some very interesting research in terms of constructing new starting solutions. 
"""

import random
import numpy as np

def solve_pMedian(filename, nLOC, maxIT, tenure, runs, no_improvement_limit, seed, penalty_fraction, alpha=0.1, rho=0.9, smoothing_threshold=0.05):
    """
    Solves the p-Median problem using a tabu search algorithm.

    Parameters:
    - filename: str, path to the file containing the cost matrix.
    - nLOC: int, number of locations to select.
    - maxIT: int, maximum number of iterations for the tabu search.
    - tenure: int, tabu tenure (number of iterations a move remains tabu).
    - runs: int, number of independent runs of the algorithm.
    - no_improvement_limit: int, number of iterations with no improvement before terminating a run.
    - seed: int, random seed for reproducibility.
    - penalty_fraction: float, fraction of the maximum distance to use for penalizing locations.
    - alpha: float, learning rate for updating the probability vector.
    - rho: float, smoothing parameter for probability vector adjustments.
    - smoothing_threshold: float, threshold for applying smoothing to the probability vector.
    """
    
    def read_costMatrix():
        """Reads the cost matrix from the given filename."""
        return np.loadtxt(filename)

    def compute_objective(solution, distances):
        """
        Computes the objective value of a solution.

        Parameters:
        - solution: list or set, selected locations.
        - distances: ndarray, cost matrix.

        Returns:
        - float, sum of the minimum distances for each customer to the nearest selected location.
        """
        solution_list = list(solution) if isinstance(solution, set) else solution
        return np.min(distances[solution_list], axis=0).sum()

    def greedy_add(distances, nLOC, penalty_matrix):
        """
        Constructs a starting solution using a greedy add algorithm.

        Parameters:
        - distances: ndarray, cost matrix.
        - nLOC: int, number of locations to select.
        - penalty_matrix: ndarray, penalty matrix to diversify the search.

        Returns:
        - list, selected locations.
        """
        penalized_distances = distances + penalty_matrix
        selected_locations = set()
        nRow = distances.shape[0]

        while len(selected_locations) < nLOC:
            remaining_locations = set(range(nRow)) - selected_locations
            best_addition = min(remaining_locations, key=lambda loc: compute_objective(selected_locations.union({loc}), penalized_distances))
            selected_locations.add(best_addition)

        selected_locations = list(selected_locations)
        print(f'The starting solution is: {selected_locations} and its objective value is {compute_objective(selected_locations, distances):.2f}')
        return selected_locations

    def construct_starting_solution(distances, nLOC, run, penalty_matrix, P):
        """
        Constructs a starting solution based on the run number.

        Parameters:
        - distances: ndarray, cost matrix.
        - nLOC: int, number of locations to select.
        - run: int, current run number.
        - penalty_matrix: ndarray, penalty matrix to diversify the search.
        - P: ndarray, probability vector for selecting locations.

        Returns:
        - list, selected locations.
        - bool, whether the solution is greedy.
        - str, mode used for generating the solution.
        """
        nRow = distances.shape[0]
        selected_locations = set()
        mode_used = None

        if run == 0:
            print('Greedy add solution')
            return greedy_add(distances, nLOC, penalty_matrix), True, 'Greedy'
        else:
            print('Combinations-based solution')
            while len(selected_locations) < nLOC:
                if random.random() < 0.2:
                    new_location = random.choice(list(set(range(nRow)) - selected_locations))
                    mode_used = 'Random'
                else:
                    mode = 'intensify' if random.random() < 0.5 else 'diversify'
                    if mode == 'intensify':
                        mode_used = 'Intensify'
                        candidates = np.argsort(P)[-nLOC:]  # Select the p largest values in the probability vector P
                    else:
                        mode_used = 'Diversify'
                        candidates = np.argsort(P)[:nLOC]  # Select the p smallest values in the probability vector P

                    candidates = list(candidates)  # Ensure candidates is a list

                    if not candidates:
                        raise ValueError("No candidates available for selection.")

                    new_location = random.choice(candidates)
                    while new_location in selected_locations:
                        candidates.remove(new_location)
                        if not candidates:
                            raise ValueError("No new candidates available for selection.")
                        new_location = random.choice(candidates)
                
                selected_locations.add(new_location)
        
            print(f'The starting solution is: {selected_locations} and its objective value is {compute_objective(selected_locations, distances):.2f}')
            if mode_used:
                print(f'Starting solution was generated using {mode_used} mode')
            return list(selected_locations), False, mode_used

    def make_tabu(stmMatrix, swap, iteration):
        """
        Adds a move to the tabu list.

        Parameters:
        - stmMatrix: ndarray, tabu status matrix.
        - swap: tuple, the move (i, j) where i is out and j is in.
        - iteration: int, current iteration number.
        """
        i, j = swap
        stmMatrix[i, j] = stmMatrix[j, i] = iteration + tenure

    def is_tabu(stmMatrix, swap, iteration):
        """
        Checks if a move is tabu.

        Parameters:
        - stmMatrix: ndarray, tabu status matrix.
        - swap: tuple, the move (i, j) where i is out and j is in.
        - iteration: int, current iteration number.

        Returns:
        - bool, True if the move is tabu, False otherwise.
        """
        i, j = swap
        return stmMatrix[i, j] > iteration or stmMatrix[j, i] > iteration

    def apply_probability_smoothing(P, threshold, rho):
        """
        Applies smoothing to the probability vector.

        Parameters:
        - P: ndarray, probability vector.
        - threshold: float, threshold for applying smoothing.
        - rho: float, smoothing parameter.

        Returns:
        - ndarray, smoothed probability vector.
        """
        P_new = np.where(P > threshold, P * rho, P)
        scaling_factors = 1 - (1 - rho) * P
        P_new /= scaling_factors
        return P_new

    def tabu_search(distances, run, best_evaluation_location, best_evaluation, best_solution, stmMatrix, penalty_matrix, P):
        """
        Performs the tabu search.

        Parameters:
        - distances: ndarray, cost matrix.
        - run: int, current run number.
        - best_evaluation_location: tuple, location of the best evaluation found so far.
        - best_evaluation: float, best objective value found so far.
        - best_solution: list, best solution found so far.
        - stmMatrix: ndarray, tabu status matrix.
        - penalty_matrix: ndarray, penalty matrix to diversify the search.
        - P: ndarray, probability vector for selecting locations.

        Returns:
        - dict, results of the tabu search including best solution and objective value.
        """
        nRow = distances.shape[0]
        solution, is_greedy, mode_used = construct_starting_solution(distances, nLOC, run, penalty_matrix, P)
        inList = set(solution)
        outList = set(range(nRow)) - inList
        best_local_solution = inList.copy()
        best_local_obj = compute_objective(best_local_solution, distances)
        aspiration_count = 0
        no_improvement_iterations = 0
        last_improvement_obj = best_local_obj

        print(f"\n{'Iteration':<10}{'Swap (Out -> In)':<20}{'Iteration Obj':<20}{'Best Local Obj':<20}{'Best Global Obj':<20}{'Aspiration Move':<15}")
        print("-" * 105)

        for iteration in range(maxIT):
            best_move = None
            best_neighbor_obj = float('inf')
            is_aspiration_move = False

            for i in inList:
                for j in outList:
                    neighbor_solution = inList.copy()
                    neighbor_solution.remove(i)
                    neighbor_solution.add(j)
                    neighbor_obj = compute_objective(neighbor_solution, distances)
                    swap = (i, j)

                    is_tabu_move = is_tabu(stmMatrix, swap, iteration)
                    if (not is_tabu_move) or (is_tabu_move and neighbor_obj < best_evaluation):
                        if neighbor_obj < best_neighbor_obj:
                            best_move = (neighbor_solution, swap, neighbor_obj)
                            best_neighbor_obj = neighbor_obj
                            is_aspiration_move = is_tabu_move and neighbor_obj < best_evaluation

            if best_move is None:
                print("No non-tabu moves available. Terminating search.")
                break

            best_neighbor_solution, best_swap, best_neighbor_obj = best_move

            if is_aspiration_move:
                aspiration_count += 1

            make_tabu(stmMatrix, best_swap, iteration)

            inList = best_neighbor_solution
            outList = set(range(nRow)) - inList

            # Update the probability vector P
            in_loc, out_loc = best_swap
            P[in_loc] = (P[in_loc] * (1 - alpha)) + alpha
            P[out_loc] *= (1 - alpha)

            # Apply probability smoothing if necessary
            if P[in_loc] > smoothing_threshold:
                P = apply_probability_smoothing(P, smoothing_threshold, rho)

            if best_neighbor_obj < best_local_obj:
                best_local_solution = inList.copy()
                best_local_obj = best_neighbor_obj

            if best_local_obj < best_evaluation:
                best_evaluation = best_local_obj
                best_solution = best_local_solution
                best_evaluation_location = (run, iteration)

            if best_neighbor_obj < last_improvement_obj:
                last_improvement_obj = best_neighbor_obj
                no_improvement_iterations = 0
            else:
                no_improvement_iterations += 1

            print(f"{iteration:<10}{f'{best_swap[0]} -> {best_swap[1]}':<20}{best_neighbor_obj:<20.2f}{best_local_obj:<20.2f}{best_evaluation:<20.2f}{'Yes' if is_aspiration_move else 'No':<15}")

            if no_improvement_iterations >= no_improvement_limit:
                print("No improvement limit reached. Terminating search.")
                break

        results = {
            'best_evaluation': best_evaluation,
            'best_solution': best_solution,
            'best_evaluation_location': best_evaluation_location,
            'penalty_matrix': penalty_matrix,
            'probability_vector': P,
            'is_greedy': is_greedy,
            'mode_used': mode_used
        }

        return results

    distances = read_costMatrix()
    best_evaluation_location = None
    best_evaluation = float('inf')
    best_solution = None
    best_run = None

    print(f'Initial random seed: {seed}\n')
    random.seed(seed)

    P = np.ones(distances.shape[0]) / distances.shape[0]
    penalty_matrix = np.zeros(distances.shape)
    penalty_increment = penalty_fraction * np.max(distances)

    for run in range(runs):
        print(f'Run {run + 1}/{runs}\n{"-"*30}\n')
        stmMatrix = np.zeros((distances.shape[0], distances.shape[0]))

        results = tabu_search(distances, run, best_evaluation_location, best_evaluation, best_solution, stmMatrix, penalty_matrix, P)

        best_evaluation = results['best_evaluation']
        best_solution = results['best_solution']
        best_evaluation_location = results['best_evaluation_location']
        penalty_matrix = results['penalty_matrix']
        P = results['probability_vector']

        if results['is_greedy']:
            penalty_matrix[best_solution] += penalty_increment
            print(f'Penalty matrix updated for greedy solution: {penalty_matrix}\n')
        else:
            print(f'Penalty matrix not updated for non-greedy solution.\n')

    print(f'\nBest evaluation: {best_evaluation:.2f} at run {best_evaluation_location[0] + 1}, iteration {best_evaluation_location[1] + 1}')
    print(f'Best solution: {best_solution}\n')

    return best_solution

if __name__ == '__main__':
     # Solve the p-Median problem with the defined parameters
    solve_pMedian(
        filename='spd100new.dat', # File containing the distance matrix
        nLOC=12,                  # Number of locations to select
        maxIT=200,                # Maximum iterations per run
        tenure=35,                # Tabu tenure (how long a move remains tabu)
        runs=5,                   # Number of times to run the algorithm
        no_improvement_limit=1000,# Iterations without improvement before triggering long-term memory
        seed=123,                 # Random seed for reproducibility   
        penalty_fraction=0.005)   # Fraction of max distance to use as penalty