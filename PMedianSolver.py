import random
import matplotlib.pyplot as plt
import time 

def solve_pMedian(filename, nLOC, maxIT, tenure, runs):

    def read_costMatrix():
        costMatrix = []
        with open(filename, 'r') as data:
            start = 0
            while start <= nLOC:
                for line in data:
                    costMatrix.append(list(map(int, line.split()))) #turn the file into a 2D matrix
                start += 1
        return costMatrix

    def compute_objective(solution, costMatrix):
        objVal = 0
        for i in range(len(costMatrix)): 
            min_dist = float('inf')
            for j in solution: 
                if costMatrix[i][j] < min_dist:
                    min_dist = costMatrix[i][j] 
            objVal += min_dist
        return objVal

    def greedy_add(costMatrix, probabilityMatrix, nLOC):
        selected_locations = []
        nRow = len(costMatrix)
        
        while len(selected_locations) < nLOC:
            best_addition = None
            best_objective = float('inf')

            for loc in range(nRow):
                if loc not in selected_locations:
                    num = random.random()
                    if probabilityMatrix[loc] < num:
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

    def construct_starting_solution(costMatrix, probabilityMatrix, nLOC, run):
        nRow = len(costMatrix)
        if run == 0:
            return greedy_add(costMatrix, probabilityMatrix, nLOC) #for the very first time, just do a simple greedy add 
        else:
            num = random.random()
            if num >= 0.4: # 80% chance that we use the probability matrix with a greedy add to construct the starting solution (intensification)
                print('greedy add solution')
                return greedy_add(costMatrix, probabilityMatrix, nLOC) 
            else: # 20% chance that we restart with a totally random solution (diversification)
                print('random solution')
                return [random.randrange(0, nRow) for i in range(nLOC)]

    def tabu_search(costMatrix, probabilityMatrix, run, tabu_list, best_evaluation_location, best_evaluation, best_solution):
        nRow = len(costMatrix)
        locations = list(range(nRow))
        inList = construct_starting_solution(costMatrix, probabilityMatrix, nLOC, run)  # use the probability matrix to come up with the initial solution
        outList = [i for i in locations if i not in inList]
        best_local_solution = inList.copy()
        best_local_obj = compute_objective(best_local_solution, costMatrix)  
        iterations = []
        evaluations = []
        aspiration_count = 0

        best_neighbor = None
        best_neighbor_obj = float('inf')

        for iteration in range(maxIT):
            for i in range(nLOC):
                for j in outList:
                    neighbor_solution = inList.copy()
                    neighbor_solution[i] = j #make a swap
                    neighbor_obj = compute_objective(neighbor_solution, costMatrix)
                    swap = (inList[i], j)

                    if swap in tabu_list and neighbor_obj < best_evaluation:
                        best_neighbor = neighbor_solution
                        best_local_obj = neighbor_obj
                        best_evaluation = neighbor_obj
                        aspiration_count += 1
                        tabu_list.remove(swap)

                    elif swap not in tabu_list and neighbor_obj < best_local_obj:
                        best_neighbor = neighbor_solution
                        best_local_obj = neighbor_obj
                        tabu_list.append(swap)

                if best_neighbor is not None:
                    inList = best_neighbor
                    outList = [i for i in locations if i not in inList]

                if best_local_obj < best_evaluation:
                    best_solution = best_neighbor
                    best_evaluation = best_local_obj
                    best_evaluation_location = (iteration, run)

                if len(tabu_list) > tenure:
                    tabu_list.pop(0)

            iterations.append(iteration)
            evaluations.append(best_evaluation)
            print('Iteration: ' + str(iteration) +
              ' | Best local objective: ' + str(best_local_obj) +
              ' | Best overall objective value: ' + str(best_evaluation) +
              ' | Aspiration count for current run: ' + str(aspiration_count))

            for location in best_local_solution:
                probabilityMatrix[location] += 0.2
        
        print('the tabu list is ' + str(tabu_list))

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

        run += 1

        return iteration_result

    costMatrix = read_costMatrix()
    probabilityMatrix = [(1/nLOC) for i in range(len(costMatrix))]  # Initialize penalty matrix based on cost matrix
    tabu_list = []
    best_solution = [None] * nLOC
    best_evaluation_location = (0, 0)
    best_evaluation = float('inf')
    
    for run in range(runs):
        print('-------------------- Run #' + str(run) + ' --------------------')
        iteration_result = tabu_search(costMatrix, probabilityMatrix, run, tabu_list, best_evaluation_location, best_evaluation, best_solution)
        if iteration_result['best evaluation'] < best_evaluation:
            best_evaluation = iteration_result['best evaluation']
            best_solution = iteration_result['best solution']
            best_evaluation_location = iteration_result['best_evaluation_location']
        print('The best solution is ' + str(best_solution) + '\n' 
              + 'its objective value is ' + str(best_evaluation) + '\n'
              + 'its length is ' + str(len(best_solution)) + '\n'
             + 'and it was found in iteration ' + str(best_evaluation_location[0]) + 
             ' of run ' + str(best_evaluation_location[1]))

if __name__ == '__main__':
    solve_pMedian(filename = 'spd19.dat', nLOC = 80, maxIT = 100, tenure = 100, runs = 30)


#things to add
#probability smoothing 
#change probability matrix after each update so that the rows all sum to one, so its a real probability matrix 

#things to fix
#local search doesn't do much after the first three iterations 

