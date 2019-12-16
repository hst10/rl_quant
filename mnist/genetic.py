#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
import csv
import numpy as np
from deap import base
from deap import creator
from deap import tools
import evaluate_quant as model
from copy import deepcopy
import os, sys

OUTPATH = os.getcwd() + '/mnist/'

from parallelProcess import parallelProcess

# Set these numbers
AVG_ITERS = 1
INTERVAL = 50

def touch(fname):
    """
        Create a new file.
    """
    try:
        os.utime(fname, None)
    except OSError:
        open(fname, 'a').close()

def crossover(ind1, ind2):
    """
        Implements a specialized crossover function.

        There are 4 different components to the individual. The child shall inherit
        50% of each.
    """
    ret_ind = deepcopy(ind1)
    ind1,ind2 = ind1[0],ind2[0]
    Xprob = [1,1,0,0]
    for idx,prob in enumerate(Xprob):
        ret_ind[0][idx] = ind1[idx] if(prob==1) else ind2[idx]
    return ret_ind

def mutate(ind):
    """
        Randomly replace some of the components with a random value.

        The probability of replacement is 0.5.
    """
    # Create a random neighbor
    new_ind = model.getNeighbor()
    ret_ind = deepcopy(ind)
    ind=ind[0]
    # with equal probability, swap the random one to the existing individual
    Xprob = [random.random()>0.5 for _ in range(len(ind))]
    for idx,prob in enumerate(Xprob):
        ret_ind[0][idx] = new_ind[idx] if(prob) else ind[idx]
    return ret_ind

def evolutionaryLearn_unpack(args):
    return evolutionaryLearn(*args)

def evolutionaryLearn(costFn, getNeighbor, maxsteps=5000):

    """
        This is the main function that performs evolutionary learning.
    """

    # Minimizing objective
    creator.create("FitnessMax", base.Fitness, weights=[1.0,])
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator 
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability) 
    ## Import the generate schedule function #TODO
    toolbox.register("attr_bool", getNeighbor)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    # THis is not necessary #TODO
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        toolbox.attr_bool, 1)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    # This will be replaced with our eval function #TODO
    def evalOneMax(individual):
        return [costFn(individual[0]),]

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)

    # random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=50)

    # Check for change in probability
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.05

    # print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    evals = 0

    # Variable for keeping the best result
    result = []
    stats = []
    eval_count = []

    # Save the results.
    fname = OUTPATH + 'saved_genetic.npy'
    touch(fname)

    # Begin the evolution and run for number of iterations.
    while evals <= maxsteps:
        # A new generation
        g = g + 1

        # Anneal the crossover and mutation probabilities
        if((g%25==0) and (g!=0)):
           CXPB *= 0.75
           MUTPB *= 0.75

        # print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # We will slice the arrays for
        # Apply crossover and mutation on the offspring
        for idx in range(len(offspring)/2):
            child1_idx, child2_idx = idx, (len(offspring)/2)+idx
            child1, child2 = offspring[child1_idx], offspring[child2_idx]

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                new_child1 = crossover(child1, child2)
                new_child2 = crossover(child2, child1)

                # Perform in-place update
                offspring[child1_idx], offspring[child2_idx] = new_child1,new_child2
                # fitness values of the children
                # must be recalculated later
                del offspring[child1_idx].fitness.values
                del offspring[child2_idx].fitness.values

                # This involves 2 evaluations.
                evals += 2

        for idx in range(len(offspring)):

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                evals += 1
                offspring[idx] = mutate(offspring[idx])
                del offspring[idx].fitness.values
                evals += 2

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print("  Evaluated %i individuals" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]


        ## Store the good results
        ## Note that the number of evals
        eval_count.append(evals)
        best_ind = tools.selBest(pop, 1)[0]
        layerwise_quant = [best_ind[4*i:4*(i+1)] for i in range(4)]
        result.append(best_ind.fitness.values[0])
        print(best_ind.fitness.values[0])
        modelStat = model.getStats(best_ind[0])
        print(modelStat)
        stats.append(modelStat)

        if((evals%50 == 0) and (evals!=0)):
            final_result = [eval_count, result, stats]
            np.save(fname, final_result)
            print("Finished {0} iterations ".format(evals))

        print("Finished {0} populations ".format(g))

    # print("-- End of (successful) evolution --")

    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # return best_ind.fitness.values[0]

    print("Preprocessing now")

    print(eval_count, result, stats)
    # Write the results and return the values
    final_result = [eval_count, result, stats]
    np.save(fname, final_result)

    costArr = post_process(eval_count, result, maxsteps)
    # print(pid, result)

    print("Finished ")

    return costArr

def post_process(eval_count, result, maxsteps):
    """
        We want to get the same number of outputs for each trajectory.
    """
    step_count = INTERVAL
    idx = 0
    costArr = []

    # The retrned array shoul be of this length:
    arr_len = (maxsteps/INTERVAL) - 1
    for idx, count in enumerate(eval_count):
        if(count > step_count):
            sliced = result[:idx]
            if sliced:
                costArr.append(min(sliced))
            step_count += INTERVAL

    costArr_len = len(costArr)
    if(costArr_len > arr_len):
        return np.array(costArr[:arr_len])
    elif(costArr_len < arr_len):
        deficit_arr = [costArr[-1]]*(arr_len-costArr_len)
        return np.array(costArr + deficit_arr)
    else:
        return np.array(costArr)

def main():

    # For different number of iterations, run the simulated annealing
    # call the genetic algorithm
    data = evolutionaryLearn(model.getReward, model.getNeighbor,maxsteps=1000)
    # Dump the data into a CSV
    with open('dataGenetic.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for lst in data:
            # lst = np.array(lst)
            wr.writerow(lst)

if __name__ == '__main__':
    # runGenetic()
    main()
    print("Done!")
