# -*- coding: utf-8 -*-
from __future__ import print_function
import os,sys
import math
import random
from simanneal import Annealer
import numpy as np
import evaluate_quant as model
import csv
from joblib import Parallel, delayed
from parallelProcess import parallelProcess

AVG_ITERS = 1
OUTPATH = os.getcwd() + '/mnist/'
TEMP = 50

def touch(fname):
    """
        Create a new file.
    """
    try:
        os.utime(fname, None)
    except OSError:
        open(fname, 'a').close()

def simAnneal_unpack(args):
    return simAnneal(*args)

def acceptPt(cost, new_cost, T):
    """
        Probablity of accepting
    """
    np.seterr(all='ignore')

    if(cost < new_cost):
        accept_prob = 1.0
    else:
        accept_prob = 1.0/(1.0+np.exp(float(-new_cost+cost)/float(T)))
    # print("For the requested new cost {0}, changed from original {1}, the p is {2}".format(new_cost, cost, accept_prob))
    accept =accept_prob > random.random()
    # raw_input(str(accept))
    return accept

def simAnneal(costFn, getNeighbour, temperature, maxsteps=5000):
    """
        Implement Simulated Annealing.
    """

    # Save the results.
    fname = OUTPATH + 'saved_simAnneal.npy'
    touch(fname)

    # Start the temperature
    T = temperature

    # Start at a random place
    starting = getNeighbour()
    stat = costFn(starting)
    start_cost = stat[-1]
    state, cost = starting, start_cost

    states, costs = [starting,], [stat,]

    for step in range(maxsteps):

        # Anneal
        if((step%200 == 0) and (step != 0)):
            T = T * 0.5

        # Periodic save
        if((step%100 == 0)):
            print("Complete {0} iteration ".format(step))
            np.save(fname, [states,costs])

        # New cost and state
        new_state = getNeighbour()
        new_stat = costFn(new_state)
        new_cost = new_stat[-1]

        # Accepeted the move
        if(acceptPt(cost, new_cost, T)):
            state, cost, stat = new_state, new_cost, new_stat
            states.append(state)

        print(stat)

        costs.append(stat)

    np.save(fname, [states,costs])
    return costs

def main():

    data = simAnneal(model.getStats, model.getNeighbor, TEMP, maxsteps=1000)

    # Dump the data into a CSV
    with open('dataSimAnneal.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for lst in data:
            wr.writerow(lst)

if __name__ == '__main__':
    # runSimAnn()
    main()
    print("Done!")
