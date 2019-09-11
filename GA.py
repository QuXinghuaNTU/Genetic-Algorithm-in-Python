"""
@author: Qu Xinghua (xinghua001@e.ntu.edu.sg)
Sep 2019@NTU
"""
import numpy as np
from scipy import randn, zeros
from scipy import random as rd, array
from random import choice, random, gauss, shuffle, sample
from numpy import ndarray

class GeneticAlgorithm():
    def __init__(self,function,popsize,max_generation,x_initial,L_bound,U_bound):
        self.popsize    = popsize
        self.population = None
        self.top_proportion   = 0.2
        self.elite_proportion = 0.5*self.top_proportion
        self.top_size         = self.top_proportion*self.popsize
        self.parents_size = int(self.popsize*self.top_proportion)
        self.elite_size = int(self.popsize*self.elite_proportion)
        #: mutation probability
        self.mutationProb = 0.1
        self.mutationStdDev = 0.5
        self.initRangeScaling = 10.
        self.mustMaximize   = True
        self.max_generation = max_generation
        self.best_individuals = []
        self.pop = None
        self.x_initial = x_initial
        self.dimension = len(x_initial)
        self.L_bound = L_bound
        self.U_bound = U_bound
        self.initialPopulation = None
        self.best_fitness      = []
        self.best_individual   = []
        self.objective = function

        
    def initial_pop(self):
        if self.initialPopulation is not None:
            self.pop = self.initialPopulation
        else:
            self.pop = [self.x_initial]
            for i in range(self.popsize):
                indiviudual = self.x_initial+randn(self.dimension)*self.mutationStdDev*self.initRangeScaling
#                 indiviudual = self.x_initial + np.multiply(randn(self.dimension),np.array(self.U_bound)-np.array(self.L_bound))
                indiviudual = np.ndarray.clip(indiviudual,self.L_bound,self.U_bound)
                self.pop.append(indiviudual)
        return self.pop
    
    def evaluate(self,function,individual):
        fitness = function(individual)
        self.fitnesses.append(fitness)
        return self.fitnesses
            
    
    def select(self):
        """ select some of the individuals of the population, taking into account their fitnesses
        :return: list of selected parents """
        tmp = list(zip(self.fitnesses, self.pop))
        tmp.sort(key = lambda x: x[0])
        tmp2 = list(reversed(tmp))[:self.parents_size]
        return [x[1] for x in tmp2]
    
    def crossOver(self, parents, nbChildren):
        xdim = self.dimension
        children = []
        for _ in range(nbChildren):
            p1 = choice(parents)
            if xdim < 2:
                children.append(p1)
            else:
                p2 = choice(parents)
                point = choice(list(range(xdim-1)))
                point += 1
                res = zeros(xdim)
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
        return children
        
    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        #to avoid having a child identical to one of the currentpopulation'''
        for i in range(self.dimension):
            if random() < self.mutationProb:
                res[i] = max(min(indiv[i] + gauss(0, self.mutationStdDev),self.U_bound[i]),self.L_bound[i])
        return res
    
    def produceOffspring(self):
        parents = self.select()
        es = self.elite_size
        self.pop = parents[:es]
        nbchildren = self.popsize - es
        if self.popsize - es <= 0:
            nbchildren = len(parents)
        for child in self.crossOver(parents, nbchildren ):
            self.pop.append(self.mutated(child))
        return self.pop
    
    def record_convergence(self):
        best_index   = np.argmax(self.fitnesses)
        fitness = np.max(self.fitnesses)
        indiv   = self.pop[best_index] 
        self.best_fitness.append(fitness)
        self.best_individual.append(indiv)
        return fitness,indiv
                 
    def optimize(self):
        self.initialPopulation = None
        self.initial_pop()
        for gen in range(self.max_generation):
            self.fitnesses = []
            for ind in range(self.popsize):
                self.fitnesses.append(self.objective(self.pop[ind]))
            best_fitnes,best_indiv = self.record_convergence()
            new_pop = self.produceOffspring()
        return self.best_fitness, self.best_individual
