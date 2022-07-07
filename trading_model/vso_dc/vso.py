#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : ${2020/05/01}
# @Author  : Ray Zhixi LI
# @FileName: ${vso}.py
# @Software: ${Virus Spread Optimizer}
# @Version: ${Beta V2}

import numpy as np
import sys
sys.path.append("..")
from vso_dc.host import Host
from vso_dc.virus import Virus
import time
import copy
from tqdm import tqdm 

class VSO:

    def __init__(self, dim = None, pop = 50, max_iter = 1000, 
                 bound = None, strategy = None, num_of_dc = None, budget = None, Qmax = None, 
                 imported_infection = False, show_train = True):
        self.dim = dim
        self.bound = bound
        self.strategy = strategy
        self.num_of_dc = num_of_dc
        self.budget = budget
        self.Qmax = Qmax
        self.imported_infection = imported_infection
        self.show_train = show_train
        if imported_infection:
           self.main_pop_size = int(pop * 0.6) # Pop size of main VSO colony
           self.import_pop_size = pop - self.main_pop_size # Pop size of imported infection colony
        else:
           self.main_pop_size = pop
        self.iterations = max_iter
        self.history = []
        self.gbest = {'host': None, 'fitness': float('inf'), 'solution': None} 
        # fitness = infinity becox the smaller the fitness, the better the solution
        self.import_gbest = {'host': None, 'fitness': float('inf'), 'solution': None}
        self.H = 1

    def fitness(self, ind, num_of_dc, strategy, budget):
        Q, b1, b2, b3 = ind[0], ind[1], ind[2], ind[3]
        W = [ind[i] for i in range(4, num_of_dc + 4)]
        theta = [ind[i] for i in range(num_of_dc + 4, len(ind))]
        strategy.budget = budget
        strategy.cash = budget
        strategy.PFL = 0
        trade = strategy.trade_dc_pattern(W, theta, Q, b1, b2, b3)
        return trade
    
    def run(self):
        hosts = []
        imported_hosts = []

        infect_probs = {
                'critical': {'infect_prob': 0.8, 'severe': 0.8, 'mild': 0.2},
                'severe': {'infect_prob': 0.2, 'severe':0.5,'mild': 0.5},
                'mild': {'infect_prob': 0.2, 'severe': 0.0, 'mild': 1.0},
        } #infection transformation matrix
        
        # Initialize hosts
        for _ in range(self.main_pop_size):
            #create population of healthy hosts
            hosts.append(Host(self.dim, self.bound, self.Qmax, self.num_of_dc))    

        if self.imported_infection:
            for _ in range(self.import_pop_size):
                _host = Host(self.dim, self.bound, self.Qmax, self.num_of_dc)
                imported_hosts.append(_host)
            self.im_infection_solutions = np.array([_host.virus.rna for _host in imported_hosts])
            print(self.im_infection_solutions)

        self.history = []
#        stuck_counter = 1
#        last_fitness = float('inf')
#        last_xx = np.array([0 for i in range(self.dim)])
        for iteration in range(self.iterations):
            pbest = {'host':None, 'fitness': float('inf'), 'solution': None}
            # Calculate fitness

            for idx, host in tqdm(enumerate(hosts)):
                solution = host.virus.rna
                host.fitness = - self.fitness(solution, self.num_of_dc, self.strategy, self.budget)
                if host.fitness < pbest['fitness']:
                    pbest['fitness'] = host.fitness
                    pbest['solution'] = host.virus.rna
                    pbest['host'] = host

            #if pbest['host'].type != 'critical':
            #    pbest['host'].infected = True
            #    pbest['host'].type = 'severe'
            
            # Selection
            if pbest['host'].fitness < self.gbest['fitness']:
                existing_critical_host = [h for h in hosts if h.type == 'critical']
                if len(existing_critical_host) > 0: 
                        existing_critical_host[0].type = 'severe' 
                self.gbest['fitness'] = pbest['fitness']
                self.gbest['solution'] = pbest['solution']
                self.gbest['host'] = pbest['host']
                self.gbest['host'].infected = True
                self.gbest['host'].type = 'critical'

            # Sorting: largest fitness -> smallest fitness (worst quality -> best quality)
            hosts = sorted(hosts, key = lambda host: host.fitness, reverse = True) 
            infected_hosts = [host for host in hosts if host.infected]
            
            # Recovery operation, 0.8 -> recPercent
            if len(infected_hosts) == int(self.main_pop_size):
                for host in infected_hosts[0:int(len(infected_hosts) * 0.8)]: 
                    host.recover()

            infected_hosts = [host for host in hosts if host.infected]
            healthy_hosts = [host for host in hosts if not host.infected]
            
            # Infection operation
            for infected_host in infected_hosts[::-1]: 
                 infect_type = infected_host.type
                 if len(healthy_hosts) >= self.H:
                     contacted_hosts = healthy_hosts[0 : self.H]
                     for healthy_host in contacted_hosts:
                         if np.random.rand() <= infect_probs[infect_type]['infect_prob']:
                            type_prob = np.random.rand()
                            if type_prob >= 0 and type_prob <= infect_probs[infect_type]['mild']:
                                healthy_host.infect(infected_host, 'mild')
                            else:
                                healthy_host.infect(infected_host, 'severe')
                            healthy_hosts.remove(healthy_host)

            ## Imported Infection
            if self.imported_infection:
                self.im_mutation(len(self.im_infection_solutions))
                self.im_crossover(len(self.im_infection_solutions))
                self.im_selection()
                if np.random.rand() <= (0.5 - 0.0) * iteration/self.iterations:
                    if self.import_gbest['fitness'] < self.gbest['fitness']:
                       self.gbest['host'].virus.rna =  self.import_gbest['solution']
                       self.gbest['host'].fitness = self.import_gbest['fitness']
                       self.gbest['solution'] =  self.import_gbest['solution']
                       
            ## Mutation operation
            for idx, host in enumerate(hosts):
                host.mutate(self.gbest['solution'])

            if self.show_train:
                print('Iteration:' + str(iteration))
                # print("Solution: ", np.where(self.gbest['solution'] < 0.5, 0, 1))
                # print("Solution_x: ", self.gbest['solution'])
                # print("Diff: ", self.gbest['solution']-last_xx)
                # last_xx = self.gbest['solution']
                print(str(self.gbest['solution']) + str(self.gbest['fitness']) + '\n')

            self.history.append(self.gbest['fitness'])


            # Dr Tam's suggestion
            # Tam_Method = False
            # if Tam_Method:
            #     if self.gbest['fitness'] == last_fitness:
            #         stuck_counter = stuck_counter + 1
            #         if stuck_counter == 20:
            #             # Get into local minima, restart host
            #             stuck_counter = 1
            #
            #             hosts = []
            #             imported_hosts = []
            #
            #             for _ in range(self.main_pop_size):
            #                 hosts.append(Host(self.dim, self.func, self.bound))
            #
            #             # This is just to initialize the solutions for the imported colony based on VSO
            #             if self.imported_infection:
            #                 for _ in range(self.import_pop_size):
            #                     _host = Host(self.dim, self.func, self.bound)
            #                     imported_hosts.append(_host)
            #                 self.im_infection_solutions = np.array([_host.virus.rna for _host in imported_hosts])
            #
            #     else:
            #         last_fitness = self.gbest['fitness']
            #         stuck_counter = 1
            #     print("Stuck Counter: ", str(stuck_counter))

        return self.gbest['solution'], self.gbest['fitness']

    def all_hisotry(self):
        return self.history


    ################# Imported colony based on DE  #####################################
    def im_x_to_y(self):
        self.Y_raw = []
        for k in range(len(self.im_infection_solutions)):
            self.Y_raw.append(self.func(self.im_infection_solutions[k]))
        self.Y = np.array(self.Y_raw)
        return self.Y

    def im_mutation(self, size_pop):
        X = self.im_infection_solutions
        random_idx = np.random.randint(0, size_pop, size=(size_pop, 3))
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        self.V = X[r1, :] + 0.5 * (X[r2, :] - X[r3, :])
        mask = np.random.uniform(low=self.bound[0], high=self.bound[1], size=(size_pop, self.dim))
        self.V = np.where(self.V < self.bound[0], mask, self.V)
        self.V = np.where(self.V > self.bound[1], mask, self.V)
        return self.V

    def im_crossover(self,size_pop):
        mask = np.random.rand(size_pop, self.dim) < 0.3
        self.U = np.where(mask, self.V, self.im_infection_solutions)
        return self.U

    def im_selection(self):
        X = self.im_infection_solutions.copy()
        f_X = self.im_x_to_y().copy()
        self.import_gbest['fitness'] = f_X.min()
        self.import_gbest['solution'] = X[self.Y.argmin()]
        self.im_infection_solutions = U = self.U
        f_U = self.im_x_to_y()
        self.im_infection_solutions = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.im_infection_solutions
