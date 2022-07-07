#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : ${2020/05/01}
# @Author  : Ray Zhixi LI
# @FileName: ${host}.py
# @Software: ${Virus Spread Optimizer}
# @Version: ${Beta V2}

import numpy as np
from .virus import Virus

class Host: # define 1 host

    def __init__(self, solution_dim, solution_bound, Qmax, num_of_dc):
        self.solution_bound = solution_bound
        self.solution_dim = solution_dim
        self.Qmax = Qmax
        self.num_of_dc = num_of_dc
        self.infected = False
        self.virus = Virus(self.solution_dim, self.solution_bound, self.Qmax, self.num_of_dc) 
        # initialise the RNA
        self.fitness = None
        self.type = 'healthy'

    def recover(self):
        infect_type = self.type
        self.__init__(self.solution_dim, self.solution_bound, self.Qmax, self.num_of_dc) 
        # initialize the Virus
        if infect_type == 'severe':
           self.type = 'mild'
           self.infected = True
        elif infect_type == 'mild':
           self.infected = False
           self.type = 'healthy'

    def infect(self, infected_host, infect_type):
        if not self.infected: 
            self.infected = True
            self.type = infect_type
            if infect_type == 'severe': # if healthy host is infected to be severe host
                self.virus.rna = infected_host.virus.rna
            else: # infect_type == 'mild', if healthy host is infected to be mild host
                cross_points = np.random.randint(0, 2, size = self.solution_dim).astype(np.bool)
                for idx, point in enumerate(cross_points):
                    if point:
                        self.virus.rna[idx] = infected_host.virus.rna[idx]

    def mutate(self, gbest):
        if self.infected: # severe or mild host
           self.virus.update_mutation_strenth(self.type, gbest)
           self.virus.mutate(self.type)
        else: #healthy host
           self.virus.random_mutate(self.Qmax, self.num_of_dc)
