#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : ${2020/05/01}
# @Author  : Ray Zhixi LI
# @FileName: ${virus}.py
# @Software: ${Virus Spread Optimizer}
# @Version: ${Beta V2}

import numpy as np

class Virus: # define 1 virus

    def __init__(self, dim, bound, Qmax, num_of_dc): 
        self.dim = dim
        self.bound = bound
        self.mut_intensity_s = 1 / np.random.uniform() 
        self.mut_intensity_m = self.randgen_ind(Qmax, num_of_dc) / 10
        self.rna = self.randgen_ind(Qmax, num_of_dc)
    
    def randgen_ind(self, Qmax, num_of_dc):
        b1 = np.random.uniform(0, 1)
        b2 = np.random.uniform(b1, 1)
        ind_temp = [np.random.uniform(0, 1) for i in range(1 + num_of_dc)] # 1-> b3
        ind = [np.random.uniform(0, Qmax), b1, b2] + ind_temp
        if num_of_dc == 5:
            ind.append(np.random.uniform(0, 1))
            ind.append(np.random.uniform(0, 0.1))
            ind.append(np.random.uniform(0, 0.01))
            ind.append(np.random.uniform(0, 0.001))
            ind.append(np.random.uniform(0, 0.0001))
        return np.array(ind)

    def mutate(self, infect_type):
        if infect_type == 'critical':
            self.rna = self.rna
        elif infect_type == 'severe':
            self.rna = self.rna + np.random.normal(0, self.mut_intensity_s) * self.rna 
        elif infect_type == 'mild':
            self.rna = self.rna + self.mut_intensity_m

        # Handle the bound constraints
        for i in range(self.dim):
            if i == 2:
                self.rna[i] = np.clip(self.rna[i], self.rna[i - 1], self.bound[i][1])
            else:
                self.rna[i] = np.clip(self.rna[i], *self.bound[i]) 
        if self.rna[1] > self.rna[2]:
            print('Error')
            
    # 0.9 -> decay rate of Severe Hosts
    # 0.1 -> alpha, 2.0 -> y
    def update_mutation_strenth(self, infect_type, gbest): 
        if infect_type == 'severe':
            self.mut_intensity_s = self.mut_intensity_s * 0.9 
        elif infect_type == 'mild':
            self.mut_intensity_m = 0.1 * self.mut_intensity_m + 2.0 * np.random.rand() * (gbest - self.rna)

    def random_mutate(self, Qmax, num_of_dc):
        self.rna = self.randgen_ind(Qmax, num_of_dc)
