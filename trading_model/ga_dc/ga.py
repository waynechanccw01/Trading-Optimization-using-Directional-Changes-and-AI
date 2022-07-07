from tqdm import tqdm
from numpy.random import randint
from numpy.random import rand
import random
import sys
sys.path.append("..")
import copy
import dc.multi_dc as multi_dc
from dc.multi_dc import Multidc

class GA:
    def __init__(self, n_iter, n_pop, elitism_r, pc, pm, Qmax, num_of_dc, strategy, 
                 budget, show_train = True, train_detail = True):
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.elitism_r = elitism_r
        self.pc = pc
        self.pm = pm
        self.Qmax = Qmax
        self.num_of_dc = num_of_dc
        self.strategy = strategy
        self.budget = budget
        self.show_train = show_train
        self.train_detail = train_detail
        
    def fitness(self, ind, num_of_dc, strategy, budget):
        Q, b1, b2, b3 = ind[0], ind[1], ind[2], ind[3]
        W = [ind[i] for i in range(4, num_of_dc + 4)]
        theta = [ind[i] for i in range(num_of_dc + 4, len(ind))]
        strategy.budget = budget
        strategy.cash = budget
        strategy.PFL = 0
        trade = strategy.trade_dc_pattern(W, theta, Q, b1, b2, b3)
        return trade
    
    def create_ind(self, Qmax, N0): #N0 is the number of thresholds
        b1 = random.uniform(0, 1)
        b2 = random.uniform(b1, 1)
        ind_temp = [random.uniform(0, 1) for i in range(1 + N0)] # 1-> b3
        ind = [random.uniform(0, Qmax), b1, b2] + ind_temp
        return ind

    def create_ind_theta(self, N0): #N0 is the number of thresholds
        ind = []
        if N0 == 5:
            ind.append(random.uniform(0, 1))
            ind.append(random.uniform(0, 0.1))
            ind.append(random.uniform(0, 0.01))
            ind.append(random.uniform(0, 0.001))
            ind.append(random.uniform(0, 0.0001))
        return ind
    
    def elitism(self, pop, r):
        elites_pool = pop[len(pop) - r:]
        elites = [i for i in elites_pool]
        return elites
    
    # tournament selection
    def t_selection(self, pop, scores, k = 3):
        tournament = randint(0, len(pop), k) 
        selection_ix = tournament[0]
        for ix in tournament:   
            if scores[ix] > scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def uniform_crossover(self, p1, p2, pc, indpb = 0.5):
        if rand() < pc:
            for i in range(len(p1)):
                if rand() < indpb:
                    temp = p1[i]
                    p1[i] = p2[i]
                    p2[i] = temp
        return [p1, p2]

    def uniform_mutation(self, m, pr, Qmax, indpb = 0.5):
        if rand() < pr:
            for i in range(len(m)):
                if rand() < indpb:
                    if i == 0:
                        m[i] = random.uniform(0, Qmax)
                    elif i == 2:
                        m[i] = random.uniform(m[1], 1)
                    elif i == 9:
                        m[i] = random.uniform(0, 1)
                    elif i == 10:
                        m[i] = random.uniform(0, 0.1)
                    elif i == 11:
                        m[i] = random.uniform(0, 0.01)
                    elif i == 12:
                        m[i] = random.uniform(0, 0.001)
                    elif i == 13:
                        m[i] = random.uniform(0, 0.0001)
                    else:
                        m[i] = random.uniform(0, 1)
        return m
    
    def genetic_algorithm(self):

        pop = [self.create_ind(self.Qmax, self.num_of_dc) + 
               self.create_ind_theta(self.num_of_dc) for _ in range(self.n_pop)] # initial population 
        # keep track of best solution
        best, best_eval = 0, self.fitness(pop[0], self.num_of_dc, self.strategy, self.budget) 
        

        for gen in tqdm(range(self.n_iter)):  
            # evaluate all candidates in the population
            scores = [self.fitness(c, self.num_of_dc, self.strategy, self.budget) for c in pop]            
            pool = [(fitness, ind) for ind, fitness in zip(pop, scores)]
            pool = sorted(pool)
            pop = [i[1] for i in pool]
            scores = [i[0] for i in pool]

            if self.train_detail:
                print()
                print(f'sorted scores = {scores}')
                print()
                for i in pop:
                    print(f'sorted ind = {i}')

            if scores[-1] > best_eval: # check for new best solution 
                best, best_eval = pop[-1], scores[-1]
                best = copy.deepcopy(best)
                best_eval = copy.deepcopy(best_eval)

            elites = self.elitism(pop, self.elitism_r) # elitism
            elites = copy.deepcopy(elites) 

            #tournament selection
            selected = [self.t_selection(pop, scores) for _ in range(self.n_pop - self.elitism_r)]  

            offsprings = [] # create the next generation
            for i in range(1, self.n_pop - self.elitism_r, 2):
                p1, p2 = selected[i-1], selected[i]
                for c in self.uniform_crossover(p1, p2, self.pc):
                    self.uniform_mutation(c, self.pm, self.Qmax)
                    offsprings.append(c) # store for next generation

            pop = elites + offsprings # replace population
        if self.show_train:
            print(f'best individual = {best}, score = {best_eval}')
        return best, best_eval
