import numpy as np
import pandas as pd
import os
import sys
import math
import random
import numpy as np 
import statistics as stat
import operator
from datetime import date
from datetime import timedelta
from dc.multi_dc import Multidc
from ga_dc.ga import GA
from vso_dc.vso import VSO

def roll_screen(stock, interval, algo, dollar): 
    training_t = 1
    ###### multidc Parameters
    if dollar == 'GBP':
        Qmax = 500
        budget = 500000        
    elif dollar == 'USD':
        Qmax = 685
        budget = 685425
    elif dollar == 'HKD':
        Qmax = 5331
        budget = 5330915
    num_of_dc = 5
    ###### GA parameters
    ga_iter = 100
    ga_pop = 50
    elitism_r = 5
    pc = 0.9
    pm = 0.1   
    ###### VSO parameters
    dim = 14
    bound = ((0, Qmax), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0.1),
            (0, 0.01), (0, 0.001), (0, 0.0001))   
    vso_iter = 100
    vso_pop = 50
    
    overall_return = ()
    
    for i in range(training_t):
        return_t = ()
        
        if interval == 'daily_closing':
            df = pd.DataFrame(pd.read_csv("./stock_data/" + stock + ".csv"))
            df = df.dropna()
            last_date = df['Date'][len(df) - 1]
            train_start = df['Date'][0]
            train_end = str(int(df['Date'][0].split('-')[0]) + 4) + '-01-01'
            test_start = train_end
            test_end = str(int(test_start.split('-')[0]) + 1) + '-01-01'

            train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]
            test = df[(df['Date'] >= test_start) & (df['Date'] < test_end)]
            print(train)
            print(test)

            strategy = Multidc(num_of_dc, budget, train)
            if algo == 'ga':
                dc_ga = GA(ga_iter, ga_pop, elitism_r, pc, pm, Qmax, num_of_dc, strategy, 
                           budget, show_train = False, train_detail = False)
                best, best_eval = dc_ga.genetic_algorithm()
                trained_para = best
            if algo == 'vso':
                dc_vso = VSO(dim = dim, bound = bound, strategy = strategy, num_of_dc = num_of_dc,
                          budget = budget, Qmax = Qmax, pop = vso_pop, max_iter = vso_iter, show_train = False)
                best, best_eval = dc_vso.run()
                trained_para = best

            multidc = Multidc(num_of_dc, budget, test)
            Return = multidc.trade_dc_pattern(trained_para[4:9], trained_para[9:14], trained_para[0], trained_para[1], trained_para[2], trained_para[3])
            return_t += (Return, )
            print(f'Return = {Return}')

            while test_end < last_date:
                train_start = str(int(train_start.split('-')[0]) + 1) + '-01-01'
                train_end = str(int(train_end.split('-')[0]) + 1) + '-01-01'
                test_start = train_end
                test_end = str(int(test_start.split('-')[0]) + 1) + '-01-01'
                train = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]
                test = df[(df['Date'] >= test_start) & (df['Date'] < test_end)]
                print(train)
                print(test)

                strategy = Multidc(num_of_dc, budget, train)
                if algo == 'ga':
                    dc_ga = GA(ga_iter, ga_pop, elitism_r, pc, pm, Qmax, num_of_dc, strategy, 
                               budget, show_train = False, train_detail = False)
                    best, best_eval = dc_ga.genetic_algorithm()
                    trained_para = best
                if algo == 'vso':
                    dc_vso = VSO(dim = dim, bound = bound, strategy = strategy, num_of_dc = num_of_dc,
                              budget = budget, Qmax = Qmax, pop = vso_pop, max_iter = vso_iter, 
                                 show_train = False)
                    best, best_eval = dc_vso.run()
                    trained_para = best

                multidc = Multidc(num_of_dc, budget, test)
                Return = multidc.trade_dc_pattern(trained_para[4:9], trained_para[9:14], 
                                                  trained_para[0], trained_para[1], trained_para[2], 
                                                  trained_para[3])
                print(f'Return = {Return}')
                return_t += (Return, )
        if overall_return == ():
            overall_return = tuple(0 for i in return_t) 
#         print(overall_return)
#         print(return_t)
        overall_return = tuple(map(operator.add, overall_return, return_t))
        print(overall_return)
    avg_return = tuple(i / training_t for i in overall_return)        
    return avg_return

stock = '0388.HK_HKEX_daily_closing_Jan2011-Dec2020'
avg_return = roll_screen(stock, 'daily_closing', 'ga', 'HKD')
mean = stat.mean(avg_return)
std = stat.stdev(avg_return)
Max = max(avg_return)
Min = min(avg_return)
CV = std / mean
print(f'mean = {mean}, std = {std}, max = {Max}, min = {Min}, CV = {CV}')
