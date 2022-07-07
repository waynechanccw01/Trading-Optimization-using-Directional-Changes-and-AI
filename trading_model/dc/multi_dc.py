import numpy as np
import pandas as pd
import os
import sys
import math
import random

# np.random.seed(20) 

class Multidc:

    def __init__(self, num_of_dc, budget, df): 
        self.num_of_dc = num_of_dc 
        self.budget = budget
        self.cash = budget
        self.PFL = 0
        self.data = df
        self.generate_threshold() 
        self.load_stock_data()

    def generate_threshold(self): # Randomly generate DC thresholds
        threshold = []
        if self.num_of_dc == 5:
            threshold.append(random.uniform(0, 1))
            threshold.append(random.uniform(0, 0.1))
            threshold.append(random.uniform(0, 0.01))
            threshold.append(random.uniform(0, 0.001))
            threshold.append(random.uniform(0, 0.0001))            
        self.dc_threshold = np.array(threshold)    
        
        FIXED = True
        if FIXED:
            self.dc_threshold = np.array([0.0001, 0.00013, 0.00015, 0.00018, 0.0002]) # same as research paper

    def load_stock_data(self):
        sys.path.append("..") #### https://kknews.cc/code/z5zkq4p.html
#        optimization = True
#        if optimization == True: # run in model.py
#            data_path = "./stock_data/" + str(self.stock_code) + ".csv"
#        else: # run this programme multi_dc.ipynb only
#            data_path = "../stock_data/" + str(self.stock_code) + ".csv"
#        self.data = pd.DataFrame(pd.read_csv(data_path))
        self.data['Trade_Price'] = self.data['Open'].shift(-1) # The open price of next day as trade price
        self.data.reset_index(drop = True, inplace = True)

    def trade_dc_pattern(self, Wj, theta, Q = 1, b1 = 0, b2 = 1, b3 = 1):
        
        # Initialization 
        if theta != []:
            self.dc_threshold = theta     
        if Wj == []:
            Wj = [1 for i in range(self.num_of_dc)] # Weight of each threshold = 1            
        event_dict = {} 
        for dc in self.dc_threshold:
            event_dict[dc] = [True]     
        ru = rd = 2
        ph = [self.data['Close'][0] for i in range(self.num_of_dc)] 
        pl = [self.data['Close'][0] for i in range(self.num_of_dc)]
        Ppeak = self.data['Close'][0] # overall Peak (not within each thresholds)
        Ptrough = self.data['Close'][0] # overall Ptrough (not within each thresholds)
        tdc_0 = [0 for i in range(self.num_of_dc)]
        tdc_1 = [0 for i in range(self.num_of_dc)]
        tos_0 = [0 for i in range(self.num_of_dc)]
        tos_1 = [0 for i in range(self.num_of_dc)]
        tU_0 = [0 for i in range(self.num_of_dc)]
        tU_1 = [0 for i in range(self.num_of_dc)]
        tD_0 = [0 for i in range(self.num_of_dc)]
        tD_1 = [0 for i in range(self.num_of_dc)] 

        for idx in range(1, len(self.data) - 1): 
            WB = 0
            WS = 0
            Nup = 0
            Ndown = 0
            Pc = self.data["Close"][idx]
            Trade_Price = self.data['Trade_Price'][idx]
            if self.data["Close"][idx] > Ppeak:
                Ppeak = self.data["Close"][idx]

            if self.data["Close"][idx] < Ptrough:
                Ptrough = self.data["Close"][idx]
                
            for dc_idx, dc in enumerate(self.dc_threshold):
                
                if event_dict[dc][-1]: # True -> Downward OS
                    if self.data['Close'][idx] <= (ph[dc_idx] * (1 - dc)):
                        event_dict[dc].append(False) # Confirmation point of downward DC
                        try: #error when t = 1
                            if event_dict[dc][-3] == False and event_dict[dc][-2] == True: # for the 
                                # case [False], [True], [False] -> 2 consecutive confirmation point
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass 
                        pl[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tD_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b1 
                        # "(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of downward OS 
                        tD_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b2 
                        # b1 and b2 defines the trading window
                    else:
                        event_dict[dc].append(event_dict[dc][-1]) # No DC pattern
                        if ph[dc_idx] < self.data['Close'][idx]:
                            ph[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == False and event_dict[dc][-2] == True: 
                                    #[False], [True], [True] -> now is right after the confirmation 
                                    # point, yet it is not low enough to be directional change, but 
                                    # lower than or same as the confirmation point
                                    tdc_0[dc_idx] = idx - 1 
                            except:
                                pass  
                else: # False -> Downward OS
                    if self.data['Close'][idx] >= (pl[dc_idx] * (1 + dc)):
                        event_dict[dc].append(True)  # Confirmation point of upward DC
                        try:
                            if event_dict[dc][-3] == True and event_dict[dc][-2] == False: 
                                # for the case [True], [False], [True] -> 2 consecutive confirmation point
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass
                        ph[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tU_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b1 
                        tU_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b2
                        # "(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of upward OS
                    else: 
                        event_dict[dc].append(event_dict[dc][-1])  # No DC pattern
                        if pl[dc_idx] > self.data['Close'][idx]:
                            pl[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == True and event_dict[dc][-2] == False:
                                    tdc_0[dc_idx] = idx - 1
                            except:
                                pass
                #trade
                if event_dict[dc][-1] == False and event_dict[dc][-2] == False: 
                    # if event is downward and not confirmation point
                    WB = WB + Wj[dc_idx] 
                    if tD_0[dc_idx] <= idx and idx <= tD_1[dc_idx]:
                        Ndown += 1
                    else:
                        Ndown -= 1

                if event_dict[dc][-1] == True and event_dict[dc][-2] == True: 
                    # if event is Upward and not confirmation point
                    WS = WS + Wj[dc_idx] 
                    if tU_0[dc_idx] <= idx and idx <= tU_1[dc_idx]:
                        Nup += 1
                    else:
                        Nup -= 1
            
            if WS > WB:        
                self.trade_action("sell", b3, Nup, Ppeak, Pc, Trade_Price, Q)
            elif WS < WB:
                self.trade_action("buy", b3, Ndown, Ptrough, Pc, Trade_Price, Q)

        Wealth = self.cash + self.PFL * Pc
        Return = 100 * (Wealth - self.budget) / self.budget
        
        return Return
        
    def trade_action(self, action, b3, N, P, Pc, Trade_Price, Q):
        N0 = self.num_of_dc
        if action == "sell":
            Nup = N
            if Nup > 0 and Pc >= P * b3:
                Qtrade = math.floor((1 + Nup / N0) * Q)
                if self.PFL > Qtrade:
                    self.cash = self.cash + Qtrade * Trade_Price
                    self.PFL = self.PFL - Qtrade
                else:
                    self.cash = self.cash + self.PFL * Trade_Price
                    self.PFL = 0
            
        elif action == "buy":
            Ndown = N
            if Ndown > 0 and Pc <= P + (P * (1 - b3)):
                Qtrade = math.floor((1 + Ndown / N0) * Q)
                if self.cash > Qtrade * Trade_Price:
                    self.cash = self.cash - Qtrade * Trade_Price
                    self.PFL = self.PFL + Qtrade
                else:
                    Qtrade_m = self.cash // Trade_Price 
                    self.cash = self.cash - Qtrade_m * Trade_Price
                    self.PFL = self.PFL + Qtrade_m
