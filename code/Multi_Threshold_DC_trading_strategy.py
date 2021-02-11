import numpy as np
import pandas as pd
import os
import sys
import math
import random

np.random.seed(20)

class Multidc:

    def __init__(self, num_of_dc, stock_code, budget, dollar): 
        self.num_of_dc = num_of_dc # number of DC thresholds
        self.dc_interval = 1
        self.stock_code = stock_code
        self.budget = budget
        self.cash = budget # the amount of cash we are currently holding
        self.PFL = 0 # the amount/quantity for the stock we are currently holding
        self.dollar = dollar # currency

        self.generate_threshold() # Randomly generate DC thresholds
        self.load_stock_data()
        self.trade_dc_pattern() 

    def generate_threshold(self):

        threshold_interval = [i / 100 for i in range(self.num_of_dc + 1)]
        threshold = []
        for i in range(1, self.num_of_dc + 1):
            threshold.append(random.uniform(threshold_interval[i-1], threshold_interval[i]))
        self.dc_threshold = np.array(threshold)    

        FIXED = True
        if FIXED:
            self.dc_threshold = np.array([0.0001, 0.00013, 0.00015, 0.00018, 0.0002])
        print(self.dc_threshold)
        print("threshold value: ",end='')
        for i in self.dc_threshold:
            print("{:0.4f}%, ".format(i * 100),end='')
        print()

    def load_stock_data(self):
        sys.path.append("..") 
        data_path = "./stock_data/" + str(self.stock_code) + ".csv"
        self.data = pd.DataFrame(pd.read_csv(data_path))
        self.data['Trade_Price'] = self.data['Open'].shift(-1) # The open price of next day as trade price
        self.data = self.data[::self.dc_interval] # Change the observation time interval. 
        self.data.reset_index(drop = True, inplace = True)
#         print(self.data)

    def trade_dc_pattern(self):
        # Initialization
        event_dict = {} 
        for dc in self.dc_threshold:
            event_dict[dc] = [True] # True -> Upward trend, False -> Downward trend
        b1 = 0 # b1 and b2 defines the trading window
        b2 = 1
        b3 = 1 # we want buy at the price close to the price trough P_trough and sell at the price close to the peak price P_peak 
        Q = 1 # Q controls the trading quantity
        ru = rd = 2 # predicts the length of OS 
        Wj = [1 for i in range(self.num_of_dc)] # Weight of each threshold = 1
        ph = [self.data['Close'][0] for i in range(self.num_of_dc)] # the highest price in the current upward trend
        pl = [self.data['Close'][0] for i in range(self.num_of_dc)] # the lowest price in the current downward trend
        Ppeak = self.data['Close'][0] # the highest recorded price  
        Ptrough = self.data['Close'][0] # the lowest recorded price
        tdc_0 = [0 for i in range(self.num_of_dc)] # directional change 
        tdc_1 = [0 for i in range(self.num_of_dc)]
        tos_0 = [0 for i in range(self.num_of_dc)] # overshoot event
        tos_1 = [0 for i in range(self.num_of_dc)]
        tU_0 = [0 for i in range(self.num_of_dc)] # trading window
        tU_1 = [0 for i in range(self.num_of_dc)]
        tD_0 = [0 for i in range(self.num_of_dc)]
        tD_1 = [0 for i in range(self.num_of_dc)] 

        for idx in range(1, len(self.data)): 
            WB = 0 # Weight of Buy
            WS = 0 # Weight of Sell
            Nup = 0 # the number of thresholds that recommending a sell action
            Ndown = 0 # the number of thresholds that recommending a buy action
            Pc = self.data["Close"][idx] # current price 
            Trade_Price = self.data['Trade_Price'][idx] # current trade price
            
            if self.data["Close"][idx] > Ppeak:
                Ppeak = self.data["Close"][idx]
            elif self.data["Close"][idx] < Ptrough:
                Ptrough = self.data["Close"][idx]
                
            for dc_idx, dc in enumerate(self.dc_threshold): 
                if event_dict[dc][-1]: 
                    if self.data['Close'][idx] <= (ph[dc_idx] * (1 - dc)):
                        event_dict[dc].append(False) # Confirmation point of downward DC
                        try: #error when t = 1
                            if event_dict[dc][-3] == False and event_dict[dc][-2] == True: # for the case [False], [True], [False] -> 2 consecutive confirmation points
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass 
                        pl[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tD_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b1 # "(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of downward OS 
                        tD_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b2 # b1 and b2 defines the trading window
                    else:
                        event_dict[dc].append(event_dict[dc][-1]) # No DC 
                        if ph[dc_idx] < self.data['Close'][idx]:
                            ph[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == False and event_dict[dc][-2] == True: # [False], [True], [True] -> the current time point is right after the confirmation point, yet its price is not low enough to be directional change, but lower than or same as the confirmation point
                                    tdc_0[dc_idx] = idx - 1 
                            except:
                                pass  
                else: # False -> Downward trend
                    if self.data['Close'][idx] >= (pl[dc_idx] * (1 + dc)):
                        event_dict[dc].append(True)  # Confirmation point of upward DC
                        try:
                            if event_dict[dc][-3] == True and event_dict[dc][-2] == False: # for the case [True], [False], [True] -> 2 consecutive confirmation points
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass
                        ph[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tU_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b1 # "(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of upward OS 
                        tU_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b2 # b1 and b2 defines the trading window
                    else: 
                        event_dict[dc].append(event_dict[dc][-1])  # No DC 
                        if pl[dc_idx] > self.data['Close'][idx]:
                            pl[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == True and event_dict[dc][-2] == False: # [True], [False], [False] -> the current time point is right after the confirmation point, yet its price is not high enough to be directional change, but higher than or same as the confirmation point
                                    tdc_0[dc_idx] = idx - 1
                            except:
                                pass
                # trade
                if event_dict[dc][-1] == False and event_dict[dc][-2] == False: # if event is downward trend and the current time point is not confirmation point
                    WB = WB + Wj[dc_idx] 
                    if tD_0[dc_idx] <= idx and idx <= tD_1[dc_idx]: # if the current time point is within the trading window
                        Ndown += 1
                    else:
                        Ndown -= 1
                        
                if event_dict[dc][-1] == True and event_dict[dc][-2] == True: # if event is upward trend and the current time point is not confirmation point
                    WS = WS + Wj[dc_idx] 
                    if tU_0[dc_idx] <= idx and idx <= tU_1[dc_idx]: # if the current time point is within the trading window
                        Nup += 1
                    else:
                        Nup -= 1
                        
            if WS > WB:
                self.trade_action("sell", b3, Nup, Ppeak, Pc, Trade_Price, Q) # sell
            elif WS < WB:
                self.trade_action("buy", b3, Ndown, Ptrough, Pc, Trade_Price, Q) # buy

        Wealth = self.cash + self.PFL * Pc
        Return = 100 * (Wealth - self.budget) / self.budget # calculate return
        print(Return)
        print("Wealth = {:0.4f} ".format(Wealth) + self.dollar + ", Return = {:0.4f}".format(Return) + "%")
        
    def trade_action(self, action, b3, N, P, Pc, Trade_Price, Q): # performs the buy and sell actions
        N0 = self.num_of_dc
        if action == "sell":
            Nup = N
            if Nup > 0 and Pc >= P * b3: # sell
                Qtrade = math.floor((1 + Nup / N0) * Q)
                if self.PFL > Qtrade:
                    self.cash = self.cash + Qtrade * Trade_Price
                    self.PFL = self.PFL - Qtrade
                else:
                    self.cash = self.cash + self.PFL * Trade_Price
                    self.PFL = 0
            else:
                pass #hold
            
        elif action == "buy":
            Ndown = N
            if Ndown > 0 and Pc <= P + (P * (1 - b3)): # buy
                Qtrade = math.floor((1 + Ndown / N0) * Q)
                if self.cash > Qtrade * Trade_Price:
                    self.cash = self.cash - Qtrade * Trade_Price
                    self.PFL = self.PFL + Qtrade
                else:
                    Qtrade_m = self.cash // Trade_Price 
                    self.cash = self.cash - Qtrade_m * Trade_Price
                    self.PFL = self.PFL + Qtrade_m
            else:
                pass #hold
        
HSBC = Multidc(5, "0005.HK_HSBC_HKD", 775335, "HKD") 
