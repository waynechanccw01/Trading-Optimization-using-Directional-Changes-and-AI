import numpy as np
import pandas as pd
import os
import sys
import math
import random

np.random.seed(20)

class Singledc: #Return the profit of a number of single_dc thresholds

    def __init__(self, num_of_singledc, stock_code, budget, dollar, generate_mode): 
        self.num_of_singledc = num_of_singledc #number of single DC thresholds
        self.dc_interval = 1
        self.stock_code = stock_code
        self.budget = budget 
        self.cash = [budget for i in range(self.num_of_singledc)] #the amount of cash we are currently holding
        self.PFL = [0 for i in range(self.num_of_singledc)] #the amount/quantity for the stock we are currently holding
        self.dollar = dollar #currency

        self.generate_threshold(generate_mode) # Randomly generate DC thresholds
        self.load_stock_data()
        self.trade_dc_pattern()

    def generate_threshold(self, generate_mode):
        single_range0 = 0
        single_range1 = 1 
        mode = generate_mode
        if mode == "step up":
            threshold_interval = [i / 100 for i in range(self.num_of_singledc + 1)]
            threshold = []
            for i in range(1, self.num_of_singledc + 1):
                threshold.append(random.uniform(threshold_interval[i-1], threshold_interval[i]))
            self.dc_threshold = np.array(threshold)    
        
        if mode == "single range":
            threshold = []
            for i in range(self.num_of_singledc):
                threshold.append(random.uniform(single_range0 / 100, single_range1 / 100))
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

    def trade_dc_pattern(self):
        # Initialization
        event_dict = {}  
        for dc in self.dc_threshold:
            event_dict[dc] = [True] #True -> Upward trend, False -> Downward trend
        b1 = 0 #b1 and b2 defines the trading window
        b2 = 1
        b3 = 1 #we want buy at the price close to the price trough P_trough and sell at the price close to the peak price P_peak 
        Q = 1 #Q controls the trading quantity
        ru = rd = 2 #predicts the length of OS 
        ph = [self.data['Close'][0] for i in range(self.num_of_singledc)] #the highest price in the current upward trend
        pl = [self.data['Close'][0] for i in range(self.num_of_singledc)] #the lowest price in the current downward trend
        Ppeak = self.data['Close'][0] #the highest recorded price 
        Ptrough = self.data['Close'][0] #the lowest recorded price
        tdc_0 = [0 for i in range(self.num_of_singledc)] #directional change 
        tdc_1 = [0 for i in range(self.num_of_singledc)]
        tos_0 = [0 for i in range(self.num_of_singledc)] #overshoot event
        tos_1 = [0 for i in range(self.num_of_singledc)]
        tU_0 = [0 for i in range(self.num_of_singledc)] #trading window
        tU_1 = [0 for i in range(self.num_of_singledc)]
        tD_0 = [0 for i in range(self.num_of_singledc)]
        tD_1 = [0 for i in range(self.num_of_singledc)] 

        for idx in range(1, len(self.data)): 
            WB = [0 for i in range(self.num_of_singledc)] #Weight of Buy
            WS = [0 for i in range(self.num_of_singledc)] #Weight of Sell
            Nup = [0 for i in range(self.num_of_singledc)] #the number of thresholds that recommending a sell action
            Ndown = [0 for i in range(self.num_of_singledc)] #the number of thresholds that recommending a buy action
            Pc = self.data["Close"][idx] #current price 
            Trade_Price = self.data['Trade_Price'][idx] #current trade price
            
            if self.data["Close"][idx] > Ppeak:
                Ppeak = self.data["Close"][idx]
            elif self.data["Close"][idx] < Ptrough:
                Ptrough = self.data["Close"][idx]
                
            for dc_idx, dc in enumerate(self.dc_threshold): 
                if event_dict[dc][-1]: 
                    if self.data['Close'][idx] <= (ph[dc_idx] * (1 - dc)):
                        event_dict[dc].append(False) # Confirmation point of downward DC
                        try: 
                            if event_dict[dc][-3] == False and event_dict[dc][-2] == True: #for the case [False], [True], [False] -> 2 consecutive confirmation points
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass 
                        pl[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tD_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b1 #"(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of downward OS 
                        tD_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * rd * b2 #b1 and b2 defines the trading window
                    else:
                        event_dict[dc].append(event_dict[dc][-1]) # No DC 
                        if ph[dc_idx] < self.data['Close'][idx]:
                            ph[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == False and event_dict[dc][-2] == True: #[False], [True], [True] -> now is right after the confirmation point, yet it is not low enough to be directional change, but lower than or same as the confirmation point
                                    tdc_0[dc_idx] = idx - 1 
                            except:
                                pass  
                else: 
                    if self.data['Close'][idx] >= (pl[dc_idx] * (1 + dc)):
                        event_dict[dc].append(True)  # Confirmation point of upward DC
                        try:
                            if event_dict[dc][-3] == True and event_dict[dc][-2] == False: #for the case [True], [False], [True] -> 2 consecutive confirmation points
                                tdc_0[dc_idx] = idx - 1
                        except:
                            pass
                        ph[dc_idx] = self.data['Close'][idx]
                        tdc_1[dc_idx] = idx
                        tos_0[dc_idx] = idx + 1
                        tU_0[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b1 #"(tdc_1[dc_idx] - tdc_0[dc_idx]) * rd" predicts the length of upward OS 
                        tU_1[dc_idx] = tdc_1[dc_idx] + 1 + (tdc_1[dc_idx] - tdc_0[dc_idx]) * ru * b2 #b1 and b2 defines the trading window
                    else: 
                        event_dict[dc].append(event_dict[dc][-1])  # No DC
                        if pl[dc_idx] > self.data['Close'][idx]:
                            pl[dc_idx] = self.data['Close'][idx]
                            tdc_0[dc_idx] = idx
                            tos_1[dc_idx] = idx - 1
                        else:
                            try:
                                if event_dict[dc][-3] == True and event_dict[dc][-2] == False: #[True], [False], [False] -> the current time point is right after the confirmation point, yet it is not low enough to be directional change, but lower than or same as the confirmation point
                                    tdc_0[dc_idx] = idx - 1
                            except:
                                pass
                #trade
                if event_dict[dc][-1] == False and event_dict[dc][-2] == False: #if event is downward trend and the current time point is not confirmation point
                    WB[dc_idx] = WB[dc_idx] + 1
                    if tD_0[dc_idx] <= idx and idx <= tD_1[dc_idx]: #if the current time point is within the trading window
                        Ndown[dc_idx] += 1
                    else:
                        Ndown[dc_idx] -= 1

                if event_dict[dc][-1] == True and event_dict[dc][-2] == True: #if event is upward trend and the current time point is not confirmation point
                    WS[dc_idx] = WS[dc_idx] + 1
                    if tU_0[dc_idx] <= idx and idx <= tU_1[dc_idx]: #if the current time point is within the trading window
                        Nup[dc_idx] += 1
                    else:
                        Nup[dc_idx] -= 1
                        
                if WS[dc_idx] > WB[dc_idx]:
#                     print(f"WS > WB, Nup = {Nup}") 
                    self.trade_action("sell", b3, Nup[dc_idx], Ppeak, Pc, Trade_Price, Q, dc_idx) #sell
                elif WS[dc_idx] < WB[dc_idx]:
#                     print(f"WB > WS, Ndown = {Ndown}")
                    self.trade_action("buy", b3, Ndown[dc_idx], Ptrough, Pc, Trade_Price, Q, dc_idx) #buy

        Wealth = []
        Return = []
        for i in range(len(self.cash)):
            final_Wealth = self.cash[i] + self.PFL[i] * Pc
            Wealth.append(final_Wealth)
            Return.append(100 * (final_Wealth - self.budget) / self.budget) #calculate return
        print()
        for i in range(len(Return)):
            print(f"{self.dc_threshold[i]}: ")
            print("Wealth = {:0.4f} ".format(Wealth[i]) + self.dollar + ", Return = {:0.4f}".format(Return[i]) + "%")
                
    def trade_action(self, action, b3, N, P, Pc, Trade_Price, Q, dc_idx): #performs the buy and sell actions
        if action == "sell":
            Nup = N
            if Nup > 0 and Pc >= P * b3: #sell
                Qtrade = Q
                if self.PFL[dc_idx] > Qtrade:
                    self.cash[dc_idx] = self.cash[dc_idx] + Qtrade * Trade_Price
                    self.PFL[dc_idx] = self.PFL[dc_idx] - Qtrade
                else:
                    self.cash[dc_idx] = self.cash[dc_idx] + self.PFL[dc_idx] * Trade_Price
                    self.PFL[dc_idx] = 0
            else:
                pass
            
        elif action == "buy":
            Ndown = N
            if Ndown > 0 and Pc <= P + (P * (1 - b3)): #buy
                Qtrade = Q
                if self.cash[dc_idx] > Qtrade * Trade_Price:
                    self.cash[dc_idx] = self.cash[dc_idx] - Qtrade * Trade_Price
                    self.PFL[dc_idx] = self.PFL[dc_idx] + Qtrade
                else:
                    Qtrade_m = self.cash[dc_idx] // Trade_Price 
                    self.cash[dc_idx] = self.cash[dc_idx] - Qtrade_m * Trade_Price
                    self.PFL[dc_idx] = self.PFL[dc_idx] + Qtrade_m
            else:
                pass
        
VOO = Singledc(5, "VOO_USD", 100000, "USD", "single range") 
