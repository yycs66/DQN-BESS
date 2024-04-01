# This is a test dummy algorithm to get the opportunity cost curves in DA only
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import random
import math 
import os 
from Storge_parameters import battery_parameters


#calculate the opportunity cost for charge/discharge
def get_original_cost_offers(predict_prices):
    # battery parameters
    capacity=battery_parameters['capacity']
    ch_limit=battery_parameters['max_charge']
    dis_limit=battery_parameters['max_discharge']
    effcy=battery_parameters['efficiency']
    
    def scheduler(predict_prices):
    
        number_step =len(predict_prices)
        # [START solver]
        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            return
        # [END solver]

    #Variables: all are continous
        charge = [solver.NumVar(0.0, ch_limit, "c"+str(i)) for i in range(number_step)]
        discharge = [solver.NumVar(0, dis_limit,  "d"+str(i)) for i in range(number_step)]
        soc = [solver.NumVar(0.0, capacity, "b"+str(i)) for i in range(number_step+1)]
        soc[0]=0

    #Objective function
        solver.Minimize(
            sum(predict_prices[i]*(charge[i]-discharge[i]) for i in range(number_step)))
        for i in range(number_step):
            solver.Add(soc[i] +effcy*charge[i] -discharge[i]==soc[i+1])
    # create the constaints on soc
        solver.Solve()
        #print("Solution:")
        #print("The Storage's profit =", solver.Objective().Value())
        charge_list=[]
        discharge_list=[]
        for i in range(number_step):
            charge_list.append(charge[i].solution_value())
            discharge_list.append(discharge[i].solution_value())
        return charge_list,discharge_list
        
    [charge,discharge]=scheduler(predict_prices)
    #find the index of time period for charging/discharging
    index_ch=np.asarray(np.where(np.array(charge)>0)).flatten()
    ch_index =np.where(np.array(charge)>0)[0]
    index_dis=np.asarray(np.where(np.array(discharge)>0)).flatten()
    dis_index =np.where(np.array(discharge)>0)[0]

    idx0 =np.append(dis_index,ch_index)
    idx = np.sort(idx0)

    # create two list for charging/discharging opportunity costs
    oc_dis_list=[]
    oc_ch_list=[]
    
    offer = pd.DataFrame(None,index=range(len(predict_prices)), columns=['Time','charge cost','disch cost'])

#offer =offer.astype('Float64')
     
    for index, row in offer.iterrows():
        i =index
        row['Time'] =index
        if i in ch_index:
            # for hours in charge
            indx =np.where(ch_index==i)[0]
            j =indx.item()
            if i==0:
                oc_ch = min(predict_prices[1:index_dis[j]], effcy*predict_prices[index_dis[j]])
                oc_ch_list.append(oc_ch)  
                oc_dis = oc_ch+0.01
                print("oc_ch type is", type(oc_ch))
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
                oc_dis_list.append(oc_dis)  
            else:
                #oc_ch updates
                arr =np.delete(predict_prices[0:index_dis[j]], index_ch[j])
                min_oc_temp = arr.min()
                oc_ch = min(min_oc_temp, effcy*predict_prices[index_dis[j]])
                oc_ch_list.append(oc_ch)
                #oc_dis updates
                arr1 =predict_prices[0:index_ch[j]].min()
                arr2 =predict_prices[(index_ch[j]+1):index_dis[j]].min()
                oc_dis =(-predict_prices[index_ch[j]]+arr1+arr2)/effcy
                oc_dis_list.append(oc_dis)
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
        elif i in dis_index:
                # for scheduled discharge
                indx =np.where(dis_index==i)[0]
                j =indx.item()
                arr1 = predict_prices[(index_ch[j]+1):index_dis[j]].max()
                arr2 = predict_prices[(index_dis[j]+1):24].max()
                oc_ch =(-predict_prices[index_dis[j]] +arr1 +arr2)*effcy
                oc_ch_list.append(oc_ch)
                oc_dis = max(predict_prices[index_ch[j]]/effcy, predict_prices[(index_dis[j]+1):24].max())
                oc_dis_list.append(oc_dis)
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
        elif i< ch_index[0]:
                # opportunity cost for charging
                max_ch_temp =predict_prices[(i+1):ch_index[0]].max()
                oc_ch =max(max_ch_temp*effcy, predict_prices[index_ch[0]])
                oc_ch_list.append(oc_ch)
                row['charge cost'] =oc_ch
                
                #opportunity cosy for discharging
                if i==0:
                    # Hour 0, the oc_dis = to oc_ch+0.01
                    oc_dis = oc_ch+0.01
                    oc_dis_list.append(oc_dis)
                    row['disch cost'] =oc_dis
                elif i==1:  
                    oc_dis =predict_prices[0]/effcy
                    oc_dis_list.append(oc_dis)
                    row['disch cost'] =oc_dis
                elif i>1 :
                    oc_dis= predict_prices[0:i].min()/effcy
                    oc_dis_list.append(oc_dis)  
                    row['disch cost'] =oc_dis
        elif i>dis_index[-1]:
            
                if i != len(predict_prices)-1:
                    oc_ch =predict_prices[(i+1):len(predict_prices)].max()*effcy
                    oc_ch_list.append(oc_ch) 
                    oc_dis = min(predict_prices[index_ch[-1]], predict_prices[(index_dis[-1]+1):len(pred_predict_prices)].min()/effcy)
                    oc_dis_list.append(oc_dis)
                    row['charge cost'] =oc_ch
                    row['disch cost'] =oc_dis  
                else:
                    oc_ch=0
                    oc_ch_list.append(oc_ch) 
                    oc_dis =max(predict_prices[index_ch[-1]]/effcy, predict_prices[i])
                    oc_dis_list.append(oc_dis) 
                    row['charge cost'] =oc_ch
                    row['disch cost'] =oc_dis
        
            # for hours between
            #print("orignial i is ", i)
        elif i>index_dis[0] and i< index_ch[1]:
                # hours between last discharge and next charge period
                max_ch_temp =predict_prices[i:ch_index[1]].max()
                oc_ch =max(max_ch_temp*effcy, predict_prices[index_ch[1]])
                oc_ch_list.append(oc_ch)
                row['charge cost'] =oc_ch
                oc_dis= predict_prices[0:i].min()/effcy
                oc_dis_list.append(oc_dis)  
                row['disch cost'] =oc_dis
        else:
            for k in range(len(ch_index)):
                if i== index_ch[k]+1:
                    #print("i1 is between", i)
                    oc_ch = max(predict_prices[index_ch[k]], predict_prices[i]*effcy)
                    oc_ch_list.append(oc_ch)
                    row['charge cost'] =oc_ch
                    if i!= index_dis[k]-1:
                        oc_dis =min(predict_prices[index_dis[k]],predict_prices[(i+1):index_dis[k]].min()/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    else:
                        oc_dis = predict_prices[index_dis[k]]
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                elif i>index_ch[k]+1 and i< index_dis[k]:
                    max_ch_temp = predict_prices[(index_ch[k]+1):i]
                    max_ch_temp =max_ch_temp.max()
                    oc_ch = max(max_ch_temp*effcy, predict_prices[index_ch[k]])
                    oc_ch_list.append(oc_ch)
                    row['charge cost'] =oc_ch
                    if i< index_dis[k]-2:
                        oc_dis = min(predict_prices[index_dis[k]], predict_prices[(i+1):index_dis[k]].min()/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    elif i== index_dis[k]-2:
                        oc_dis = min(predict_prices[index_dis[k]], predict_prices[i+1]/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    elif i== index_dis[k]-1:
                        oc_dis = min(predict_prices[index_dis[k]],predict_prices[index_dis[k]]/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis                 
    quantity = pd.DataFrame(charge)
    #quantity = pd.DataFrame(charge)
    #ch_cost =pd.DataFrame(oc_ch_list,columns=['cost'])
    offer_ch =pd.concat([offer['Time'], offer['charge cost'], quantity], axis=1,ignore_index=True)
    offer_ch.columns =['Time','COST','MW']

    quan2 =pd.DataFrame(discharge)
    #dis_cost = pd.DataFrame(oc_dis_list)
    offer_dis=pd.concat([offer['Time'], offer['disch cost'], quantity], axis=1,ignore_index=True)
    offer_dis.columns= ['Time','COST','MW']
          

    return offer_ch, offer_dis

if __name__ == '__main__':
    # Add argument parser for three required input arguments
    #hourly price data for a year
    price_df=pd.read_csv('data/Prices.csv')
    predict_prices = price_df['LMP'].values
    predict_prices = predict_prices[0:24]
    
    # Make the offer curves and unload into arrays
    offer_ch, offer_dis = get_original_cost_offers(predict_prices)
    charge_mc = offer_ch['COST'].values
    charge_mq = offer_ch['MW'].values
    discharge_mc = offer_dis['COST'].values
    discharge_mq = offer_dis['MW'].values
    

    
