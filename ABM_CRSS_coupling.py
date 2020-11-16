# -*- coding: utf-8 -*-
"""
Generate new div/dep request from ABM modeling
- Q-learning
- epsilon-greedy algorithm 
- Colorado Rive Basin 
- Preceding factor: winter preceipitation for Upper Basin and Lake Mead water level for Lower Basin

Created on Tue Mar 31 11:46:25 2020
@author: feh219
"""

# import necessary modules @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import os
import numpy as np
import pandas as pd
#import math
import csv

working_dir  ="C:\\RegionalABM\\CRSS"  # need to change to local folder
os.chdir(working_dir)

Group_Agt_dir = os.path.join(working_dir,'CRSS_DB','Group_Agt') 

# Directories of ABM and RiverWare inputs/outputs and simulation iteration info
CRSS_DB_dir = os.path.join(working_dir,'CRSS_DB')      # path: CRSS DB folder
R_to_A_dir = os.path.join(working_dir,'CRSS_to_ABM')   # path: CRSS to ABM
A_to_R_dir = os.path.join(working_dir,'ABM_to_CRSS')   # path: ABM to CRSS
QL_dir = os.path.join(R_to_A_dir, 'QL_Working') 
out_metrics = ['_Dep_req', '_Div_req']                 # for output filenames

# input
n_states = 21  # the number of states
n_f = 2  # the prediction is higher (1) (than last year) or lower (0)
n_a = 2  # the number of actions
A_to_R_path = os.path.join('ABM_to_CRSS')   # path: ABM to CRSS
R_to_A_path = os.path.join('CRSS_to_ABM')   # path: CRSS to ABM
CRSS_DB_path = os.path.join('CRSS_DB', 'all_st')      # path: CRSS DB folder
Modle_path = os.path.join('model')  # the file path of the ABM parameter file


def Generate_Agt_Names(group_agt_file):  # agt_names are in "Group_Slot" format 
    
    Groups, group_agt_dict = Read_Group_Agent_Dict(group_agt_file) # read the csv file
    agt_names =[]                        # create a empty list
    for g in range(len(Groups)):
        group = Groups[g]                # the water user group
        slots = group_agt_dict[group]    # the agnets
        for agt in slots:
            if agt == '':                # the end of the agent list
                break
            agt_names.append(Groups[g] + '_' + agt)  # append agt name to the list     
                   
    return agt_names, Groups, group_agt_dict

def Read_Group_Agent_Dict(file):
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        group_agt_dict = dict((rows[0],rows[1:]) for rows in reader)
    Groups = list(group_agt_dict.keys())    
    return Groups, group_agt_dict

def ST_Agt_List(ST,cumsum):
    st_agt_list = []     # a list of object names
    for i in range(len(ST)):
        for s in np.arange(cumsum[i], cumsum[i+1]):           
            obj = ST[i]
            st_agt_list.append(obj)
    return st_agt_list

def generate_ABM_parameters(agt_names):  # generate a file that stores parameters for each agent
    n_agt = len(agt_names)
    param = ['mu', 'sigma','alpha','gamma','epsilon','regret','forget'] # ABM parameter names
    df = pd.DataFrame(index=param) # create an empty dataframe
    for i in range(n_agt):
        df[agt_names[i]] = [0,1,0.9,0.5,0.2,1,"TRUE"]
        
    csv_file = "LB_ABM_params_all_st.csv"
    df.to_csv(os.path.join(Modle_path,csv_file), index = param, header=True)  # save the results in .csv in the CRSS_DB//all_st folder         


def Read_Group_Agt_File(Group_Agt_dir,Index):
    if Index == "UB":
        UB_group_agt_file = os.path.join(Group_Agt_dir,'UB_ABM_Groups_and_Agents.csv')
        ub_groups, ub_group_agt_dict = Read_Group_Agent_Dict(UB_group_agt_file) # read the csv file
        Groups = ub_groups   # the water user groups of the agriculture water users
        group_agt_dict = ub_group_agt_dict  # combine two dictionaries
    elif Index == "LB":
        LB_group_agt_file = os.path.join(Group_Agt_dir,'LB_ABM_Groups_and_Agents_prior_ordered.csv')
        lb_groups, lb_group_agt_dict = Read_Group_Agent_Dict(LB_group_agt_file) # read the csv file
        Groups = lb_groups   # the water user groups of the agriculture water users
        group_agt_dict = lb_group_agt_dict  # combine two dictionaries
    elif Index == 'All':
        LB_group_agt_file = os.path.join(Group_Agt_dir,'LB_ABM_Groups_and_Agents_prior_ordered.csv')
        lb_groups, lb_group_agt_dict = Read_Group_Agent_Dict(LB_group_agt_file) # read the csv file   
        UB_group_agt_file = os.path.join(Group_Agt_dir,'UB_ABM_Groups_and_Agents.csv')
        ub_groups, ub_group_agt_dict = Read_Group_Agent_Dict(UB_group_agt_file) # read the csv file
        Groups = ub_groups +lb_groups   # the water user groups of the agriculture water users
        group_agt_dict = {**ub_group_agt_dict, **lb_group_agt_dict}  # combine two dictionaries
    else:
        print("Error: Index not in {'UB','LB','All'}")
        
    agt_name_w_space = []
    for g in range(len(Groups)):  # search for the names with space - the space will cause error in CRSS
        group = Groups[g]              # the water user group
        slots = group_agt_dict[group]  # the agnets
        for s in slots:  
            if (s.find(" ") != -1):
                agt_name_w_space.append(s)
    agt_names = []
    for g in range(len(Groups)):
        group = Groups[g]              # the water user group
        slots = group_agt_dict[group]  # the agnets
        for agt in slots:
            if agt == '':
                break
            agt_names.append(Groups[g] + '_' + agt)       
                   
    return agt_names, Groups, group_agt_dict, agt_name_w_space


def Div_to_State(div_y, agt_states, n_states):
    bin_size = agt_states[2] - agt_states[1]  # the bin size of the agent's discrete div level
    temp = int((div_y- agt_states[1])/bin_size)
        
    if temp <= (n_states-2) and div_y >= agt_states[1]:
        s_i = temp + 1      # the discrete state of last year 
    elif div_y < agt_states[1]:
        s_i = 0      # the discrete state of last year     
    else:   # temp > 19 
        s_i = n_states-1     # the discrete state of last year 
    return s_i

def Posterior(count, a_t_i, f_t, s_0_i, s_t_i,forget):

    if forget == 'True':
        alpha = (np.sum(count[f_t[0],a_t_i,s_0_i,:])-1) / np.sum(count[f_t[0],a_t_i,s_0_i,:])
    else:
        alpha = 1
        
    count[f_t[0],a_t_i,s_0_i,:] = alpha*count[f_t[0],a_t_i,s_0_i,:]  
    
    count[f_t[0],a_t_i,s_0_i,s_t_i] = count[f_t[0],a_t_i,s_0_i,s_t_i] + 1 # updating process
    count_shape = np.shape(count)    # the dimention of the count table (2,2,21,21)
    n_f = count_shape[0]
    n_a = count_shape[1]
    
    P_tran = []     
    for j in range(n_f):
        temp = []
        for k in range(n_a):
            p_tran = count[j,k,:,:]/np.sum(count[j,k,:,:], axis=1)[:,None] 
            temp.append(p_tran)
        P_tran.append(temp)
    P_tran = np.array(P_tran)  
  
    return P_tran, count
            
def Save_Array(filename, data,fmt):
    with open(filename, 'w+') as outfile:
        shape = data.shape
        outfile.write('# Array shape: {0}\n'.format(shape))

        if len(shape) == 3:
            for data_slice1 in data:
            
                np.savetxt(outfile, data_slice1, fmt=fmt)
                outfile.write('# New slice\n')
        elif len(shape) == 4:
                    
            for data_slice1 in data:    
                for data_slice2 in data_slice1: 
                    np.savetxt(outfile, data_slice2, fmt=fmt)            
                    outfile.write('# New slice\n')
        else:
            print('Error: data shape does not match')
    return
                                                          
def Update_CDF(sample, list_s_f, count_s_f): # update CDF count tables
    count_s_f[sample] += 1   # calculate the count
    n_count = np.sum(count_s_f,axis = None)
    p_s_f = count_s_f/n_count
    P_s_f = p_s_f.flatten()
    return P_s_f, count_s_f

                          
def Q_learning(file_path, ABM_params, agt_states, a_t, Div_y, Div_y0, f_t, agt):  
    n_a = 2   # number of actions
    n_f = 2   # number of preceding factor levels        
    n_states = len(agt_states)              # number of states
    bin_size = agt_states[2]-agt_states[1]     # calculate the bin_size
    mu = float(ABM_params[agt]['mu'])          # agent's div action distribution parameter (normal(mu, sigma))
    sigma= float(ABM_params[agt]['sigma'])     # agent's div action distribution parameter (normal(mu, sigma))

    alpha = float(ABM_params[agt]['alpha'])      # learning rate in Q-learning
    gamma = float(ABM_params[agt]['gamma'])      # discout rate in Q-learning
    epsilon = float(ABM_params[agt]['epsilon'])  # the parameter of the epsilon-greedy algorithm
    regret = float(ABM_params[agt]['regret'])    # a scalar for the penalty of unmet demand     
    forget = ABM_params[agt]['forget']          # Ture or False
    actions = (mu + abs(np.random.normal(0,sigma,size = 2))) * [-1,1] * bin_size   # the increase/decrease follows normal dist. (0,100)
    Q_path = os.path.join(file_path,'Q_Table_' + agt + '.txt')    
    data = np.loadtxt(Q_path)          # Q-table is a 3D array
    Q = data.reshape((n_f,n_states,n_a))  # reshape data to 3D array       

    count_path = os.path.join(file_path,'count_'+ agt + '.txt')            
    P_tran_path = os.path.join(file_path,'P_Tran_' + agt + '.txt')     
    data = np.loadtxt(count_path)  # Note that count table is a 4D array!
    count = data.reshape((n_f,n_a,n_states,n_states))  # reshape data to 4D array
    s_t_i = Div_to_State(Div_y, agt_states, n_states)  # calculate the index of the state an agnet is in at time t
    s_0_i = Div_to_State(Div_y0, agt_states, n_states) # calculate the index of the state an agnet is in at time t-1
   
    a_t_i = int(a_t > 0)
    P_tran, count = Posterior(count,a_t_i, f_t, s_0_i, s_t_i, forget)    # the frequentist method
    s_deviation = Div_y - (Div_y0 + a_t)
    
    if s_deviation >= 0: # if the deviation is greter than 0, no penalty
        s_deviation = 0
        
    penalty = regret * s_deviation    # the penalty    
    reward = Div_y + penalty          # the expected reward    

    E_Q_a = []    # expected future Q value
    for k in range(len(actions)):  # number of actions
        p_action = P_tran[f_t[1]][k][s_t_i]  # action(increase)              
        E_Q_a.append(np.sum(Q[f_t[1],:,k]*p_action)) # expected Q of to increase/decrease div
                
    E_future_Q = max(E_Q_a)   # the (expected Q value) of the optimal action at t+1
    Q[f_t[0],s_0_i,a_t_i] = Q[f_t[0],s_0_i,a_t_i] + alpha * (reward + gamma * E_future_Q - Q[f_t[0],s_0_i,a_t_i])

    if Q[f_t[1],s_t_i,0] <= Q[f_t[1],s_t_i,1]:   # if increasing div. has the higher Q value
        a_t_new = np.random.choice(actions,p=[epsilon, 1-epsilon])      
    else:        
        a_t_new = np.random.choice(actions,p=[1-epsilon, epsilon]) 

    if Div_y + a_t_new <= 0:    # normally, new div. = div. + a_t
        a_t_new = -Div_y        # but if the action is to decrease div to below 0, the action is - (Div_y)
    
    Save_Array(Q_path, Q, fmt='%10.3f')   # X is an array    
    P_tran = np.array(P_tran)  
    Save_Array(P_tran_path, P_tran, '%-7.6f')  
    Save_Array(count_path, count, '%-7.6f')    
        
    return a_t_new

def Generate_Agt_Group_Dict(group_agt_file):
   
    Groups, group_agt_dict = Read_Group_Agent_Dict(group_agt_file) # read the csv file
    agt_names = []
    slot_list = []  # a list of the slots 
    group_list = [] # a list of the goups where the slots are in
    for g in range(len(Groups)):
        group = Groups[g]              # the water user group
        slots = group_agt_dict[group]  # the agnets
        for agt in slots:
            if agt == '':
                break
            agt_names.append(Groups[g] + '_' + agt) 
            slot_list.append(agt)
            group_list.append(group)
    slot_list_no_duplicates = list(dict.fromkeys(slot_list))
    slot_dict = {}
    for slot in slot_list_no_duplicates:
        slot_temp = []
        for i in range(len(slot_list)):
            if slot_list[i] == slot:
                slot_temp.append(group_list[i])
        slot_dict[slot] = slot_temp
    
    return agt_names, Groups, group_agt_dict, slot_dict   

def convert_monthly_to_annual():        

    Data_DB_path = os.path.join(working_dir,'CRSS_DB\\\HistoricalData')      # path: CRSS DB folder
    LB_div_monthly_file = os.path.join(Data_DB_path, 'LB_historical_monthly_diversion.csv') # LB historical diversion file
    LB_div_monthly = pd.read_csv(LB_div_monthly_file, index_col = 0)       # monthly diversion
    
    LB_div_monthly.index = pd.DatetimeIndex(LB_div_monthly.index)  # convert the index to datetime format
    LB_div_annual = pd.DataFrame(index = np.arange(1971,2019,1))
    
    for col in LB_div_monthly.columns:
        LB_div_annual[col] = np.array(LB_div_monthly[col].resample('Y').sum())

    LB_div_annual_file = os.path.join(Data_DB_path, 'LB_historical_annual_diversion.csv') # LB historical diversion file
    LB_div_annual.to_csv(LB_div_annual_file, index=True)
    return

def Iteraction_Info():
    fname = 'interaction_info.txt'                 # file name: day and year number for RiverWare-River interaction
    with open(os.path.join(working_dir, fname)) as f:
        interaction = f.readlines()
    interaction = np.array(list(map(lambda s: s.strip(), interaction)),dtype = int)   # change Div format to np array  
    yr_idx = interaction[0]                        # current year of interaction (0:2018,1:2019,....,60:1960)
    yi = yr_idx - 2019                             # run index (+1 per year)
    
    m_start = interaction[1]                       # starting month of interaction  (73,...,85,...,335,336) # data starts in 2011/12, 73 is 2018/1 and 85 is 2019/1
    m_num = np.arange(m_start, m_start + 12)       # month number sequence
    return yi, m_start, m_num

def load_data(path):
    with open(path) as f_path:
        headers = f_path.readlines()[0:6]   # load headers of outflow file
        f_path.seek(0)                                   # reset searching point
        data = f_path.readlines()[6:]            # load outflow data
        data = np.array(list(map(lambda s: s.strip(), data)),dtype = float) # remove \n in the data, change fomrat to np array
    return data, headers

def Calculate_Agt_Annual_Div(m_num, Groups, group_agt_dict, R_to_A_dir, Index):
    df_Div_agt = pd.DataFrame()  # annual diversion at T
    
    for g in  range(len(Groups)):
        group = Groups[g]                                # Group name 
        slots = group_agt_dict[group]                    # slots in a group
        total_div_fname = group + '_Total_Div.txt'       # the total div. data filename
        Div_path = os.path.join(R_to_A_dir, total_div_fname) # the path of the total div. data
  
        Div, headers = load_data(Div_path)               # Load Div data as a NP array         
        Div_group_y = sum(Div[m_num])                    # calculate the annual div. of a group
        within_dist_fname = os.path.join(R_to_A_dir,'Div_Dist', 'Within_Group_Dist_' + group +'_Div_req.csv')
        Within_dist = pd.read_csv(within_dist_fname, index_col = False)  # load ABM parameters   
       
        for slot in slots:
            if slot == '':                               # the end of the slot list
                break
            else:
                agt = group + '_' + slot                 # agt's name: obj_slot
                div_agt = Div_group_y * Within_dist[agt] # calculate agent's diversion
                df_Div_agt[agt] = div_agt                # save agent's diversion in a dataframe
    
    div_fname = os.path.join(R_to_A_dir, 'Div_Dist', Index + '_Div_agts.csv')
    df_Div_agt.to_csv(div_fname, index = False)          # save the dataframe to .csv file
    return df_Div_agt   

def Calculate_Annual_Div_QL_UB(QL_agts, m_num, Index):
    df_Div_agt = Calculate_Agt_Annual_Div(m_num, Groups, group_agt_dict, R_to_A_dir, Index) # (actual) annual div at time T   
    QL_Div = pd.DataFrame()                              # annual div for QL calculation
    QL_Div_req = pd.DataFrame()                          # annual div requested for QL calculation
    for ql_agt in QL_agts:
        div_y = df_Div_agt[ql_agt][0]                   # agt's actual diversion at T
        agt_no_space = ql_agt.replace(" ", "")          # remove space in the agent names                                       
        div_req_fname = agt_no_space + out_metrics[1] + '.txt'          # agt's diversion requested            
        div_req_path = os.path.join(R_to_A_dir,div_req_fname) # path: diversion of agent i, RiverWare_to_ABM folder
        div_req, headers = load_data(div_req_path)   # load Div req data   
        div_req_y = sum(div_req[m_num])              # annual diversion requested
        QL_Div[ql_agt] = [div_y]                       # save to the dataframe        
        QL_Div_req[ql_agt] = [div_req_y]               # save to the dataframe
        
    QL_Div.to_csv(os.path.join(R_to_A_dir, 'Div_Dist', Index + '_QL_Div_y0.csv'), index = False)  
    return QL_Div, QL_Div_req
	
def Calculate_Annual_Div_QL_LB(QL_agts, multi_priority_agts, Groups, group_agt_dict, m_num, Index):
    df_Div_agt = Calculate_Agt_Annual_Div(m_num, Groups, group_agt_dict, R_to_A_dir, Index) # annual div and dep at time T    
    QL_Div = pd.DataFrame()      # annual div for QL calculation
    QL_Div_req = pd.DataFrame()  # annual div requested for QL calculation
    for ql_agt in QL_agts:  
        if ql_agt in multi_priority_agts: # for agents with multiple water rights
            div_agt = list()  # create a list
            div_req_agt = list()  # create a list
            
            for g in range(len(Groups)): 
                group = Groups[g]
                slots = group_agt_dict[group] # the agents of interest in that group    
                for slot in slots:
                    agt = group + '_' + slot   # agt's name: obj_slot
                    if ql_agt == 'MohaveValleyIDD' and slot in {'MohaveValleyIDD', 'MohaveValleyIDDAgPortion'}:
                        div_y = df_Div_agt[agt][0]  # this year's actual diversion
                        div_agt.append(div_y)       # store the div
                        
                        agt_no_space = agt.replace(" ", "")  # remove space in the agent names                                       
                        div_req_fname = agt_no_space + out_metrics[1] + '.txt' # agt's diversion requested            
                        div_req_path = os.path.join(R_to_A_dir,div_req_fname)  # path: diversion of agent i, RiverWare_to_ABM folder
                        div_req, headers = load_data(div_req_path)    # load Div req data   
                        div_req_y = (sum(div_req[m_num])) # annual diversion requested
                        div_req_agt.append(div_req_y) # store the div req
    
                    elif slot == ql_agt:      # others mp_agt have the same slot name
                        div_y = df_Div_agt[agt][0]  # this year's actual diversion
                        div_agt.append(div_y) # store the div
    
                        agt_no_space = agt.replace(" ", "")  # remove space in the agent names                                       
                        div_req_fname = agt_no_space + out_metrics[1] + '.txt'          # agt's diversion requested            
                        div_req_path = os.path.join(R_to_A_dir,div_req_fname) # path: diversion of agent i, RiverWare_to_ABM folder
                        div_req, headers = load_data(div_req_path)    # load Div req data   
                        div_req_y = (sum(div_req[m_num])) # annual diversion requested
                        div_req_agt.append(div_req_y) # store the div req
                        
            Div_agt = sum(np.array(div_agt))             # calculate QL_agt's annual div 
            QL_Div[ql_agt] = [Div_agt]                   # save to the dataframe
            
            Div_agt_req = sum(np.array(div_req_agt))     # calculate the QL_agt's annual div req sum
            QL_Div_req[ql_agt] = [Div_agt_req]           # save to the dataframe
            
        else:                            
            for g in range(len(Groups)): 
                group = Groups[g]
                slots = group_agt_dict[group] # the agents of interest in that group    
                for slot in slots:
                    if slot == '':  # skip loop
                        break
                    if slot == ql_agt: # find the aget's annual div value
                        agt = group + '_' + slot   # agt's name: obj_slot
                        agt_no_space = agt.replace(" ", "")  # remove space in the agent names
    
                        QL_Div[ql_agt] = df_Div_agt[agt]     # annual div at T
    
                        div_req_fname = agt_no_space + out_metrics[1] + '.txt'          # agt's diversion requested            
                        div_req_path = os.path.join(R_to_A_dir,div_req_fname) # path: diversion of agent i, RiverWare_to_ABM folder
                        div_req, headers = load_data(div_req_path)    # load Div req data            
                        Div_agt_req = (sum(div_req[m_num])) # annual diversion requested
                        QL_Div_req[ql_agt] = [Div_agt_req]       # save to the dataframe
             
        QL_Div.to_csv(os.path.join(R_to_A_dir, 'Div_Dist',Index + '_QL_Div_y0.csv'),index =False)  
    return QL_Div, QL_Div_req

def Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir):
    out_metrics = ['_Dep_req', '_Div_req']                 # for output filenames
    div_monthly = div_req_agt * monthly_dist[agt]          # next years monthly diversion requested
                    
    agt_no_space = agt.replace(" ", "")                    # remove space in the agent names                                       
    div_req_fname = agt_no_space + out_metrics[1] + '.txt' # agt's diversion requested            
    div_req_path = os.path.join(R_to_A_dir,div_req_fname)  # path: diversion of agent i, RiverWare_to_ABM folder

    div_req, headers = load_data(div_req_path)  
    update_month = m_num    # update the div_req for the next 12 months (m_start+12 is the next year's Jan)
    div_req[update_month] = div_monthly                    # update the monthly div requested array
    dep_req = div_req * Dep_Div_Ratio[agt][0]              # update the monthly dep requested array
    data = [dep_req, div_req]                              # create a data list 
    for m in range(len(out_metrics)):                      # write the updated dep/div req to ABM_to_CRSS folder 
        ABM_file_name = agt_no_space +  out_metrics[m] + '.txt'     # filename
        ABM_file_path = os.path.join(A_to_R_dir, ABM_file_name)
        
        with open(ABM_file_path, 'w+') as f:
            for L in range(len(headers)-1):                # write the header
                f.writelines(headers[L])
            comment = '# Series Slot:' + agt + out_metrics[m] +' [1 acre-ft/month]\n'
            f.writelines(comment)                          # write comments

            for j in range(len(data[m])):
                    f.writelines(str(data[m][j])+'\n')     # write data
    return

def Calculate_Within_Group_Dist(m_num, Index): # Index: "UB" or "LB
    group_agt_file = os.path.join(Group_Agt_dir, Index +'_Group_Agents_For_Within_Group.csv')  # the path of UB/LB agt names
    agt_names, Groups, group_agt_dict = Generate_Agt_Names(group_agt_file) # read group and agent names
    
    for metric in out_metrics:
        Monthly_Dist = pd.DataFrame(index = range(12))
        for g in range(len(Groups)):
            group = Groups[g]              # the water user group
            slots = group_agt_dict[group]  # the agnets
            group_annual = pd.DataFrame()    # A group's div/dep allocation to agents in fractions
            
            for slot in slots: 
                s_no_space = slot.replace(" ", "")   
                if slot == '':          ## '' signals the end of the agent list     
                    break    
                else:
                    agt_no_space = group + '_' + s_no_space   # agt's name: obj_slot
                    fname = agt_no_space + metric + '.txt'          # agt's diversion requested            
                    fpath = os.path.join(R_to_A_dir, fname) # path: diversion of agent i, RiverWare_to_ABM folder
                    Div_req, headers = load_data(fpath)              # load data and seperate data and headers  

                    annual_sum = sum(Div_req[m_num])
    #                print(agt, ':', annual_sum)
                    agt = group + '_' + slot
                    group_annual[agt] = [annual_sum] # save the agent's annual div/dep
                    if annual_sum > 0:
                        Div_req_dist_m = Div_req[m_num]/annual_sum   # clculate 2019's monthly div distribution
                    else:
                        Div_req_dist_m = np.ones([12])/12  # if sum <=0, then all 1
    
                    Monthly_Dist[agt] = Div_req_dist_m  # add an agent's mothly div distrubtion to the DB
            if np.array(group_annual.sum(axis = 1)) == 0:
                within_group_perc = np.zeros(len(group_annual.values[0]))  # if the sum = 0, create a lost of zeros
            else:
                within_group_perc = group_annual.values[0]/np.array(group_annual.sum(axis = 1))
            Within_Group = pd.DataFrame([within_group_perc], columns = group_annual.columns) # create a dataframe to store the within group dist results
            within_group_fname = 'Within_Group_Dist_' + group + metric +'.csv' 
            Within_Group.to_csv(os.path.join(R_to_A_dir,'Div_Dist', within_group_fname), index=False)
    return 

def Load_QL_Data(Index, m_num): 
    monthly_dist = pd.read_csv(os.path.join(R_to_A_dir, 'Div_Dist',Index + '_Monthly_Div_req.csv' ), index_col = False) # monthly diversion allocation distribution
    Dep_Div_Ratio = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist' ,Index + '_Dep_Div_Ratio.csv'), index_col = False)  # read dep/div ratios
    ABM_params_fname = Index + '_ABM_params_cal.csv'                      # file name: ABM paramaters
    ABM_params = pd.read_csv(os.path.join('model',ABM_params_fname), index_col = 0)  # load ABM parameters

    agt_names, Groups, group_agt_dict, agt_name_w_space = Read_Group_Agt_File(Group_Agt_dir, Index) 
    States_agt = pd.read_csv(os.path.join(CRSS_DB_dir , "Div_States", Index + "_discrete_states.csv"))  
    Calculate_Within_Group_Dist(m_num, Index)   # Calculate_Dep_Div_Ratio_LB(m_num)
    return monthly_dist, Dep_Div_Ratio, ABM_params, agt_names, Groups, group_agt_dict, States_agt

#%% Main
yi, m_start, m_num = Iteraction_Info()        
m_num_y0 = m_num - 12 
Index = "LB" 
multi_priority_agts = {'MohaveValleyIDD','UnitBIDD', 'CocopahIndRes','GilaMonsterFarms', 'NorthGilaValleyIDD','YumaCountyWUA'}

if yi == 0:
    monthly_dist, Dep_Div_Ratio, ABM_params, agt_names, Groups, group_agt_dict, States_agt = Load_QL_Data(Index, m_num) 
else: 
    monthly_dist, Dep_Div_Ratio, ABM_params, agt_names, Groups, group_agt_dict, States_agt = Load_QL_Data(Index, m_num_y0) 

QL_agts = list(States_agt.columns) # the agent list for QL

if yi == 0:  
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2017.csv'), index_col = False)              
    QL_Div = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2018.csv'), index_col = False)          
    QL_Div_req = QL_Div                                                       
elif yi == 1: # 2020/Jan
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2018.csv'), index_col = False)            
    QL_Div, QL_Div_req = Calculate_Annual_Div_QL_LB(QL_agts, multi_priority_agts, Groups, group_agt_dict, m_num_y0, Index)     

else:
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_y0.csv'), index_col = False)           
    QL_Div, QL_Div_req = Calculate_Annual_Div_QL_LB(QL_agts, multi_priority_agts, Groups, group_agt_dict, m_num_y0, Index)      

Preceding_fname = os.path.join(R_to_A_dir, 'Mead_Elev.txt')     
f_data, headers = load_data(Preceding_fname)                
fy_data = f_data[11::12]                                        
fy_data = np.insert(fy_data, 0, 1082.52, axis = 0)              
fy_data = np.insert(fy_data, 0, 1081.46, axis = 0)              
fy_data = np.nan_to_num(fy_data, nan = 0)                       

d_f = fy_data[1:] - fy_data[0:-1]     
f_t_all = [int(i) for i in (d_f >= 0)]  
f_t = f_t_all[yi:(yi+2)]              
agt_list =  list(States_agt.columns) 
QL_UpperBounds = pd.read_csv(os.path.join(QL_dir, 'QL_Div_2019_UpperBounds.csv'), index_col = False) 

for ql_agt in QL_agts:  
                  
    Div_y = QL_Div[ql_agt][0]                                 
    Div_req_y0 = QL_Div_req[ql_agt][0]                            

    Div_y0 = QL_Div_y0[ql_agt][0]                            
        
    a_t = Div_req_y0 - Div_y0                                          
                
    States = States_agt[ql_agt]                                     
    a_t_new = Q_learning(QL_dir, ABM_params, States, a_t, Div_y, Div_y0, f_t, ql_agt)  
    Div_req_update = Div_y + a_t_new                     
 
    if ql_agt in multi_priority_agts:        
        for g in range(len(Groups)):      
            group = Groups[g]
            slots = group_agt_dict[group]                                  
            for slot in slots:
                agt = group + '_' + slot                                 
                if slot == ql_agt:                                        
                    if group[-1] == '1':                                 
                        if Div_req_update > QL_UpperBounds[agt][0]:     
                            div_req_agt = QL_UpperBounds[agt][0]         
                                                                        
                            Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                            Div_req_update = Div_req_update - QL_UpperBounds[agt][0] 
                        else:
                            div_req_agt = Div_req_update                 
                                                                        
                            Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                            Div_req_update = Div_req_update - div_req_agt  
                            
                    elif group[-1] == '3'and slot != 'GilaMonsterFarms': 
                        div_req_agt = Div_req_update                       
                                                                           
                        Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                    elif group[-1] == '3'and slot == 'GilaMonsterFarms':                        
                        if Div_req_update > QL_UpperBounds[agt][0]:       
                            div_req_agt = QL_UpperBounds[agt][0]        
                                                                         
                            Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                            Div_req_update = Div_req_update - QL_UpperBounds[agt][0] 
                        else:
                            div_req_agt = Div_req_update          
                            Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                            Div_req_update = Div_req_update - div_req_agt 
                    else:                                                 
                        div_req_agt = Div_req_update  
                        Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                    break                                                 

                elif group[-1] == '4' and ql_agt == 'MohaveValleyIDD' and  slot == 'MohaveValleyIDDAgPortion':
                    div_req_agt = Div_req_update                         
                    Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir) 
                elif slot == '':
                    break                                               
    else:
        for g in range(len(Groups)):                                    
            group = Groups[g]
            slots = group_agt_dict[group]                           
            for slot in slots:
                agt = group + '_' + slot  
                if slot == ql_agt:                    
                    div_req_agt = Div_req_update                 
                    Update_Requested(agt, div_req_agt, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir)                     
                    break                                         
            if slot == ql_agt:    
                break                                              
                                
#%% Upper Basin
Index = "UB"
if yi == 0:
    monthly_dist, Dep_Div_Ratio, ABM_params, agt_names, Groups, group_agt_dict, States_agt = Load_QL_Data(Index, m_num) 
else: 
    monthly_dist, Dep_Div_Ratio, ABM_params, agt_names, Groups, group_agt_dict, States_agt = Load_QL_Data(Index, m_num_y0) 

QL_agts = agt_names 

STs = ['WY', 'UT1', 'UT2', 'UT3', 'NM', 'CO1', 'CO2', 'CO3', 'AZ'] 
no_state_group = pd.Series([0, 5, 11, 1 , 2, 7, 5, 15, 9, 3]) 
cumsum = list(no_state_group.cumsum())  
winter_precip_fname = 'PrismWinterPrecip_ST_NOAA_Future.csv'    
winter_precip_path = os.path.join(CRSS_DB_dir,'HistoricalData', winter_precip_fname)   
UB_precip = pd.read_csv(winter_precip_path,header = 0, index_col = 0)           

states_active = np.sum(States_agt,axis = 0)            
agt_inactive  = states_active[states_active == 0].index 
       
if yi == 0: 
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2017.csv'), index_col = False)               
    QL_Div = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2018.csv'), index_col = False)             
    QL_Div_req = QL_Div
elif yi == 1: 
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir,'Div_Dist', Index + '_QL_Div_2018.csv'), index_col = False)           
    QL_Div, QL_Div_req = Calculate_Annual_Div_QL_UB(QL_agts, m_num_y0, Index) 
else:
    QL_Div_y0 = pd.read_csv(os.path.join(R_to_A_dir, 'Div_Dist', Index + '_QL_Div_y0.csv'), index_col = False)         
    QL_Div, QL_Div_req = Calculate_Annual_Div_QL_UB(QL_agts, m_num_y0, Index)     

for st in range(len(STs)):
    ST = STs[st]
    ub_precip = np.array(UB_precip[ST])   
    d_f = ub_precip[1:] - ub_precip[0:-1]     
    f_t_all = [int(i) for i in (d_f>0)]     
    f_t = f_t_all[yi:(yi+2)]                

    st_agts = QL_agts[cumsum[st]:cumsum[st+1]] 

    for agt in st_agts:        
        if agt not in agt_inactive:  
            Div_y = QL_Div[agt][0]                                            
            Div_req_y0 = QL_Div_req[agt][0]                    
            if yi == 0:                                
                Div_y0 = QL_Div[agt][0]               
            else:
                Div_y0 = QL_Div_y0[agt][0]            
                
            a_t = Div_req_y0 - Div_y0            
            States = States_agt[agt]             
            a_t_new = Q_learning(QL_dir, ABM_params, States, a_t, Div_y, Div_y0, f_t, agt) 
            Div_req_update = Div_y + a_t_new          
            Update_Requested(agt, Div_req_update, monthly_dist, Dep_Div_Ratio, m_num, A_to_R_dir)                     
        

interaction_info = [yi + 2019 +1, m_start + 12]      
np.savetxt('interaction_info.txt', interaction_info, fmt = '%i') 


