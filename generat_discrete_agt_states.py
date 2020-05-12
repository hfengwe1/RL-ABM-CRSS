# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:36:59 2020
Generate the discrete states 
@author: feh219
"""
# import necessary modules @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import os
import numpy as np
import pandas as pd
import math
# input
n_states = 21  # the number of states


A_to_R_path = os.path.join('ABM_to_CRSS')   # path: ABM to CRSS
R_to_A_path = os.path.join('CRSS_to_ABM')   # path: CRSS to ABM
CRSS_DB_path = os.path.join('CRSS_DB')      # path: CRSS DB folder

Groups = ['NewMexicoAgricultureArchToFarm', 'NewMexicoAgFarmToShip','NavajoIndianIrrigationProjectNIIPandExports']  # NM group names
NewMexicoAgricultureArchToFarm = ['HammondIrrigation', 'PLNewMexicoAgriculture','AnimasLaPlataNMAg',
                                  'NMNavajoSJClaims']
NewMexicoAgFarmToShip =['JewettValley','NavajoTributaryIrrigation', 'NMAgFruitland','NMHogbackCudei',
                        'TributaryIrrigation']
NIIP = ['NIIP']
Metrics = ['_Depletion Requested', '_Diversion Requested']
out_metrics = ['_Dep_req', '_Div_req']
slots = [NewMexicoAgricultureArchToFarm, NewMexicoAgFarmToShip, NIIP]

#%% Agent names @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
agt_names = []
for g in range(len(Groups)):
    for agt in slots[g]:
        agt_names.append(Groups[g] + '_' + agt)
        
n_agt = len(agt_names)                                    # number of agents

# Save the discrete states as a csv file
States_dict = dict()    # a dictionary of the agts' discrete states 
m_num = np.arange(73,85,1)
for i in range(n_agt):
    Div_fname = agt_names[i] + out_metrics[1] + '.txt'     # diversion/depletion             
    Div_path = os.path.join(CRSS_DB_path,Div_fname) # path: diversion of agent i, RiverWare_to_ABM folder
    
    with open(Div_path) as f_Div:
        f_Div.seek(0)
        Div = f_Div.readlines()[6:]    
     
# data cleaning (remove \n in each line)
#        Div_headers = Div_headers.replace('\n','', regex=True)
    Div = np.array(list(map(lambda s: s.strip(), Div)),dtype = float)           # change Div format to np array  
    Div_y=sum(Div[m_num])                     # clculate this year's annual div/dep
    Div_y0 = sum(Div[m_num-12])              # previous year's annual div/dep
    if Div_y0 > 0:
        bin_size = round(Div_y0/math.ceil(n_states/2), -int(math.log10(Div_y0/math.ceil(n_states/2))))  # round up (to the highest digit)
    else:
        bin_size = 0 

    States = bin_size*(np.arange(0,n_states,1))     # The states (diversion) 
    States_dict[agt_names[i]] = list(States)
    
# Write the States_dict to csv
csv_file = "States_dict.csv"
df_state = pd.DataFrame(States_dict)
df_state.to_csv (os.path.join(A_to_R_path,csv_file), index = False, header=True)