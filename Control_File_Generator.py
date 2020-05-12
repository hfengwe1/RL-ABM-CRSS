# -*- coding: utf-8 -*-
"""
# Control File Generator
Created on Fri Apr  3 18:31:25 2020

@author: feh219
"""
import os

working_dir  ="C:\\RegionalABM\\CRSS\\control"
os.chdir(working_dir)

US_States = ['WY', 'CO', 'NM', 'UT' , 'AZ UB', 'CA', 'NV']

Obj_names = []     # a list of object names
for s in US_States:
    obj = s + ' Schedules'
    Obj_names.append(obj)
    
#%% New Mexico
Groups = ['NewMexicoAgricultureArchToFarm', 'NewMexicoAgFarmToShip','NavajoIndianIrrigationProjectNIIPandExports']  # NM group names
NewMexicoAgricultureArchToFarm = ['HammondIrrigation', 'PLNewMexicoAgriculture','AnimasLaPlataNMAg',
                                  'NMNavajoSJClaims']
NewMexicoAgFarmToShip =['JewettValley','NavajoTributaryIrrigation', 'NMAgFruitland','NMHogbackCudei',
                        'TributaryIrrigation']
NIIP = ['NIIP']
Metrics = ['_Depletion Requested', '_Diversion Requested']
out_metrics = ['_Dep_req', '_Div_req']
slots = [NewMexicoAgricultureArchToFarm, NewMexicoAgFarmToShip, NIIP]
state = Obj_names[2]

with open("Ctl_CRSS_read_database.ctl", "w") as text_file:
    for g in range(len(Groups)):
        group = Groups[g]
        for s in slots[g]: 
            for m in range(len(Metrics)):
                line = state + '.' + group + '_' + s + Metrics[m] + ': file=C:\\RegionalABM\\CRSS\\CRSS_DB\\'
                line += group +'_' + s + out_metrics[m] + '.txt'+ ' ' + 'units=acre-ft/month scale=1.0 import=resize'
                print(line, file = text_file)
                

#%% ABM_to_CRSS.ctl                
Input_DB = 'Data_ABM'

out_metrics = ['_Dep_req', '_Div_req']

with open("Ctl_ABM_to_CRSS.ctl", "w") as text_file:
    for g in range(len(Groups)):
        group = Groups[g]
        for s in slots[g]: 
            for m in range(len(out_metrics)):
                line = Input_DB + '.' + group + '_' + s + out_metrics[m] + ': file=C:\\RegionalABM\\CRSS\\ABM_to_CRSS\\'
                line += group +'_' + s + out_metrics[m] + '.txt'+ ' ' + 'units=acre-ft/month scale=1.0 import=resize'
                print(line, file = text_file)
                
#%% CRSS_to_ABM.ctl
Metrics = ['.Depletion Requested', '.Diversion Requested']
out_metrics = ['_Dep_req', '_Div_req']
                
T_Metrics = ['Total Diversion', 'Total Depletion']
T_out_metrics = ['Total_Div', 'Total_Dep']
                
with open("Ctl_CRSS_to_ABM.ctl", "w+") as text_file:
    for g in range(len(Groups)):
        group = Groups[g]
        for m in range(len(T_Metrics)):
            line = group + '.' + T_Metrics[m] + ': file=C:\\RegionalABM\\CRSS\\CRSS_to_ABM\\'
            line += group +'_' + T_out_metrics[m] + '.txt'+ ' ' + 'units=acre-ft/month scale=1.0 import=resize'
#            print(line)
            print(line, file = text_file)   

        for s in slots[g]: 
            for m in range(len(Metrics)):
                line =   group + ':' + s + Metrics[m] + ': file=C:\\RegionalABM\\CRSS\\CRSS_to_ABM\\'
                line += group +'_' + s + out_metrics[m] + '.txt'+ ' ' + 'units=acre-ft/month scale=1.0 import=resize'
                print(line, file = text_file)
            
#%% Ctl_CRSS_Output.ctl            
            
Metrics = ['Total Diversion', 'Total Diversion Requested','Total Depletion Shortage']
out_metrics = ['Div', 'DivReq','DepShortage']
# Reservoir Elevation
Res = ['FlamingGorge', 'Crystal', 'Navajo', 'Mead','Powell']            
Res_Metrics = ['Pool Elevation']
Res_out_metrics = ['Elev']
                
with open("Ctl_CRSS_Output.ctl", "w+") as text_file:
    for g in range(len(Groups)):
        group = Groups[g]
        for m in range(len(Metrics)):
            line =   group + '.' + Metrics[m] + ': file=C:\\RegionalABM\\CRSS\\data\\'
            line += group +'_' + out_metrics[m] + '.txt'+ ' ' + 'units=acre-ft/month scale=1.0 import=resize'
            print(line, file = text_file)            
            
    for r in range(len(Res)):
        res = Res[r]
        for m in range(len(Res_Metrics)):
            line =   res + '.' + Res_Metrics[m] + ': file=C:\\RegionalABM\\CRSS\\data\\'
            line += res +'_' + Res_out_metrics[m] + '.txt'+ ' ' + 'units=ft scale=1.0 import=resize'
            print(line, file = text_file)                        

