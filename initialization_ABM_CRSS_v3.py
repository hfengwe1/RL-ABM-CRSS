# -*- coding: utf-8 -*-
"""
Initialize ABM model 
- interaction_info

Created on Mon Apr 20 17:40:52 2020
@author: feh219
"""
import os
import numpy as np
import shutil

working_dir  ="C:\\RegionalABM\\CRSS"
os.chdir(working_dir)
fname = 'interaction_info.txt'   

interaction_info = [2019, 0]        
np.savetxt('interaction_info.txt', interaction_info, fmt = '%i') 

R_to_A_path = os.path.join('CRSS_to_ABM')   # path: CRSS to ABM
A_to_R_path = os.path.join('ABM_to_CRSS')   # path: ABM to CRSS

initial_file_path = os.path.join(working_dir,'CRSS_DB','Initial_Files', 'All')
filenames = os.listdir(initial_file_path) #  a list of all the filenames in that folder
for f in filenames:
    print(f)
    shutil.copy2(os.path.join(initial_file_path, f), os.path.join(R_to_A_path, "QL_Working", f))
    
