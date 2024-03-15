#!/usr/bin/python
#-*-coding:GBK -*- 


from BPNN_connect import BPNN_connect
from BPNN_first import BPNN_first
from BPNN_XGBoost import BPNN_XGBoost 
from BPNN_catboost import BPNN_catboost    
from BPNN_RF import  BPNN_RF 
from BPNN_lightGBM import BPNN_lightGBM
from BPNN_SVM import BPNN_SVM 
from BPNN_BPNN import BPNN_BPNN


import os
import pandas as pd
import time

def get_all_excel(excel_read):

     time_start = time.time()
     
     excel_duplicates_data = BPNN_first(excel_read)
     excel_second_data = BPNN_connect(excel_duplicates_data)
     BPNN_catboost(excel_second_data)
     BPNN_RF(excel_second_data)
     BPNN_lightGBM(excel_second_data)
     BPNN_SVM (excel_second_data) 
     BPNN_XGBoost (excel_second_data)
     BPNN_BPNN (excel_second_data)

     time_end = time.time()
     print("The time  is:", time_end - time_start)


def main():
 
    ip='/.xlsx'
   

    get_all_excel(ip)


try:
    main()
except IndexError:
    print("Error")




 