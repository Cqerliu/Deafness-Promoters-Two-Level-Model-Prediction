#!/usr/bin/python
#-*-coding:GBK -*- 


import pandas as pd

def BPNN_connect(excel_duplicates_data):

    excel_second_data = 'data.csv'
    data = pd.read_csv(excel_duplicates_data)
    data = data.drop_duplicates()
    data.to_csv(excel_second_data, index=False)   

    return   excel_second_data





