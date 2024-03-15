#!/usr/bin/python
# -*- coding: gbk -*-

import matplotlib.pyplot as plt
from sklearn import neural_network ,preprocessing
from sklearn.model_selection import StratifiedShuffleSplit # 分层随机分割交叉验证器
from sklearn.metrics import roc_curve,auc,confusion_matrix
from sklearn.pipeline import Pipeline
from scipy import interpolate
from openpyxl import Workbook #openpyxl最好用的python操作excel表格库
from collections import Counter
import math
import random
import time
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

def BPNN_XGBoost(excel_second_data):

    dieasea_result = 'result.xlsx'
    def get_data():  

        df = pd.read_csv(excel_second_data)
        data_all = df.iloc[:, 2:116].values
        label_D_all= df.iloc[:, 117].values
        return data_all,label_D_all

    def MAXmin(train_feature,test_feature):
       
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
        train_minmax = min_max_scaler.transform(train_feature)
        test_minmax = min_max_scaler.transform(test_feature)
        return train_minmax, test_minmax

    time_start = time.time()
    wb = Workbook()
    sheet = wb.create_sheet("diseaseout",index = 0)
    row = ["fold_num","TP","TN","FP","FN","ACC","AUC","P","R", "F-measure", "G-mean"]
    sheet.append(row)
    i = 11
    tprs = []
    aucs = []
    accs = []
    tprs_50 = []
    aucs_50 = []
    accs_50 = []
    P_50 = []
    R_50 = []
    F1_SCORE = []
    G_mean = []
    mean_fpr = np.linspace(0, 1, 100)
    num = 2
    j=1
    for j in range(50):
            data_all, label_all= get_data()
            data = np.vstack(data_all)
            label  = np.hstack(label_all)
           
            best_rate = 0
            promoter_num = 0
            deafness_num = 0

            skf = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
            for train_index, test_index in skf.split(data, label):
                train_feature = np.array(data)[train_index]
                train_label = np.array(label)[train_index]

                test_feature = np.array(data)[test_index]
                test_label = np.array(label)[test_index]

                train_minmax, test_minmax = MAXmin(train_feature, test_feature)
                min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)

                XGB = xgb.XGBClassifier()
                model = XGB.fit(train_minmax, train_label)
                ACC = model.score(test_minmax, test_label)
                accs.append(ACC)


                pro = model.predict_proba(test_minmax)
                fpr, tpr, thresholds = roc_curve(test_label, pro[:,1], pos_label=1)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                AUC= auc(fpr, tpr)
                aucs.append(AUC)
                y_pred = model.predict(test_minmax)
                promoter_num  += Counter(y_pred)[1]
            confusion = confusion_matrix(test_label, y_pred)
            TP = confusion[0][0]
            FN = confusion[0][1]
            FP = confusion[1][0]
            TN = confusion[1][1]
            P = TP / (TP + FP)
            P_50.append(P)
            R = TP / (TP + FN)
            R_50.append(R)
            FPR = FP / (FP + TN)
            F_Measure = (2 * P * R) / (P + R)
            F1_SCORE.append(F_Measure)
            G_Mean = math.sqrt(R * (1 - FPR))
            G_mean.append(G_Mean)



            sheet["A%d" % (num)] = i - 1
            sheet["B%d" % (num)] = TP
            sheet["C%d" % (num)] = TN
            sheet["D%d" % (num)] = FP
            sheet["E%d" % (num)] = FN
            sheet["F%d" % (num)] = ACC
            sheet["G%d" % (num)] = AUC
            sheet["H%d" % (num)] = P
            sheet["I%d" % (num)] = R
            sheet["J%d" % (num)] = F_Measure
            sheet["K%d" % (num)] = G_Mean
            wb.save(dieasea_result)
    accs_50.append(np.mean(accs))
    aucs_50.append(np.mean(aucs))
    print(np.mean(accs_50))
    print(np.mean(aucs_50))
    deafness_num_mean = deafness_num/10
    print(deafness_num_mean)
    print(np.mean(P_50))
    print(np.mean(R_50))
    print(np.mean(F1_SCORE))
    print(np.mean(G_mean))
    plt.plot(fpr, tpr,lw =1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i-1, AUC))
    i += 1
    num+= 1

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, linestyle = "-",color='b', label=r'mean AUC=%0.2f' % mean_auc, lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('P:N = 1:1时   ROC')
    plt.legend(loc='lower right')
    plt.show()
    time_end = time.time()




