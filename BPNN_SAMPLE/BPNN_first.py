#!/usr/bin/python
#-*-coding:GBK -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neural_network ,preprocessing
from sklearn.model_selection import StratifiedShuffleSplit # �ֲ�����ָ����֤��
from sklearn.metrics import roc_curve,auc,confusion_matrix
from scipy import interpolate
from openpyxl import Workbook #openpyxl����õ�python����excel�����
from collections import Counter
from sklearn.model_selection import train_test_split
import math
import time

   
def BPNN_first(excel_read):

     
    excel_duplicates_data = '.xlsx'
    promoter_result = '.xlsx'

    def get_data():
        df = pd.read_excel(excel_read, sheet_name="Sheet5")
       
        data_all= df.iloc[:, 2:116].values 
        label_all = df.iloc[:, 116].values
        return data_all, label_all



    def MAXmin(train_feature,test_feature):
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
        train_minmax = min_max_scaler.transform(train_feature)
        test_minmax = min_max_scaler.transform(test_feature)
        return train_minmax, test_minmax

    time_start = time.time()
    wb = Workbook()
    sheet = wb.create_sheet(index = 0)
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
    j = 1
    for j in range(50):
            data_all, label_all= get_data()
            data = np.vstack(data_all)
            label  = np.hstack(label_all)
           
            best_rate = 0
            deafness_num = 0

            skf = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
            for train_index, test_index in skf.split(data, label):
                train_feature = np.array(data)[train_index]
                train_label = np.array(label)[train_index]

                test_feature = np.array(data)[test_index]
                test_label = np.array(label)[test_index]

                train_minmax, test_minmax = MAXmin(train_feature, test_feature)
                min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)

                clf = neural_network.MLPClassifier()
                model = clf.fit(train_minmax, train_label)
                ACC = model.score(test_minmax, test_label)
                accs.append(ACC)

                inputdata = pd.read_excel(excel_read,sheet_name="Sheet5")
                char_1=[1]
                count=0
                pos_label= model.predict(train_minmax)
                str_list=pos_a
                for each_char in str_list:
                     count+=1
                     if each_char==char_1:
                        line=count-1
                        list1 = inputdata.iloc[line].values
                        dfpromter0 = pd.DataFrame(list1)
                        dfpromter = dfpromter0.T
                        dfpromter.to_csv(excel_duplicates_data, encoding='utf-8-sig', index=False,mode='a', header=False)

                        '''            
                inputdata1 = pd.read_excel(excel_read_t,sheet_name="Sheet5")
                char_2=[1]
                count1=0
                pos_label1= model.predict(test_minmax)#
                pos_b=pos_label1.reshape(-1,1)#          
                str_list1=pos_b
                for each_char in str_list1:
                   count1+=1
                   if each_char==char_2:
                       line1=count-1
                       list2 = inputdata1.iloc[line1].values
                       dfpromter1 = pd.DataFrame(list2)   
                       dfpromter1 = dfpromter1.T #���л���                                               
                       dfpromter1.to_csv(excel_duplicates_data, encoding='utf-8-sig', index=False,mode='a', header=False)

'''


                pro = model.predict_proba(test_minmax)
                fpr, tpr, thresholds = roc_curve(test_label, pro[:,1], pos_label=1)
                tprs.append(np.interp(mean_fpr, fpr, tpr))

                AUC = auc(fpr, tpr)
                aucs.append(AUC)
                y_pred = model.predict(test_minmax)
                deafness_num += Counter(y_pred)[1]

            confusion = confusion_matrix(test_label, y_pred)
            TP = confusion[0,0]
            FN = confusion[0,1]
            FP = confusion[1,0]
            TN = confusion[1,1]
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
            wb.save(promoter_result)
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
        ##ƽ��AUCֵ#####
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
    plt.title('P:N = 1:1ʱ   ROC')
    plt.legend(loc='lower right')
    plt.show()
    time_end = time.time()
    timeall = time_end - time_start
    #print("The time  is:", time_end - time_start)

    return excel_duplicates_data


