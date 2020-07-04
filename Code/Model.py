#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import string
import numpy as np
import pandas as pd
from scipy import interp
from tabulate import tabulate

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import KernelPCA

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[2]:


def sub_feature_importance(X, y, name):
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X, y)
    
    importances = random_forest_classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest_classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    important_features = []
    for i in range(X.shape[1]):
        if name == 'PDF':
            if indices[i] <= 3299:
                important_features.append(indices[i])
        if name == 'PDG_1':
            if indices[i] >= 3300 and indices[i] <= 10403:
                important_features.append(indices[i])
        if name == 'PDG_2':
            if indices[i] >= 10404 and indices[i] <= 17507:
                important_features.append(indices[i])
        if name == 'PDG_3':
            if indices[i] >= 17508 and indices[i] <= 19875:
                important_features.append(indices[i])
                
    return important_features[:100]


# In[3]:


def feature_importance(X, y):
    
    important_features = []
        
    PDF   = sub_feature_importance(X, y, 'PDF')
    PDG_1 = sub_feature_importance(X, y, 'PDG_1')
    PDG_2 = sub_feature_importance(X, y, 'PDG_2')
    PDG_3 = sub_feature_importance(X, y, 'PDG_3')
    
    for i in PDF:
        important_features.append(i)
        
    for i in PDG_1:
        important_features.append(i)
        
    for i in PDG_2:
        important_features.append(i)
        
    for i in PDG_3:
        important_features.append(i)
        
    important_features.append(19876)
   
    return important_features


# In[4]:


def scores(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_proba = classifier.predict_proba(X_test)
    predictions_proba = predictions_proba[:, 1]
    
    fp, tp, threshold = roc_curve(y_test, predictions_proba)
    precision_position, recall_position, _ = precision_recall_curve(y_test, predictions_proba)
    
    accuracy    = round(accuracy_score(y_test, predictions) * 100, 4)
    auc         = round(roc_auc_score(y_test, predictions_proba) * 100, 4)
    aupr        = round(average_precision_score(y_test, predictions_proba) * 100, 4)
    precision   = round(precision_score(y_test, predictions, average='binary') * 100, 4)
    recall      = round(recall_score(y_test, predictions, average='binary') * 100, 4)
    specificity = round(specificity_score(y_test, predictions) * 100, 4)
    f1          = round(f1_score(y_test, predictions, average='binary') * 100, 4)
    mcc         = round(matthews_corrcoef(y_test, predictions) * 100, 4)
        
    return accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions_proba


# In[5]:


def independent_test_scores(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_proba = classifier.predict_proba(X_test)
    predictions_proba = predictions_proba[:, 1]
    
    fp, tp, threshold = roc_curve(y_test, predictions_proba)
    precision_position, recall_position, _ = precision_recall_curve(y_test, predictions_proba)
    
    accuracy    = round(accuracy_score(y_test, predictions) * 100, 4)
    auc         = round(roc_auc_score(y_test, predictions_proba) * 100, 4)
    aupr        = round(average_precision_score(y_test, predictions_proba) * 100, 4)
    precision   = round(precision_score(y_test, predictions, average='binary') * 100, 4)
    recall      = round(recall_score(y_test, predictions, average='binary') * 100, 4)
    specificity = round(specificity_score(y_test, predictions) * 100, 4)
    f1          = round(f1_score(y_test, predictions, average='binary') * 100, 4)
    mcc         = round(matthews_corrcoef(y_test, predictions) * 100, 4)
        
    return accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions_proba


# In[6]:


df = pd.read_csv("All_Features_Dataset.csv")


# In[7]:


df.head()


# In[8]:


X = df.iloc[:,1:]
y = df.iloc[:,0]


# In[9]:


X.head()


# In[10]:


encoder = preprocessing.LabelEncoder()


# In[11]:


y = encoder.fit_transform(y)
print(y)


# In[12]:


train = [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,13 ,14 ,15 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,46 ,48 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,96 ,97 ,98 ,99 ,100 ,102 ,103 ,104 ,105 ,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,134 ,135 ,136 ,137 ,138 ,139 ,140 ,141 ,142 ,144 ,145 ,147 ,148 ,149 ,150 ,151 ,152 ,153 ,154 ,155 ,156 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184 ,185 ,186 ,187 ,188 ,189 ,190 ,191 ,192 ,193 ,194 ,195 ,196 ,197 ,198 ,199 ,200 ,201 ,202 ,203 ,204 ,205 ,206 ,207 ,208 ,209 ,211 ,212 ,213 ,215 ,216 ,217 ,218 ,219 ,220 ,221 ,222 ,223 ,224 ,225 ,226 ,227 ,228 ,229 ,231 ,232 ,233 ,234 ,235 ,236 ,237 ,238 ,239 ,240 ,241 ,242 ,244 ,245 ,246 ,247 ,248 ,249 ,250 ,251 ,252 ,253 ,254 ,255 ,256 ,257 ,258 ,259 ,260 ,262 ,263 ,264 ,265 ,266 ,267 ,268 ,269 ,270 ,271 ,272 ,273 ,274 ,275 ,276 ,277 ,278 ,279 ,280 ,281 ,282 ,283 ,285 ,286 ,287 ,289 ,290 ,291 ,292 ,293 ,294 ,295 ,296 ,297 ,298 ,299 ,300 ,301 ,302 ,303 ,304 ,305 ,306 ,307 ,308 ,309 ,310 ,311 ,312 ,313 ,315 ,316 ,317 ,320 ,321 ,322 ,324 ,325 ,326 ,327 ,328 ,329 ,330 ,332 ,333 ,334 ,335 ,336 ,337 ,338 ,339 ,341 ,342 ,346 ,347 ,348 ,349 ,350 ,351 ,352 ,353 ,354 ,355 ,356 ,357 ,359 ,360 ,361 ,362 ,363 ,364 ,365 ,366 ,367 ,368 ,370 ,371 ,372 ,373 ,374 ,375 ,376 ,378 ,379 ,380 ,381 ,382 ,383 ,384 ,385 ,386 ,387 ,388 ,389 ,390 ,391 ,392 ,393 ,394 ,395 ,396 ,397 ,398 ,399 ,400 ,401 ,402 ,403 ,404 ,405 ,406 ,407 ,408 ,410 ,412 ,413 ,414 ,415 ,416 ,417 ,418 ,419 ,420 ,421 ,424 ,425 ,426 ,427 ,428 ,429 ,430 ,431 ,432 ,433 ,434 ,435 ,437 ,439 ,440 ,441 ,442 ,443 ,444 ,445 ,446 ,447 ,448 ,450 ,451 ,452 ,453 ,454 ,455 ,456 ,457 ,458 ,460 ,461 ,462 ,463 ,464 ,465 ,466 ,467 ,468 ,469 ,470 ,471 ,472 ,473 ,474 ,475 ,476 ,477 ,478 ,479 ,480 ,482 ,483 ,484 ,487 ,489 ,490 ,491 ,492 ,493 ,494 ,495 ,496 ,497 ,498 ,499 ,500 ,501 ,502 ,503 ,504 ,505 ,506 ,508 ,510 ,511 ,512 ,513 ,514 ,515 ,516 ,517 ,518 ,519 ,520 ,521 ,522 ,523 ,524 ,525 ,526 ,527 ,528 ,529 ,530 ,531 ,532 ,533 ,534 ,535 ,536 ,537 ,538 ,539 ,540 ,542 ,543 ,544 ,545 ,547 ,548 ,550 ,551 ,552 ,553 ,555 ,556 ,557 ,558 ,559 ,560 ,562 ,564 ,565 ,566 ,567 ,568 ,569 ,570 ,571 ,572 ,573 ,574 ,575 ,576 ,577 ,578 ,580 ,581 ,582 ,583 ,584 ,586 ,587 ,588 ,590 ,591 ,592 ,593 ,594 ,596 ,597 ,599 ,600 ,601 ,603 ,604 ,605 ,606 ,607 ,608 ,609 ,610 ,611 ,612 ,613 ,614 ,615 ,616 ,617 ,618 ,619 ,620 ,621 ,622 ,623 ,624 ,626 ,627 ,628 ,630 ,631 ,632 ,633 ,635 ,636 ,637 ,638 ,639 ,640 ,641 ,642 ,644 ,645 ,646 ,647 ,648 ,649 ,650 ,651 ,653 ,654 ,655 ,656 ,657 ,658 ,659 ,660 ,661 ,662 ,663 ,664 ,665 ,666 ,667 ,668 ,669 ,671 ,672 ,673 ,674 ,675 ,676 ,677 ,678 ,679 ,680 ,683 ,684 ,687 ,688 ,689 ,690 ,691 ,692 ,693 ,694 ,695 ,696 ,697 ,698 ,699 ,700 ,701 ,702 ,703 ,704 ,705 ,706 ,707 ,709 ,710 ,712 ,713 ,714 ,715 ,716 ,717 ,718 ,719 ,720 ,722 ,723 ,724 ,725 ,726 ,727 ,728 ,729 ,730 ,731 ,732 ,733 ,734 ,735 ,737 ,738 ,739 ,740 ,741 ,742 ,743 ,744 ,745 ,746 ,747 ,748 ,749 ,750 ,751 ,752 ,754 ,755 ,756 ,757 ,758 ,759 ,760 ,761 ,762 ,763 ,764 ,766 ,767 ,768 ,769 ,770 ,771 ,772 ,773 ,774 ,775 ,777 ,778 ,779 ,780 ,781 ,782 ,783 ,784 ,785 ,786 ,788 ,789 ,790 ,792 ,793 ,794 ,795 ,796 ,797 ,798 ,799 ,800 ,801 ,803 ,804 ,806 ,807 ,808 ,809 ,810 ,811 ,812 ,813 ,815 ,816 ,817 ,818 ,819 ,820 ,821 ,822 ,823 ,824 ,825 ,826 ,827 ,828 ,829 ,830 ,831 ,832 ,834 ,835 ,836 ,837 ,838 ,839 ,840 ,841 ,842 ,843 ,844 ,845 ,846 ,847 ,848 ,849 ,850 ,851 ,852 ,853 ,854 ,855 ,856 ,857 ,858 ,859 ,860 ,861 ,863 ,864 ,866 ,867 ,868 ,869 ,870 ,871 ,872 ,873 ,874 ,875 ,876 ,877 ,878 ,879 ,880 ,881 ,882 ,883 ,884 ,885 ,886 ,887 ,889 ,890 ,891 ,892 ,893 ,894 ,895 ,896 ,897 ,898 ,899 ,900 ,901 ,902 ,903 ,904 ,905 ,906 ,907 ,908 ,909 ,910 ,911 ,913 ,914 ,915 ,916 ,917 ,918 ,919 ,920 ,921 ,922 ,923 ,924 ,925 ,927 ,929 ,930 ,931 ,932 ,933 ,934 ,935 ,936 ,937 ,938 ,939 ,940 ,941 ,943 ,944 ,945 ,946 ,947 ,948 ,949 ,950 ,951 ,952 ,953 ,954 ,955 ,956 ,957 ,958 ,959 ,960 ,961 ,962 ,963 ,964 ,965 ,966 ,967 ,969 ,970 ,972 ,973 ,974 ,975 ,976 ,977 ,978 ,979 ,980 ,981 ,982 ,983 ,984 ,986 ,987 ,988 ,990 ,991 ,992 ,993 ,994 ,995 ,996 ,997 ,998 ,1000 ,1001 ,1002 ,1003 ,1004 ,1005 ,1006 ,1007 ,1009 ,1010 ,1011 ,1012 ,1013 ,1014 ,1015 ,1016 ,1018 ,1019 ,1020 ,1021 ,1022 ,1023 ,1024 ,1027 ,1028 ,1029 ,1030 ,1031 ,1032 ,1033 ,1034 ,1036 ,1037 ,1038 ,1039 ,1040 ,1042 ,1043 ,1044 ,1045 ,1047 ,1048 ,1049 ,1050 ,1051 ,1052 ,1053 ,1054 ,1055 ,1056 ,1057 ,1058 ,1059 ,1060 ,1062 ,1063 ,1064 ,1065 ,1066 ,1067 ,1068 ,1070 ,1071 ,1072 ,1073 ,1074 ,1075 ,1076 ,1077 ,1078 ,1079 ,1080 ,1082 ,1083 ,1084 ,1085 ,1086 ,1088 ,1089 ,1090 ,1091 ,1093 ,1094 ,1095 ,1096 ,1097 ,1098 ,1099 ,1100 ,1101 ,1103 ,1104 ,1105 ,1106 ,1107 ,1108 ,1109 ,1110 ,1111 ,1112 ,1113 ,1114 ,1115 ,1116 ,1117 ,1118 ,1119 ,1120 ,1121 ,1122 ,1123 ,1124 ,1125 ,1126 ,1127 ,1129 ,1131 ,1132 ,1133 ,1135 ,1136 ,1137 ,1138 ,1139 ,1140 ,1141 ,1142 ,1143 ,1144 ,1145 ,1146 ,1148 ,1149 ,1150 ,1151 ,1152 ,1153 ,1154 ,1155 ,1156 ,1157 ,1158 ,1159 ,1160 ,1162 ,1163 ,1165 ,1166 ,1167 ,1168 ,1169 ,1170 ,1171 ,1172 ,1173 ,1174 ,1177 ,1178 ,1179 ,1180 ,1181 ,1183 ,1185 ,1186 ,1187 ,1188 ,1189 ,1192 ,1193 ,1194 ,1195 ,1196 ,1197 ,1198 ,1199 ,1200 ,1201 ,1203 ,1204 ,1205 ,1207 ,1208 ,1209 ,1210 ,1211 ,1212 ,1213 ,1214 ,1215 ,1216 ,1217 ,1218 ,1219 ,1220 ,1221 ,1222 ,1224 ,1225 ,1226 ,1228 ,1229 ,1230 ,1231 ,1232 ,1233 ,1234 ,1235 ,1236 ,1237 ,1238 ,1239 ,1240 ,1241 ,1242 ,1243 ,1244 ,1245 ,1246 ,1247 ,1248 ,1249 ,1251 ,1252 ,1253 ,1255 ,1256 ,1257 ,1258 ,1259 ,1260 ,1261 ,1262 ,1263 ,1264 ,1265 ,1266 ,1267 ,1268 ,1269 ,1270 ,1271 ,1272 ,1273 ,1274 ,1277 ,1278 ,1281 ,1282 ,1283 ,1284 ,1285 ,1286 ,1287 ,1288 ,1289 ,1290 ,1291 ,1292 ,1293 ,1294 ,1295 ,1296 ,1297 ,1298 ,1299 ,1300 ,1301 ,1302 ,1303 ,1304 ,1305 ,1307 ,1308 ,1309 ,1310 ,1311 ,1312 ,1313 ,1314 ,1315 ,1316 ,1317 ,1318 ,1319 ,1320 ,1321 ,1322 ,1323]
test  = [1035, 1254, 1087, 1202, 912, 999, 1161, 791, 681, 670, 942, 753, 802, 1184, 1250, 1130, 1175, 736, 787, 805, 1046, 711, 926, 928, 1190, 765, 989, 971, 1041, 1176, 1092, 1017, 862, 1306, 1061, 685, 985, 1147, 814, 1182, 1223, 1026, 721, 1191, 1275, 1025, 1081, 686, 682, 1128, 1134, 968, 1069, 833, 776, 1164, 1227, 708, 1280, 1102, 1206, 888, 1276, 865, 1008, 1279, 95, 284, 541, 377, 16, 579, 146, 509, 409, 340, 438, 210, 563, 423, 422, 331, 652, 343, 459, 485, 598, 549, 101, 369, 314, 318, 629, 143, 243, 481, 436, 507, 344, 585, 411, 488, 230, 634, 323, 602, 625, 358, 546, 449, 214, 643, 486, 11, 133, 595, 49, 157, 78, 561, 169, 29, 345, 17, 87, 45, 319, 47, 261, 288, 554, 589]


# In[13]:


X_train_independent = X.iloc[train, :]
y_train_independent = y[train]

X_test_independent = X.iloc[test, :]
y_test_independent = y[test]

print(X_train_independent.shape, y_train_independent.shape)
print(X_test_independent.shape, y_test_independent.shape)


# In[14]:


classifier1 = SVC(probability=True) 
classifier2 = GaussianNB()                
classifier3 = LogisticRegression()

classifier_list = []
classifier_list.append(classifier1)
classifier_list.append(classifier2)
classifier_list.append(classifier3)

classifier_names = ['SVM', 'GaussianNB', 'Logistic Regression']



CV_table = []
test_table = []

std_CV_table = []
std_test_table = []
models = []


# In[ ]:


curve_fp_list = []
curve_tp_list = []
curve_precision_list = []
curve_recall_list = []

p = 0
for classifier in classifier_list:

    print(classifier)

    skf = StratifiedKFold(n_splits=5)
    accuracies     = []
    aucs           = []
    auprs          = []
    precisions     = [] 
    recalls        = []
    specificities  = []
    f1s            = []
    mccs           = []
    tp_list        = [] 
    mean_fp        = np.linspace(0, 1, 100)
    y_test_avg     = []
    predictions_avg = []
    count = 0

    print("CV Results:\n")
    
    temp_accuracy    = []
    temp_auc         = []
    temp_aupr        = []
    temp_precision   = []
    temp_recall      = []
    temp_specificity = []
    temp_f1          = []
    temp_mcc         = []
    
    for i in range(10):

        for train_index, test_index in skf.split(X_train_independent, y_train_independent):

            X_train   = X_train_independent.iloc[train_index, :]
            y_train   = y_train_independent[train_index]
            X_test    = X_train_independent.iloc[test_index, :]
            y_test    = y_train_independent[test_index]
                                    
            features = feature_importance(X_train, y_train)

            X_train = X_train.iloc[:, features]
            X_test  = X_test.iloc[:, features]

            accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions = scores(classifier, X_train, X_test, y_train, y_test)

            tp_list.append(interp(mean_fp, fp, tp))
            tp_list[-1][0] = 0.0

            accuracies.append(accuracy)
            aucs.append(auc)
            auprs.append(aupr)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1s.append(f1)
            mccs.append(mcc)
            y_test_avg.append(y_test)
            predictions_avg.append(predictions)

            count += 1
            print("CV -", count)

            final_y_test      = np.concatenate(y_test_avg)
            final_predictions = np.concatenate(predictions_avg)
            mean_tp     = np.mean(tp_list, axis=0)
            mean_tp[-1] = 1.0
            precision_pos, recall_pos, _ = precision_recall_curve(final_y_test, final_predictions)

        average_accuracy    = np.mean(accuracies)
        average_auc         = np.mean(aucs)
        average_aupr        = np.mean(auprs)
        average_precision   = np.mean(precisions)
        average_recall      = np.mean(recalls)
        average_specificity = np.mean(specificities)
        average_f1          = np.mean(f1s)
        average_mcc         = np.mean(mcc)
        
        temp_accuracy.append(average_accuracy)
        temp_auc.append(average_auc)
        temp_aupr.append(average_aupr)
        temp_precision.append(average_precision)
        temp_recall.append(average_recall)
        temp_specificity.append(average_specificity)
        temp_f1.append(average_f1)
        temp_mcc.append(average_mcc)
        
    models.append(classifier)
        
    curve_fp_list.append(mean_fp)
    curve_tp_list.append(mean_tp)
    curve_precision_list.append(precision_pos)
    curve_recall_list.append(recall_pos)
        
    accuracy    = np.mean(temp_accuracy)
    auc         = np.mean(temp_auc)
    aupr        = np.mean(temp_aupr)
    precision   = np.mean(temp_precision)
    recall      = np.mean(temp_recall)
    specificity = np.mean(temp_specificity)
    f1          = np.mean(temp_f1)
    mcc         = np.mean(temp_mcc)
    
    std_accuracy    = np.std(temp_accuracy, dtype = np.float32)
    std_auc         = np.std(temp_auc, dtype = np.float32)
    std_aupr        = np.std(temp_aupr, dtype = np.float32)
    std_precision   = np.std(temp_precision, dtype = np.float32)
    std_recall      = np.std(temp_recall, dtype = np.float32)
    std_specificity = np.std(temp_specificity, dtype = np.float32)
    std_f1          = np.std(temp_f1, dtype = np.float32)
    std_mcc         = np.std(temp_mcc, dtype = np.float32)
    
    CV_table.append([classifier_names[p], accuracy, auc, aupr, precision, recall, specificity, f1, mcc])
    std_CV_table.append([classifier_names[p], std_accuracy, std_auc, std_aupr, std_precision, std_recall, 
                           std_specificity, std_f1, std_mcc])

    print("Accuracy\t:", accuracy)
    print("AUC\t\t:", auc)
    print("AUPR\t\t:", aupr)
    print("Precision\t:", precision)
    print("Recall\t\t:", recall)
    print("Specificity\t:", specificity)
    print("F1\t\t:", f1)
    print("MCC\t\t:", mcc)
    
    p += 1
    
colors = ['red', '#03a9f4', '#4caf50']    
labels = ['SVM', 'GNB', 'LR']

for i in range(len(curve_fp_list)):
    pyplot.plot(curve_fp_list[i], curve_tp_list[i], color=colors[i], label = labels[i], lw=1.5, alpha=1)
pyplot.plot([0, 1], [0, 1], color='black', linestyle = "--", lw=1.5, alpha=1)
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('False Positive')
pyplot.ylabel('True Positive')
pyplot.title('ROC Curve')
pyplot.legend(loc="lower right")
pyplot.show()

for i in range(len(curve_precision_list)):
    pyplot.plot(curve_recall_list[i], curve_precision_list[i], color=colors[i], label = labels[i], lw=1.5, alpha=1)
pyplot.plot([0, 1], [1, 0], color='black', linestyle = "--", lw=1.5, alpha=1)
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.title('AUPR Curve')
pyplot.legend(loc="lower left")
pyplot.show()


# In[22]:


features = feature_importance(X_train_independent, y_train_independent)

X_train_independent = X_train_independent.iloc[:, features]
X_test_independent  = X_test_independent.iloc[:, features]

curve_fp_list = []
curve_tp_list = []
curve_precision_list = []
curve_recall_list = []

p = 0

for classifier in models:

    print(classifier)
    print("\nIndependent Results:\n")
    
    temp_accuracy    = []
    temp_auc         = []
    temp_aupr        = []
    temp_precision   = []
    temp_recall      = []
    temp_specificity = []
    temp_f1          = []
    temp_mcc         = []
    
    for i in range(10):
        accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions = independent_test_scores(classifier, X_train_independent , X_test_independent, y_train_independent, y_test_independent)
        temp_accuracy.append(accuracy)
        temp_auc.append(auc)
        temp_aupr.append(aupr)
        temp_precision.append(precision)
        temp_recall.append(recall)
        temp_specificity.append(specificity)
        temp_f1.append(f1)
        temp_mcc.append(mcc)
        print(i)
        
    precision_pos, recall_pos, _ = precision_recall_curve(y_test  , predictions)

    curve_fp_list.append(fp)
    curve_tp_list.append(tp)
    curve_precision_list.append(precision_pos)
    curve_recall_list.append(recall_pos)
    
    accuracy    = np.mean(temp_accuracy)
    auc         = np.mean(temp_auc)
    aupr        = np.mean(temp_aupr)
    precision   = np.mean(temp_precision)
    recall      = np.mean(temp_recall)
    specificity = np.mean(temp_specificity)
    f1          = np.mean(temp_f1)
    mcc         = np.mean(temp_mcc)
    
    std_accuracy    = np.std(temp_accuracy, dtype = np.float32)
    std_auc         = np.std(temp_auc, dtype = np.float32)
    std_aupr        = np.std(temp_aupr, dtype = np.float32)
    std_precision   = np.std(temp_precision, dtype = np.float32)
    std_recall      = np.std(temp_recall, dtype = np.float32)
    std_specificity = np.std(temp_specificity, dtype = np.float32)
    std_f1          = np.std(temp_f1, dtype = np.float32)
    std_mcc         = np.std(temp_mcc, dtype = np.float32)
    
    test_table.append([classifier_names[p], accuracy, auc, aupr, precision, recall, specificity, f1, mcc])
    
    std_test_table.append([classifier_names[p], std_accuracy, std_auc, std_aupr, std_precision, std_recall, 
                           std_specificity, std_f1, std_mcc])

    print("Accuracy\t:", accuracy)
    print("AUC\t\t:", auc)
    print("AUPR\t\t:", aupr)
    print("Precision\t:", precision)
    print("Recall\t\t:", recall)
    print("Specificity\t:", specificity)
    print("F1\t\t:", f1)
    print("MCC\t\t:", mcc)
    
    p += 1

print("\n## CV Result ##")
print(tabulate(CV_table, headers= ['Classifier', 'Accuracy', 'AUC', 'AUPR', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']))

print("\n## STD CV Result ##")
print(tabulate(std_CV_table, headers= ['Classifier', 'Accuracy', 'AUC', 'AUPR', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']))


print("\n## Test Result ##")
print(tabulate(test_table, headers= ['Classifier', 'Accuracy', 'AUC', 'AUPR', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']))
    

print("\n## STD Test Result ##")
print(tabulate(std_test_table, headers= ['Classifier', 'Accuracy', 'AUC', 'AUPR', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']))

colors = ['red', '#03a9f4', '#4caf50']    
labels = ['SVM', 'GNB', 'LR']

for i in range(len(curve_fp_list)):
    pyplot.plot(curve_fp_list[i], curve_tp_list[i], color=colors[i], label = labels[i], lw=1.5, alpha=1)
pyplot.plot([0, 1], [0, 1], color='black', linestyle = "--", lw=1.5, alpha=1)
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('False Positive')
pyplot.ylabel('True Positive')
pyplot.title('ROC Curve')
pyplot.legend(loc="lower right")
pyplot.show()

for i in range(len(curve_precision_list)):
    pyplot.plot(curve_recall_list[i], curve_precision_list[i], color=colors[i], label = labels[i], lw=1.5, alpha=1)
pyplot.plot([0, 1], [1, 0], color='black', linestyle = "--", lw=1.5, alpha=1)
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.title('AUPR Curve')
pyplot.legend(loc="lower left")
pyplot.show()

