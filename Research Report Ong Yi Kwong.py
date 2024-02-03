import pandas as pd
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_score,classification_report, recall_score, f1_score, confusion_matrix
from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats

downloads_directory = r'C:\Users\Yi Kwong\Downloads'


os.chdir(downloads_directory)


print("Current Directory:", os.getcwd())


raw_data=pd.read_csv(r"C:\Users\Yi Kwong\Downloads\train.csv (1)\Mortarlity risk raw data.csv")

raw=pd.DataFrame(raw_data)

raw=raw.drop(["Id"],axis=1)

print(len(raw))

raw["Product_Info_2"] = raw["Product_Info_2"].replace({'A1': 0,'A2': 1,'A3': 2,'A4': 3,'A5': 4,'A6': 5,'A7': 6,'A8': 7,'B1': 8 ,'B2': 9, 'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'D1': 14, 'D2': 15, 'D3': 16, 'D4': 17, 'E1': 18})

missing = raw.isnull().sum()
columns_with_missing = missing [missing >=1].index
print(columns_with_missing)

raw = raw.fillna(raw.mean())

# correlation_series = raw.corr()['Response']
# selected_columns = correlation_series[abs(correlation_series) >= 0.001].index
# raw = raw[selected_columns]

# columns_to_check = ["Ins_Age", "Ht", "Wt", "BMI", "Product_Info_4", "Employment_Info_1", "Employment_Info_6", 'Insurance_History_5', "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]

# for column in columns_to_check:
#     Q1 = raw[column].quantile(0.25)
#     Q3 = raw[column].quantile(0.75)
#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     raw= raw[(raw[column] >= lower_bound) & (raw[column] <= upper_bound)]

# raw = raw.fillna(raw.mean())

X = raw.drop('Response', axis=1)
y = raw['Response']

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=55)

print(len(raw))
##########################################################
#Default KNN
start_time = time.time()

knn_classifier = KNeighborsClassifier(n_neighbors=55)

knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

end_time = time.time()
knn_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

#############################################################################

#Grid Optimization KNN

start_time = time.time()

knn_param_grid = {
    'n_neighbors': np.arange(5,256,5)         
}



knn_grid_search = GridSearchCV(
    knn_classifier,
   knn_param_grid,  
   cv=5
)

knn_grid_search.fit(X_train, y_train)
y_pred = knn_grid_search.predict(X_test)

end_time = time.time()
knn_grid_time=end_time-start_time

knn_grid_param= knn_grid_search.best_estimator_
knn_grid_accuracy = accuracy_score(y_test, y_pred)
knn_grid_recall = recall_score(y_test, y_pred, average='weighted')
knn_grid_precision=precision_score(y_test, y_pred, average='weighted')
knn_grid_f1 = f1_score(y_test, y_pred, average='weighted')


print("knn_grid_best_parameters: ", knn_grid_param)
print("knn accuracy grid:", knn_grid_accuracy)
print("knn recall grid:", knn_grid_recall)
print("knn precision grid:", knn_grid_precision)
print("knn f1 grid:", knn_grid_f1)
print("knn time grid", knn_grid_time)
########################################################################################
#Random Optimization KNN

start_time = time.time()

knn_param_random = {
    'n_neighbors': np.arange(5,256,5)         
}


knn_random_search = RandomizedSearchCV(
    knn_classifier,
   knn_param_random,  
   cv=5
)

knn_random_search.fit(X_train, y_train)
y_pred = knn_random_search.predict(X_test)

end_time = time.time()
knn_random_time=end_time-start_time

knn_random_param= knn_random_search.best_estimator_
knn_random_accuracy = accuracy_score(y_test, y_pred)
knn_random_recall = recall_score(y_test, y_pred, average='weighted')
knn_random_precision=precision_score(y_test, y_pred, average='weighted')
knn_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("knn_random_best_parameters: ", knn_random_param)
print("knn accuracy random:", knn_random_accuracy)
print("knn recall random:", knn_random_recall)
print("knn precision random:", knn_random_precision)
print("knn f1 random:", knn_random_f1)
print("knn time random", knn_random_time)
############################################################################################
#Bayesian Optimization KNN

start_time = time.time()

knn_param_bayesian = {
    'n_neighbors': np.arange(5,256,5)  
}

knn_bayesian_search = BayesSearchCV(
    knn_classifier,
    knn_param_bayesian,
    cv=5
)

knn_bayesian_search.fit(X_train, y_train)
y_pred = knn_bayesian_search.predict(X_test)

end_time = time.time()
knn_bayesian_time = end_time - start_time

knn_bayesian_param = knn_bayesian_search.best_estimator_
knn_bayesian_accuracy = accuracy_score(y_test, y_pred)
knn_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
knn_bayesian_precision = precision_score(y_test, y_pred, average='weighted')
knn_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("knn_bayesian_best_parameters: ", knn_bayesian_param)
print("knn accuracy bayesian:", knn_bayesian_accuracy)
print("knn recall bayesian:", knn_bayesian_recall)
print("knn precision bayesian:", knn_bayesian_precision)
print("knn f1 bayesian:", knn_bayesian_f1)
print("knn time bayesian", knn_bayesian_time)
############################################################################################3
#Default LOG
start_time = time.time()

log_classifier = LogisticRegression(C=24.6,random_state=55)

log_classifier.fit(X_train, y_train)
y_pred = log_classifier.predict(X_test)

end_time = time.time()
log_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

###############################################################################################

#Grid Optimization LOG

start_time = time.time()

log_param_grid = {
    'C': np.arange(0.1,26.1,0.5) 
}


log_grid_search = GridSearchCV(
    log_classifier,
   log_param_grid,  
   cv=5
)

log_grid_search.fit(X_train, y_train)
y_pred = log_grid_search.predict(X_test)

end_time = time.time()
log_grid_time=end_time-start_time

log_grid_param= log_grid_search.best_estimator_
log_grid_accuracy = accuracy_score(y_test, y_pred)
log_grid_recall = recall_score(y_test, y_pred, average='weighted')
log_grid_precision=precision_score(y_test, y_pred, average='weighted')
log_grid_f1 = f1_score(y_test, y_pred, average='weighted')

print("log_grid_best_parameters: ", log_grid_param)
print("log accuracy grid:", log_grid_accuracy)
print("log recall grid:", log_grid_recall)
print("log precision grid:", log_grid_precision)
print("log f1 grid:", log_grid_f1)
print("log time grid", log_grid_time)
#############################################################################################
#Random Optimization LOG

start_time = time.time()

log_param_random = {
    'C': np.arange(0.1,26.1,0.5)         
}

log_random_search = RandomizedSearchCV(
    log_classifier,
   log_param_random,  
   cv=5
)

log_random_search.fit(X_train, y_train)
y_pred = log_random_search.predict(X_test)

end_time = time.time()
log_random_time=end_time-start_time

log_random_param= log_random_search.best_estimator_
log_random_accuracy = accuracy_score(y_test, y_pred)
log_random_recall = recall_score(y_test, y_pred, average='weighted')
log_random_precision=precision_score(y_test, y_pred, average='weighted')
log_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("log_random_best_parameters: ", log_random_param)
print("log accuracy random:", log_random_accuracy)
print("log recall random:", log_random_recall)
print("log precision random:", log_random_precision)
print("log f1 random:", log_random_f1)
print("log time random", log_random_time)
##################################################################################################
#Bayesian Optimization LOG

start_time = time.time()

log_param_bayesian = {
    'C': np.arange(0.1,26.1,0.5)         
}

log_bayesian_search = BayesSearchCV(
    log_classifier,
   log_param_bayesian,  
   cv=5
)

log_bayesian_search.fit(X_train, y_train)
y_pred = log_bayesian_search.predict(X_test)

end_time = time.time()
log_bayesian_time=end_time-start_time

log_bayesian_param= log_bayesian_search.best_estimator_
log_bayesian_accuracy = accuracy_score(y_test, y_pred)
log_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
log_bayesian_precision=precision_score(y_test, y_pred, average='weighted')
log_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("log_bayesian_best_parameters: ", log_bayesian_param)
print("log accuracy bayesian:", log_bayesian_accuracy)
print("log recall bayesian:", log_bayesian_recall)
print("log precision bayesian:", log_bayesian_precision)
print("log f1 bayesian:", log_bayesian_f1)
print("log time bayesian", log_bayesian_time)
########################################################################################
#Default SVM
start_time = time.time()

svm_classifier = SVC(random_state=55)

svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

end_time = time.time()
svm_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

############################################################################################
#Grid Optimization SVM

start_time = time.time()

svm_param_grid = {
    'C': np.arange(0.1,26.1,0.5) 
}


svm_grid_search = GridSearchCV(
    svm_classifier,
   svm_param_grid,  
   cv=5
)

svm_grid_search.fit(X_train, y_train)
y_pred = svm_grid_search.predict(X_test)

end_time = time.time()
svm_grid_time=end_time-start_time

svm_grid_param= svm_grid_search.best_estimator_
svm_grid_accuracy = accuracy_score(y_test, y_pred)
svm_grid_recall = recall_score(y_test, y_pred, average='weighted')
svm_grid_precision=precision_score(y_test, y_pred, average='weighted')
svm_grid_f1 = f1_score(y_test, y_pred, average='weighted')

print("svm_grid_best_parameters: ", svm_grid_param)
print("svm accuracy grid:", svm_grid_accuracy)
print("svm recall grid:", svm_grid_recall)
print("svm precision grid:", svm_grid_precision)
print("svm f1 grid:", svm_grid_f1)
print("svm time grid", svm_grid_time)
#################################################################################################
#Random Optimization SVM

start_time = time.time()

svm_param_random = {
    'C': np.arange(0.1,26.1,0.5)         
}

svm_random_search = RandomizedSearchCV(
    svm_classifier,
   svm_param_random,  
   cv=5
)

svm_random_search.fit(X_train, y_train)
y_pred = svm_random_search.predict(X_test)

end_time = time.time()
svm_random_time=end_time-start_time

svm_random_param= svm_random_search.best_estimator_
svm_random_accuracy = accuracy_score(y_test, y_pred)
svm_random_recall = recall_score(y_test, y_pred, average='weighted')
svm_random_precision=precision_score(y_test, y_pred, average='weighted')
svm_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("svm_random_best_parameters: ", svm_random_param)
print("svm accuracy random:", svm_random_accuracy)
print("svm recall random:", svm_random_recall)
print("svm precision random:", svm_random_precision)
print("svm f1 random:", svm_random_f1)
print("svm time random", svm_random_time)
#####################################################################################################
#Bayesian Optimization SVM

start_time = time.time()

svm_param_bayesian = {
    'C': np.arange(0.1,26.1,0.5)         
}

svm_bayesian_search = BayesSearchCV(
    svm_classifier,
   svm_param_bayesian,  
   cv=5
)

svm_bayesian_search.fit(X_train, y_train)
y_pred = svm_bayesian_search.predict(X_test)

end_time = time.time()
svm_bayesian_time=end_time-start_time

svm_bayesian_param= svm_bayesian_search.best_estimator_
svm_bayesian_accuracy = accuracy_score(y_test, y_pred)
svm_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
svm_bayesian_precision=precision_score(y_test, y_pred, average='weighted')
svm_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("svm_bayesian_best_parameters: ", svm_bayesian_param)
print("svm accuracy bayesian:", svm_bayesian_accuracy)
print("svm recall bayesian:", svm_bayesian_recall)
print("svm precision bayesian:", svm_bayesian_precision)
print("svm f1 bayesian:", svm_bayesian_f1)
print("svm time bayesian", svm_bayesian_time)
##########################################################################################
#Default DT
start_time = time.time()

DT_classifier = DecisionTreeClassifier(max_depth=30,random_state=55)

DT_classifier.fit(X_train, y_train)
y_pred = DT_classifier.predict(X_test)

end_time = time.time()
DT_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

###########################################################################################
#Default DT
start_time = time.time()

svm_classifier = SVC(random_state=55)

svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

end_time = time.time()
svm_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

#############################################################################################
#Grid Optimization DT

start_time = time.time()

DT_param_grid = {
    'max_depth':np.arange(10,511,10)        
}


DT_grid_search = GridSearchCV(
    DT_classifier,
   DT_param_grid,  
   cv=5
)

DT_grid_search.fit(X_train, y_train)
y_pred = DT_grid_search.predict(X_test)

end_time = time.time()
DT_grid_time=end_time-start_time

DT_grid_param= DT_grid_search.best_estimator_
DT_grid_accuracy = accuracy_score(y_test, y_pred)
DT_grid_recall = recall_score(y_test, y_pred, average='weighted')
DT_grid_precision=precision_score(y_test, y_pred, average='weighted')
DT_grid_f1 = f1_score(y_test, y_pred, average='weighted')

print("DT_grid_best_parameters: ", DT_grid_param)
print("DT accuracy grid:", DT_grid_accuracy)
print("DT recall grid:", DT_grid_recall)
print("DT precision grid:", DT_grid_precision)
print("DT f1 grid:", DT_grid_f1)
print("DT time grid", DT_grid_time)

#Random Optimization DT

start_time = time.time()

DT_param_random = {
    'max_depth':np.arange(10,511,10)        
}

DT_random_search = RandomizedSearchCV(
    DT_classifier,
   DT_param_random,  
   cv=5
)

DT_random_search.fit(X_train, y_train)
y_pred = DT_random_search.predict(X_test)

end_time = time.time()
DT_random_time=end_time-start_time

DT_random_param= DT_random_search.best_estimator_
DT_random_accuracy = accuracy_score(y_test, y_pred)
DT_random_recall = recall_score(y_test, y_pred, average='weighted')
DT_random_precision=precision_score(y_test, y_pred, average='weighted')
DT_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("DT_random_best_parameters: ", DT_random_param)
print("DT accuracy random:", DT_random_accuracy)
print("DT recall random:", DT_random_recall)
print("DT precision random:", DT_random_precision)
print("DT f1 random:", DT_random_f1)
print("DT time random", DT_random_time)
######################################################################################3
#Bayesian Optimization DT

start_time = time.time()

DT_param_bayesian = {
    'max_depth':np.arange(10,511,10)         
}

DT_bayesian_search = BayesSearchCV(
    DT_classifier,
   DT_param_bayesian,  
   cv=5
)

DT_bayesian_search.fit(X_train, y_train)
y_pred = DT_bayesian_search.predict(X_test)

end_time = time.time()
DT_bayesian_time=end_time-start_time

DT_bayesian_param= DT_bayesian_search.best_estimator_
DT_bayesian_accuracy = accuracy_score(y_test, y_pred)
DT_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
DT_bayesian_precision=precision_score(y_test, y_pred, average='weighted')
DT_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("DT_bayesian_best_parameters: ", DT_bayesian_param)
print("DT accuracy bayesian:", DT_bayesian_accuracy)
print("DT recall bayesian:", DT_bayesian_recall)
print("DT precision bayesian:", DT_bayesian_precision)
print("DT f1 bayesian:", DT_bayesian_f1)
print("DT time bayesian", DT_bayesian_time)

################################################################################################
#Default RF
start_time = time.time()

RF_classifier = RandomForestClassifier(max_depth=50,random_state=55)

RF_classifier.fit(X_train, y_train)
y_pred = RF_classifier.predict(X_test)

end_time = time.time()
RF_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

##############################################################################################

#Grid Optimization RF

start_time = time.time()

RF_param_grid = {
    'max_depth':np.arange(10,511,10)        
}


RF_grid_search = GridSearchCV(
    RF_classifier,
   RF_param_grid,  
   cv=5
)

RF_grid_search.fit(X_train, y_train)
y_pred = RF_grid_search.predict(X_test)

end_time = time.time()
RF_grid_time=end_time-start_time

RF_grid_param= RF_grid_search.best_estimator_
RF_grid_accuracy = accuracy_score(y_test, y_pred)
RF_grid_recall = recall_score(y_test, y_pred, average='weighted')
RF_grid_precision=precision_score(y_test, y_pred, average='weighted')
RF_grid_f1 = f1_score(y_test, y_pred, average='weighted')

print("RF_grid_best_parameters: ", RF_grid_param)
print("RF accuracy grid:", RF_grid_accuracy)
print("RF recall grid:", RF_grid_recall)
print("RF precision grid:", RF_grid_precision)
print("RF f1 grid:", RF_grid_f1)
print("RF time grid", RF_grid_time)
####################################################################################3
#Random Optimization RF

start_time = time.time()

RF_param_random = {
    'max_depth':np.arange(10,511,10)       
}

RF_random_search = RandomizedSearchCV(
    RF_classifier,
   RF_param_random,  
   cv=5
)

RF_random_search.fit(X_train, y_train)
y_pred = RF_random_search.predict(X_test)

end_time = time.time()
RF_random_time=end_time-start_time

RF_random_param= RF_random_search.best_estimator_
RF_random_accuracy = accuracy_score(y_test, y_pred)
RF_random_recall = recall_score(y_test, y_pred, average='weighted')
RF_random_precision=precision_score(y_test, y_pred, average='weighted')
RF_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("RF_random_best_parameters: ", RF_random_param)
print("RF accuracy random:", RF_random_accuracy)
print("RF recall random:", RF_random_recall)
print("RF precision random:", RF_random_precision)
print("RF f1 random:", RF_random_f1)
print("RF time random", RF_random_time)

####################################################################################################
#Bayesian Optimization RF

start_time = time.time()

RF_param_bayesian = {
    'max_depth':np.arange(10,511,10)         
}

RF_bayesian_search = BayesSearchCV(
    RF_classifier,
   RF_param_bayesian,  
   cv=5
)

RF_bayesian_search.fit(X_train, y_train)
y_pred = RF_bayesian_search.predict(X_test)

end_time = time.time()
RF_bayesian_time=end_time-start_time

RF_bayesian_param= RF_bayesian_search.best_estimator_
RF_bayesian_accuracy = accuracy_score(y_test, y_pred)
RF_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
RF_bayesian_precision=precision_score(y_test, y_pred, average='weighted')
RF_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("RF_bayesian_best_parameters: ", RF_bayesian_param)
print("RF accuracy bayesian:", RF_bayesian_accuracy)
print("RF recall bayesian:", RF_bayesian_recall)
print("RF precision bayesian:", RF_bayesian_precision)
print("RF f1 bayesian:", RF_bayesian_f1)
print("RF time bayesian", RF_bayesian_time)
#########################################################################################
#Default XGB
start_time = time.time()

XGB_classifier = XGBClassifier(max_depth=60,random_state=55)

XGB_classifier.fit(X_train, y_train)
y_pred = XGB_classifier.predict(X_test)

end_time = time.time()
XGB_default_time=end_time-start_time

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
report_df = pd.DataFrame(report_dict)

#########################################################################################

#Grid Optimization XGB

start_time = time.time()

XGB_param_grid = {
    'max_depth':np.arange(10,511,10)       
}


XGB_grid_search = GridSearchCV(
    XGB_classifier,
   XGB_param_grid,  
   cv=5
)

XGB_grid_search.fit(X_train, y_train)
y_pred = XGB_grid_search.predict(X_test)

end_time = time.time()
XGB_grid_time=end_time-start_time

XGB_grid_param= XGB_grid_search.best_estimator_
XGB_grid_accuracy = accuracy_score(y_test, y_pred)
XGB_grid_recall = recall_score(y_test, y_pred, average='weighted')
XGB_grid_precision=precision_score(y_test, y_pred, average='weighted')
XGB_grid_f1 = f1_score(y_test, y_pred, average='weighted')

print("XGB_grid_best_parameters: ", XGB_grid_param)
print("XGB accuracy grid:", XGB_grid_accuracy)
print("XGB recall grid:", XGB_grid_recall)
print("XGB precision grid:", XGB_grid_precision)
print("XGB f1 grid:", XGB_grid_f1)
print("XGB time grid", XGB_grid_time)
#########################################################################################
#Random Optimization XGB

start_time = time.time()

XGB_param_random = {
    'max_depth':np.arange(10,511,10)       
}

XGB_random_search = RandomizedSearchCV(
    XGB_classifier,
   XGB_param_random,  
   cv=5
)

XGB_random_search.fit(X_train, y_train)
y_pred = XGB_random_search.predict(X_test)

end_time = time.time()
XGB_random_time=end_time-start_time

XGB_random_param= XGB_random_search.best_estimator_
XGB_random_accuracy = accuracy_score(y_test, y_pred)
XGB_random_recall = recall_score(y_test, y_pred, average='weighted')
XGB_random_precision=precision_score(y_test, y_pred, average='weighted')
XGB_random_f1 = f1_score(y_test, y_pred, average='weighted')

print("XGB_random_best_parameters: ", XGB_random_param)
print("XGB accuracy random:", XGB_random_accuracy)
print("XGB recall random:", XGB_random_recall)
print("XGB precision random:", XGB_random_precision)
print("XGB f1 random:", XGB_random_f1)
print("XGB time random", XGB_random_time)
#########################################################################################
#Bayesian Optimization XGB

start_time = time.time()

XGB_param_bayesian = {
    'max_depth':np.arange(10,511,10)         
}

XGB_bayesian_search = BayesSearchCV(
    XGB_classifier,
   XGB_param_bayesian,  
   cv=5
)

XGB_bayesian_search.fit(X_train, y_train)
y_pred = XGB_bayesian_search.predict(X_test)

end_time = time.time()
XGB_bayesian_time=end_time-start_time

XGB_bayesian_param= XGB_bayesian_search.best_estimator_
XGB_bayesian_accuracy = accuracy_score(y_test, y_pred)
XGB_bayesian_recall = recall_score(y_test, y_pred, average='weighted')
XGB_bayesian_precision=precision_score(y_test, y_pred, average='weighted')
XGB_bayesian_f1 = f1_score(y_test, y_pred, average='weighted')

print("XGB_bayesian_best_parameters: ", XGB_bayesian_param)
print("XGB accuracy bayesian:", XGB_bayesian_accuracy)
print("XGB recall bayesian:", XGB_bayesian_recall)
print("XGB precision bayesian:", XGB_bayesian_precision)
print("XGB f1 bayesian:", XGB_bayesian_f1)
print("XGB time bayesian", XGB_bayesian_time)
#########################################################################################
RF_start_time=time.time()
RF_classifier = RandomForestClassifier(random_state=55)



RF_param_random = {
'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
'max_depth':np.arange(10,511,10),
'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}




RF_random_search = RandomizedSearchCV(
RF_classifier,
RF_param_random, 
cv=5
)

RF_random_search.fit(X_train, y_train)



print("RF_random_best_parameters: ", RF_random_search.best_params_)


RF_best_model_random = RF_random_search.best_estimator_
RF_accuracy_random= RF_random_search.score(X_test, y_test)
print("RF_random",RF_accuracy_random)
y_pred = RF_random_search.predict(X_test)
RF_precision=precision_score(y_test, y_pred, average='weighted')



RF_recall = recall_score(y_test, y_pred, average='weighted')


RF_f1 = f1_score(y_test, y_pred, average='weighted')
RF_end_time=time.time()
RF_total_time=RF_end_time-RF_start_time
################################################################################################################
DT_start_time=time.time()

DT_classifier = DecisionTreeClassifier(random_state=55)



DT_param_random = {
'max_depth':np.arange(10,511,10),
'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}




DT_random_search = RandomizedSearchCV(
DT_classifier,
DT_param_random, 
cv=5
)

DT_random_search.fit(X_train, y_train)



print("DT_random_best_parameters: ", DT_random_search.best_params_)


DT_best_model_random = DT_random_search.best_estimator_
DT_accuracy_random= DT_random_search.score(X_test, y_test)
print("DT_random",DT_accuracy_random)
y_pred = DT_random_search.predict(X_test)
DT_precision=precision_score(y_test, y_pred, average='weighted')



DT_recall = recall_score(y_test, y_pred, average='weighted')


DT_f1 = f1_score(y_test, y_pred, average='weighted')

DT_end_time=time.time()
DT_total_time=DT_end_time-DT_start_time

#################################################################################################################
XGB_start_time=time.time()
XGB_classifier = XGBClassifier(random_state=55)



XGB_param_random = {
'learning_rate': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
'max_depth':np.arange(10,511,10),
'subsample': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
'colsample_bytree': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
'min_child_weight': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}




XGB_random_search = RandomizedSearchCV(
XGB_classifier,
XGB_param_random, 
cv=5
)

XGB_random_search.fit(X_train, y_train)



print("XGB_random_best_parameters: ", XGB_random_search.best_params_)


XGB_best_model_random = XGB_random_search.best_estimator_
XGB_accuracy_random= XGB_random_search.score(X_test, y_test)
print("XGB_random",XGB_accuracy_random)
y_pred = XGB_random_search.predict(X_test)
XGB_precision=precision_score(y_test, y_pred, average='weighted')



XGB_recall = recall_score(y_test, y_pred, average='weighted')


XGB_f1 = f1_score(y_test, y_pred, average='weighted')

XGB_end_time=time.time()
XGB_total_time=XGB_end_time-XGB_start_time
