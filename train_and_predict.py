from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import time

def rsmle(train,test):
    return np.sqrt(np.mean(pow(np.log(test+1) - np.log(train+1),2)))

## SWITCH TRAIN MODEL ON\OFF
## If switched off, model need to be trained
trainGB_models = True
trainRF_models = True
dumpModels = True

## PREDICT ON\OFF
make_prediction = True

print "loading data"
train = pd.DataFrame.from_csv("DataProcessed\\TrainMoments.csv")
test = pd.DataFrame.from_csv("DataProcessed\\TestMoments.csv")
train_fea = pd.DataFrame.from_csv("DataProcessed\\TrainMoments_fea.csv")
test_fea = pd.DataFrame.from_csv("DataProcessed\\TestMoments_fea.csv")

## EXTRA FEATURE
train_fea['YearToSale'] = train_fea['SaleYear'] - train_fea['YearMade']
test_fea['YearToSale'] = test_fea['SaleYear'] - test_fea['YearMade']
##

train = train['SalePrice']

colToDropGB1 = ['PrevSale','ProductGroupDesc', 'MachineID','SaleYear']
colToDropGB2 = ['SaleCount', 'MaxToDate', 'PrevSale', 'ProductGroupDesc', 'MachineID',
                'Pad_Type', 'Turbocharged', 'Backhoe_Mounting', 'Differential_Type', 'SaleYear']

colToDropRF1 = ['MachineID','SaleDay','MfgYear','SaleYear','PrevSale','SaleCount',
                'MeanToDate_Machine','MedianToDate_Machine','MaxToDate_Machine','MinToDate_Machine']
colToDropRF2 = ['MachineID','SaleDay','MfgYear','SaleYear','PrevSale','SaleCount',
                'MeanToDate_Machine','MedianToDate_Machine','MaxToDate_Machine','MinToDate_Machine',
                'MeanToDate']

tot_init_time = time.time()
## RANDOM FOREST REGRESSORS - set trainRF_models for switch training on\off
rf1 = RandomForestRegressor(n_estimators=150, n_jobs=1, min_samples_leaf = 4,
                            compute_importances = True, random_state=7354)
rf2 = RandomForestRegressor(n_estimators=150, n_jobs=1, min_samples_leaf = 4,
                            compute_importances = True, random_state=7354)
if trainRF_models:
    print "fitting random forest regressor"
    init_time = time.time()
    rf1.fit(train_fea.drop(colToDropRF1, axis=1), train)
    print "RF1 done - elapsed time"+str((time.time() - init_time) / 60)
    init_time = time.time()
    rf2.fit(train_fea.drop(colToDropRF2, axis=1), train)
    print "RF2 done - elapsed time"+str((time.time() - init_time) / 60)
    ## DUMP MODELS
    if dumpModels:
        joblib.dump(rf1,'Models\\rf1_final.pk1')
        joblib.dump(rf2,'Models\\rf2_final.pk1')
else:
    print "loading serialized model - RF"
    rf1 = joblib.load('Models\\rf1_final.pk1')
    rf2 = joblib.load('Models\\rf2_final.pk1')    

### GRADIENT BOOSTING REGRESSORS - set trainGB_models for switch training on\off
gb1 = GradientBoostingRegressor(n_estimators=400,max_depth=8, random_state=9874, loss='huber')
gb2 = GradientBoostingRegressor(n_estimators=400,max_depth=8, random_state=9874, loss='huber')

if trainGB_models:
    print "fitting gradient boosting regressor"
    init_time = time.time()
    gb1.fit(train_fea.drop(colToDropGB1, axis=1), train)
    print "GB1 done - elapsed time"+str((time.time() - init_time) / 60)
    init_time = time.time()
    gb2.fit(train_fea.drop(colToDropGB2, axis=1), train)
    print "GB2 done - elapsed time"+str((time.time() - init_time) / 60)
    ## DUMP MODELS
    if dumpModels:
        joblib.dump(gb1,'Models\\gb1_final.pk1')
        joblib.dump(gb2,'Models\\gb2_final.pk1')
else:
    print "loading serialized model - GB"
    gb1 = joblib.load('Models\\gb1_final.pk1')
    gb2 = joblib.load('Models\\gb2_final.pk1')
    
print "elapsed time"+str((time.time() - tot_init_time) / 60)

## PREDICTION
if make_prediction:
    pred1_rf = rf1.predict(test_fea.drop(colToDropRF1, axis=1))
    pred2_rf = rf2.predict(test_fea.drop(colToDropRF2, axis=1))
    pred1_gb = gb1.predict(test_fea.drop(colToDropGB1, axis=1))
    pred2_gb = gb2.predict(test_fea.drop(colToDropGB2, axis=1))    
    
    ## blended weighted results given validation set results - RISKYYYYYYYYYYY!!!111
    ## GB had 0.2256 // RF had 0.2337 >>> 0.2239 combined with 70/30 ratio
    ## CHE DIO CE LA MANDI BUONA!!! :)
    pred_FINAL = ((pred1_rf+pred2_rf)*0.15 + 0.35*(pred1_gb+pred2_gb))
        
    print "printing submission to file"
    test['SalePrice'] = pred_FINAL
    test[['SalesID', 'SalePrice']].to_csv('current_prediction.csv', index=False)




