from collections import defaultdict
from dateutil.parser import parse
import numpy as np
import pandas as pd
from pandas import concat
from pandas.stats.moments import expanding_mean, expanding_count, expanding_median
import datetime

## SET INPUT DATA FILE
## CONFIGURATION
trainData = "Data\\TrainAndValid.csv"
testData = "Data\\Test.csv"
appendixData = "Data\\Machine_Appendix.csv"
##

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)

def handler(grouped):
    se = grouped.set_index('saledate')['SalePrice'].sort_index()
    # se is the (ordered) time series of sales restricted to a single basket
    # we can now create a dataframe by combining different metrics
    conc = concat(
        {
            'MeanToDate': expanding_mean(se).shift(1).fillna(method='ffill'), # cumulative mean
            'MedianToDate': expanding_median(se).shift(1).fillna(method='ffill'), # cumulative mean
            'MaxToDate': se.cummax().shift(1).fillna(method='ffill'),         # cumulative max
            'MinToDate': se.cummin().shift(1).fillna(method='ffill'),         # cumulative max
            'PrevSale': se.shift(1).fillna(method='ffill'),          # previous sale
            'SaleCount': expanding_count(se) # cumulative count
        },
        axis=1
     )
    # bring back SalesID, needed for join
    se = grouped.set_index('saledate')['SalesID'].sort_index()
    conc['SalesID'] = se
    return conc

print "loading training and test set"
train = pd.read_csv(trainData, converters={"saledate": parse})
test = pd.read_csv(testData, converters={"saledate": parse})

print "loading & appending appendix data"
appendix = pd.read_csv(appendixData)
train = train.reset_index().merge(appendix, on='MachineID', suffixes=('', '_train'), how="left").set_index('index')
test = test.reset_index().merge(appendix, on='MachineID', suffixes=('', '_train'), how="left").set_index('index')

## dropping duplicates - dropping appendix duplicates
dupsToDrop = ['ModelID_train','fiModelDesc_train','fiBaseModel_train',
        'fiSecondaryDesc_train','fiModelSeries_train','fiModelDescriptor_train',
        'fiProductClassDesc_train','ProductGroup_train','ProductGroupDesc_train',
        ## description fields
        'ProductGroup','fiManufacturerID'
        ]
train = train.drop(dupsToDrop,axis=1)
test = test.drop(dupsToDrop,axis=1)

moments = train.append(test)
# test will be appended to train, to populate the moments for the test set
# this using the fill forward option for the NA in the dataset
# akward method, but in rush and new to python
print "creating moments for Models"
momentModels = moments.groupby('ModelID').apply(handler).reset_index()
print "creating moments for Machines"
momentMachines = moments.groupby('MachineID').apply(handler).reset_index()

## merging dataframes, including moments for models and machines
test = test.reset_index().merge(momentModels, on="SalesID", suffixes=('', '_Model'), how="left").set_index('index')
test = test.reset_index().merge(momentMachines, on="SalesID", suffixes=('', '_Machine'), how="left").set_index('index')
train = train.reset_index().merge(momentModels, on="SalesID", suffixes=('', '_Model'), how="left").set_index('index')
train = train.reset_index().merge(momentMachines, on="SalesID", suffixes=('', '_Machine'), how="left").set_index('index')

columns = set(train.columns)
columns.remove("SalesID")
columns.remove("SalePrice")
columns.remove("saledate")

## remove duplicate columns, due to join
columns.remove("ModelID_Model")
columns.remove("saledate_Model")
columns.remove("MachineID_Machine")
columns.remove("saledate_Machine")

train_fea = get_date_dataframe(train["saledate"])
test_fea = get_date_dataframe(test["saledate"])

print "creating feature"
for col in columns:
    if train[col].dtype == np.dtype('object'):
        s = np.unique(train[col].fillna(-1).values)
        mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
        train_fea = train_fea.join(train[col].map(mapping).fillna(-1))
        test_fea = test_fea.join(test[col].map(mapping).fillna(-1))
    else:
        train_fea = train_fea.join(train[col].fillna(0))
        test_fea = test_fea.join(test[col].fillna(0))

print "print dataframe to files"
## writing dataset in csv

train.to_csv("DataProcessed\\TrainMoments.csv")
train_fea.to_csv("DataProcessed\\TrainMoments_fea.csv")
test.to_csv("DataProcessed\\TestMoments.csv")
test_fea.to_csv("DataProcessed\\TestMoments_fea.csv")

