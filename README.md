Fast-Iron
=========
Kaggle Blue Book for Bulldozers Competition

Windows 7 64bit on Intel QuadCore with 12GB RAM
Python 2.7 with Pandas, Numpy ,Scikit-Learn 0.13.1

##How to train your model
###How to make predictions on a new test set.
Before train make predictions, data need to be pre-processed, step below:
1) Place the training, appendix and test data in the Data folder
2) Edit prepare_data.py and change the following line with names of training, appendix and test data
      trainData = "Data\\TrainAndValid.csv"
      testData = "Data\\Test.csv"
      appendixData = "Data\\Machine_Appendix.csv"
3) Run the script. This will create four files in DataProcessed.
   This step take about 10-15 minutes depending on machine and file sizes.
   Can be incredibly optimized on request to just few minutes (seconds?) for the test set.

### PREDICT on Test.csv data
Simply run train_and_predict.py will create the output named current_prediction.csv
train_and_predict.py is already set to run to recreate the output. gradient boosting regressor
are serialized and trained. random forest need to be re-trained (too big to attach).
Training the random forests takes 102 minutes.

### TRAIN on new data
1) Edit train_and_predict.py
To train GB models change trainGB_models to True
To train RF models change trainRF_models to True
To save the models, change dumpModels to True
