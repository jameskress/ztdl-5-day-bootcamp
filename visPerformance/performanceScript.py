###################################################################################
# Script to run a series of modeling techniques on in situ rendering performance
# data. Each model will print several metrics fir accufracy from the training. 
#
# Date: 9/19/2018
###################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt

# Lists of files to work with
fileNames = ['raster_cuda_final.csv', 'raster_tbb_final.csv', 'ray_cuda_final.csv', 'ray_tbb_final.csv', 'vol_cuda_final.csv', 'vol_tbb_final.csv']

# Features to train and test each file
trainFeatures = [['actual pixels', 'objects'], 
                 ['actual pixels', 'objects', 'pixels'],
                 ['actual pixels', 'objects', 'pixels'],
                 ['actual pixels', 'objects', 'pixels'],
                 ['actual pixels', 'objects', 'pixels'],
                 ['actual pixels', 'objects', 'pixels'],
                ]
testFeature = ['Total Render']

#List of each linear model to run
classifiers = [
    #linear_model.SGDRegressor(),
    linear_model.Lasso(),
    linear_model.ElasticNet(),
    linear_model.Ridge(),
    linear_model.LinearRegression()
    ]


## Open a file
def openFile(fName):
    return pd.read_csv(fName)

count = 0
for file in fileNames:
    print("Step %i - Modeling data from file %s" % (count, file))
    f = openFile(file)
    trainingLabels = f[trainFeatures[count]]
    testingLabels = f[testFeature[0]]
    print("  Total training points: %i" % trainingLabels.size)
    print("  Total testing points: %i" % testingLabels.size)
    count = count + 1
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
    trainingLabels, testingLabels, test_size=.2, random_state=0)
    
    for item in classifiers:
        print("***********************************************************")
        print("\nTesting model %s" % item)
        clf = item
        clf.fit(X_train, y_train)
        print("coef values")
        print(clf.coef_)
        pred_test = clf.predict(X_test)
        
        # Print the plot showing differences between real and predicted
        #varience = 100 * ((y_test - pred_test)/y_test)
        #plt.figure(figsize = (14,6))
        #p1 = plt.scatter(pred_test, varience, c=X_test['objects'], cmap='plasma')
        #cb = plt.colorbar(p1)
        #cb.ax.set_title('Objects Rendered')
        #plt.grid()
        #plt.xlim(xmin=0)
        #plt.axhline(0, color='black')
        #plt.xlabel('Predicted Render Time')
        #plt.ylabel('Validation Error %')
        #plt.draw()
        #plt.show()
        
        print("\nModel score when evaluating the training data")
        print(clf.score(X_train, y_train))
        print("\nModel score when evaluating the testing data")
        print(clf.score(X_test, y_test))
        
        print("\nParamater importance to model")
        print(np.std(X_test, 0)*clf.coef_)

        #Do some cross validation with KFold
        scores = cross_val_score(clf, trainingLabels, testingLabels, cv=3)
        print("\nCross Validation Scores")
        print(scores)
        print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("===========================================================")
        print("===========================================================\n\n")
