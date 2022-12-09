import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy
from joblib import load, dump
from numba import jit
from datetime.date import today
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score

import warnings
warnings.filterwarnings('ignore')

"""
Class: input and output of data
Parameters: phenotype - phenotype of interest
            classification_or_regression - if the data is numerical (regression) or label (classification)
"""
class DataIO:

    def __init__(self, phenotype, classification_or_regression):
        
        self.phenotype = phenotype
        self.phenotype_directory = './metadata/'+self.phenotype+'/'
        os.mkdir('./results/')
        self.results_directory = './results/'+self.phenotype+'/'
        os.mkdir(self.results_directory)
        self.classification_or_regression = classification_or_regression
    
    def WriteLog(self, text):
        
        if hasattr(self, 'trainingID') == False:
            self.results_ID_directory = self.results_directory+'#'+str( today() )+'/'
            os.mkdir(self.results_ID_directory)

        with open(self.results_ID_directory+'log.txt', 'a+') as file:
            file.write(str(text)+'\n')

    def SaveModel(self, model, file_name):
        dump(self.results_ID_directory+file_name+'.joblib', model)

    @jit
    def GetTrainingData(self, file, splitTrainTest = False, labelize = False):

        number_of_OG_columns = len(load(self.phenotype_directory+'OGColumns.joblib') )
        
        x = numpy.asarray( load( self.phenotype_directory + file ) )
        x, y = numpy.hsplit(x, [number_of_OG_columns])
        x = numpy.asarray( x, dtype=numpy.float16)
        x = numpy.nan_to_num(x, nan=0)

        if labelize == True:
            labelize = MultiLabelBinarizer()
            y = [[i] for i in y]
            y = labelize.fit_transform(y)

        if splitTrainTest != False:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=splitTrainTest)
            return x_train, y_train, x_test, y_test, labelize

        return x, y, labelize
        
    def ParseParameters(self, dictionary):
        return { "estimator__"+str(key) : (transform(value) 
                 if isinstance(value, dict) else value) 
                 for key, value in dictionary.items() }
    
    def LabelClassifier(self, y):
        
        if self.phenotype == 'range_salinity':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes':
                    y[i] = 'halotolerant'
                elif y[i][0] == 'yes' and y[i][1] == 'no':
                    y[i] = 'halophilic'
                elif y[i][0] == 'no' and y[i][1] == 'yes':
                    y[i] = 'non-halophilic'
                elif y[i][0] == 'no' and y[i][1] == 'no':
                    y[i] = 'no prediction'

        if self.phenotype == 'range_tmp':             
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y[i] = 'mesophilic and psychrophilic and thermophilic'
                elif y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y[i] = 'mesophilic and thermophilic'
                elif y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y[i] = 'mesophilic and psychrophilic'
                elif y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y[i] = 'psychrophilic and thermophilic'
                if y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'no':
                    y[i] = 'mesophilic'
                if y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y[i] = 'psychrophilic'
                if y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y[i] = 'thermophilic'
                elif y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'no':
                    y[i] = 'no prediction'
        
        # verificar se precisa desta linha
        #y = numpy.hsplit(y,[1])[0]
        
        self.WriteLog(y)
        
        return y
    
    @jit
    def Metrics(self, y_test, y_pred):

        self.classification_or_regression == 'classification':
            y_test = self.LabelClassifier(y_test)
            y_pred = self.LabelClassifier(y_pred)

            self.WriteLog( confusion_matrix( y_test, y_pred ) )
            self.WriteLog( classification_report( y_test, y_pred ) )
            
        self.classification_or_regression == 'regression':
            y_test_lower, y_test_higher = numpy.hsplit(y_test,[1])
            y_pred_lower, y_pred_higher = numpy.hsplit(y_pred,[1])
            
            r2_lower = r2_score(y_test_lower, y_pred_lower)
            r2_higher = r2_score(y_test_higher, y_pred_higher)
            
            self.WriteLog( 'R2 score of lower values'+ str(r2_lower) )
            self.WriteLog( 'R2 score of higher values'+ str(r2_higher) )

            """
            self.WriteLog('Mean absolute error ' + str(mean_absolute_error(y_true, y_pred)))
            self.WriteLog('Median absolute error '+ str(median_absolute_error(y_true, y_pred)))
            self.WriteLog('Mean squared error '+ str(mean_squared_error(y_true, y_pred)))
            self.WriteLog('Max error '+ str(max_error(y_true, y_pred)))
            self.WriteLog('Explained variance score '+ str(explained_variance_score(y_true, y_pred)))
            error = y_true - y_pred
            percentil = [5,25,50,75,95]
            percentil_value = numpy.percentile(error, percentil)
            for i in range(len(percentil)):
                self.WriteLog('Percentil '+str(percentil[i])+': '+str(percentil_value[i]))
            """

    """
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
    https://runtascience.hatenablog.com/entry/2020/07/06/Python%E3%83%BC%E3%82%A8%E3%83%A9%E3%83%BC%E3%83%90%E3%83%BC%E3%82%92fill_between%E3%81%A7%E8%A1%A8%E7%A4%BA
    https://stackoverflow.com/questions/45136420/filling-range-of-graph-in-matplotlib
    https://stackoverflow.com/questions/45208226/highlight-matplotlib-points-that-go-over-or-under-a-threshold-in-colors-based-on
    """
    @jit
    def Graphs(self, y_test, y_pred):
        pass