import os
from datetime import date, datetime
from numba import jit
from joblib import load, dump
import numpy
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, multilabel_confusion_matrix, confusion_matrix, classification_report, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

"""
Class: input and output of data
Parameters: phenotype - phenotype of interest
"""
class IO:

    def __init__(self, phenotype):
        
        self.phenotype = phenotype
        self.data_folder = './results/'+self.phenotype+'/data/'
    
    def WriteLog(self, text):
        
        if hasattr(self, 'save_folder') == False:
            results_folder = './results/'+self.phenotype + '/'
            _id = '#'+ datetime.now().strftime("%Y-%m-%d_%H:%M") +'/'
            self.save_folder = results_folder + _id
            os.makedirs(self.save_folder, exist_ok=False)

        with open(self.save_folder+'log.txt', 'a+') as file:
            file.write(str(text)+'\n')

    def SaveModel(self, model, file_name):
        dump(model, self.save_folder + file_name + '.joblib')

    @jit
    def GetData(self):

        x = load( self.data_folder + 'data.joblib', mmap_mode='r+' )
        x = numpy.asarray( x, dtype=numpy.float16 )
        x = numpy.nan_to_num(x, nan=0)
        
        y = read_csv(self.data_folder + 'metadata.csv')
        y = numpy.array(y)[:,6:] # slicing to exclude first 6 columns (files name and taxonomy) and leave only phenotypes
        
        if self.phenotype in ['optimum_ph','optimum_tmp']:
            y = numpy.asarray([[j for j in i] for i in y], dtype=numpy.float16)

        return x, y
        
    @jit
    def Labelize(self, y):
        labelizer = MultiLabelBinarizer()
        y = [[i] for i in y]
        y = labelize.fit_transform(y)
        return y, labelizer
        
    @jit
    def SplitData(self, x, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
        return x_train, y_train, x_test, y_test

    @jit
    def ParseParameters(self, dictionary):
        return { 'estimator__' + str(key) : (transform(value) 
                 if isinstance(value, dict) else value) 
                 for key, value in dictionary.items() }
    
    """
    Function transforms output to something interpretable to humans
    """
    def Interpretate_output(self, y):
        
        if len(y) == 1: #For cases as PLSReg which outputs a list of the results.
            y = y[0]
        
        y_interpretated = []
        
        if self.phenotype == 'range_salinity':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes':
                    y_interpretated.append(['halotolerant'])
                elif y[i][0] == 'yes' and y[i][1] == 'no':
                    y_interpretated.append(['halophilic'])
                elif y[i][0] == 'no' and y[i][1] == 'yes':
                    y_interpretated.append(['non-halophilic'])
                elif y[i][0] == 'no' and y[i][1] == 'no':
                    y_interpretated.append(['none'])
        
        elif self.phenotype == 'range_tmp':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y_interpretated.append(['mesophilic, psychrophilic and thermophilic'])
                elif y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y_interpretated.append(['mesophilic and thermophilic'])
                elif y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y_interpretated.append(['mesophilic and psychrophilic'])
                elif y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y_interpretated.append(['psychrophilic and thermophilic'])
                if y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'no':
                    y_interpretated.append(['mesophilic'])
                if y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y_interpretated.append(['psychrophilic'])
                if y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y_interpretated.append(['thermophilic'])
                elif y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'no':
                    y_interpretated.append(['none'])
        
        elif self.phenotype == 'optimum_ph':
            y_interpretated = numpy.asarray([[ -numpy.log2(j/(10**3)) for j in i] for i in y], dtype=numpy.float16)
        
        else:
            y_interpretated = y

        return y_interpretated

    @jit
    def Confusion_matrix(self, y_test, y_pred):
        if self.phenotype in ['range_tmp','range_salinity']: #multi-label phenotypes
            self.WriteLog( multilabel_confusion_matrix( y_test, y_pred ) )
        else:
            self.WriteLog( confusion_matrix( y_test, y_pred ) )
    
    @jit
    def Metrics_Classification(self, y_true, y_pred, y_pred_prob, model_name):
        
        y_true_interpretated = self.Interpretate_output(y_true)
        y_pred_interpretated = self.Interpretate_output(y_pred)
        
        self.Confusion_matrix(y_true_interpretated, y_pred_interpretated)
        self.WriteLog( classification_report( y_true_interpretated, y_pred_interpretated ) )
        self.ROC(y_true, y_pred_prob, model_name)

    def ROC(self, y_true, y_pred_prob, name):
        
        print("y_true")
        print(y_true)
        y_true = [[0,1] if i[0] == 'yes' else [1,0] for i in y_true]
        
        if y_pred_prob is None:
            self.WriteLog( "Method doesn't have probability output." )
        else:
            print("y_true")
            print(y_true)
            print("y_pred_prob")
            y_pred_prob = y_pred_prob[0]
            print(y_pred_prob)
            for idx in range(0,len(y_true[0])):
                y_t = [row[idx] for row in y_true]
                y_p = [row[idx] for row in y_pred_prob]
                try:
                    self.WriteLog( 'AUC: ' + str( roc_auc_score(y_t, y_p) ) )
                    fpr, tpr, _ = roc_curve(y_t, y_p)
                    roc_display = RocCurveDisplay(fpr = fpr, tpr = tpr).plot()
                    plt.savefig( self.save_folder + name + str(idx) + '_ROC.png', bbox_inches="tight", dpi=600, format='png' )
                except ValueError:
                    self.WriteLog( 'Highly imbalanced results. Not possible to check AUC value.' )
    
    @jit
    def Split_y_low_high(self, y_true, y_pred):
        y_true_low, y_true_hi = numpy.hsplit(y_true,[1])
        y_pred_low, y_pred_hi = numpy.hsplit(y_pred,[1])
        
        unpredicted_rows_low = numpy.argwhere(numpy.isnan(y_pred_low))[:,:1]
        unpredicted_rows_hi = numpy.argwhere(numpy.isnan(y_pred_hi))[:,:1]
        
        y_true_low = numpy.delete(y_true_low, unpredicted_rows_low, axis=0)
        y_pred_low = numpy.delete(y_pred_low, unpredicted_rows_low, axis=0)
        y_true_hi = numpy.delete(y_true_hi, unpredicted_rows_hi, axis=0)
        y_pred_hi = numpy.delete(y_pred_hi, unpredicted_rows_hi, axis=0)

        y_true_low = numpy.ravel(y_true_low)
        y_true_hi = numpy.ravel(y_true_hi)
        y_pred_low = numpy.ravel(y_pred_low)
        y_pred_hi = numpy.ravel(y_pred_hi)
        
        return y_true_low, y_true_hi, y_pred_low, y_pred_hi, unpredicted_rows_low, unpredicted_rows_hi
        
    @jit
    def Metrics_Regression(self, y_true, y_pred, name):
        
        y_true = self.Interpretate_output(y_true)
        y_pred = self.Interpretate_output(y_pred)
        
        y_true_low, y_true_hi, y_pred_low, y_pred_hi, unpredicted_low, unpredicted_hi = self.Split_y_low_high(y_true, y_pred)
        
        r2_low = r2_score(y_true_low, y_pred_low)
        r2_high = r2_score(y_true_hi, y_pred_hi)

        self.WriteLog( 'Number of lower values not predicted: ' + str(len(unpredicted_low)))
        self.WriteLog( 'R2 score of lower values: %.2f' % r2_low )
        self.WriteLog( 'Number of higher values not predicted: ' + str(len(unpredicted_hi)))
        self.WriteLog( 'R2 score of higher values: %.2f' % r2_high )
        self.Graph_Identity(y_true_hi, y_pred_hi, y_true_low, y_pred_low, name)
        self.Graph_Error_Hist(y_true_hi, y_pred_hi, y_true_low, y_pred_low, name)
    
    @jit
    def Graph_Identity(self, y_true_hi, y_pred_hi, y_true_lo, y_pred_lo, name):
        plt.figure( name + '_identity' )
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.scatter(y_true_hi, y_pred_hi, marker='.', color='blue', label='Upper limit')
        plt.scatter(y_true_lo, y_pred_lo, marker='.', color='red', label='Lower limit')
        plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.0), fancybox=True, ncol=1)
        plt.axline((0, 0), slope=1) # identity line for reference
        plt.savefig( self.save_folder + name + '_identity.png', bbox_inches="tight", dpi=600, format='png' )
    
    @jit
    def Graph_Error_Hist(self, y_true_hi, y_pred_hi, y_true_lo, y_pred_lo, name):
        plt.figure( name + '_errors' )
        errors = numpy.concatenate( (numpy.subtract( y_pred_hi, y_true_hi ), numpy.subtract( y_pred_lo, y_true_lo ) ))
        plt.hist(errors)
        plt.xlabel('Error')
        plt.ylabel('Counts')
        plt.savefig( self.save_folder + name + '_errors.png', bbox_inches="tight", dpi=600, format='png' )