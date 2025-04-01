import os
from os import sched_getaffinity
from datetime import date, datetime
from numba import jit
from joblib import load, dump
import numpy
from pandas import read_csv, DataFrame, isnull
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, multilabel_confusion_matrix, confusion_matrix, classification_report, roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve, auc, PrecisionRecallDisplay
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sbs
sbs.set()

"""
Class: input and output of data
Parameters: phenotype - phenotype of interest
"""
class IO:

    def __init__(self, phenotype, hypothetical = False):
        
        self.phenotype = phenotype
        self.data_folder = './results/'+self.phenotype+'/data/'
        self.hypothetical = hypothetical
        
        if hasattr(self, 'save_folder') == False:
            results_folder = './results/'+self.phenotype + '/'
            _id = '#'+ datetime.now().strftime("%Y-%m-%d_%H:%M") +'/'
            self.save_folder = results_folder + _id
            os.makedirs(self.save_folder, exist_ok=False)
    
    def LoadOGcolumns(self):
        if self.hypothetical == True:
            self.OG = load( self.save_folder+'OGColumns_hypothetical.joblib')
        else:
            self.OG = load( self.data_folder+'OGColumns.joblib')
    
    def WriteLog(self, text):

        with open(self.save_folder+'log.txt', 'a+') as file:
            file.write(str(text)+'\n')

    def Save(self, file, file_name):
        
        while os.path.isfile(self.save_folder + file_name + '.joblib'):
            file_name = file_name + '_'
        dump(file, self.save_folder + file_name + '.joblib', compress = 5)
        
    def SavePlot(self, plot, file_name):
        
        while os.path.isfile(self.save_folder + file_name + '.png'):
            file_name = file_name + '_'
        plot.savefig( self.save_folder + file_name + '.png', bbox_inches="tight", dpi=200, format='png' )

    @jit
    def GetX(self):

        if self.hypothetical == True:
            x = load( self.data_folder + 'data_hypothetical.joblib', mmap_mode='r+' )
        else:
            x = load( self.data_folder + 'data.joblib', mmap_mode='r+' )
        x = numpy.asarray( x, dtype=numpy.float16 )
        x = numpy.nan_to_num(x, nan=0)

        return x
    
    @jit
    def GetY(self, processY = False):

        y = read_csv(self.data_folder + 'metadata.csv')
        
        if processY == True:
            y = self.ProcessY(y)
        return y
        
    @jit
    def ProcessY(self,y):
        y = numpy.array(y)[:,5:] # slicing to exclude first 5 columns (files name and taxonomy) and leave only phenotypes
        if self.phenotype in ['Temperature','pH','Salt conc.']:
            y = numpy.asarray([[j for j in i] for i in y], dtype=numpy.float16)
        return y
    
    @jit
    def ContinuousToBinary(self,x):
        x = DataFrame(x.tolist())
        x = x.ge(0.1).astype(int)
        return x.to_numpy()
        
    @jit
    def Labelize(self, y):
        labelizer = MultiLabelBinarizer()
        y = [[i] for i in y]
        y = labelize.fit_transform(y)
        return y, labelizer
        
    @jit
    def SplitData(self, x, y, test_size):
        if self.phenotype in ["nitrogen_fixation",'nitrate_reduction']:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, stratify=self.ProcessY(y))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
        
        train_genomes = numpy.array(y_train)[:,:1] # list of genome files
        self.Save(train_genomes, 'train_genomes')
        
        test_genomes = numpy.array(y_test)[:,:1] # list of genome files
        self.Save(test_genomes, 'test_genomes')
        
        y_test = self.ProcessY(y_test)
        y_train = self.ProcessY(y_train)
        
        self.Save(y_train, 'train_results')
        self.Save(y_test, 'test_results')
        
        return x_train, y_train, x_test, y_test
    
    @jit
    def GetAlreadySplitData(self, id_code):

        model_folder = 'results/'+ self.phenotype + '/' + id_code + '/'
        
        os.system('cp '+model_folder+'train_genomes.joblib '+self.save_folder+'train_genomes.joblib')
        x_train = load(model_folder+'train_genomes.joblib')
        
        os.system('cp '+model_folder+'train_results.joblib '+self.save_folder+'train_results.joblib')
        y_train = numpy.array(load(model_folder+'train_results.joblib'))
        
        os.system('cp '+model_folder+'test_genomes.joblib '+self.save_folder+'test_genomes.joblib')
        x_test = load(model_folder+'test_genomes.joblib')
        
        os.system('cp '+model_folder+'test_results.joblib '+self.save_folder+'test_results.joblib')
        y_test = numpy.array(load(model_folder+'test_results.joblib'))

        x_train2 = []
        for genome in x_train:
            genome = load(genome[0])
            x_train2.append( genome + [0]*( len(self.OG) - len(genome) ) )

        x_test2 = []
        for genome in x_test:
            genome = load(genome[0])
            genome = genome + [0]*( len(self.OG) - len( genome ))
            x_test2.append( genome )

        return numpy.array(x_train2), y_train, numpy.array(x_test2), y_test

    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    # https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
    @jit
    def MDS_plot(self, x, y):
        plt.figure( 'mds')
        mds = MDS(n_components=2, metric= False, 
                  n_init=20, max_iter=500,
                  eps=0.00001, n_jobs=len(sched_getaffinity(0)), 
                  random_state=1, dissimilarity='euclidean')
        x_mds = mds.fit_transform(x)

        y = self.ProcessY(y)
        y = self.Interpretate_output(y)
        if self.phenotype in ['sporulation','nitrogen_fixation','nitrate_reduction']:
            colors = ['blue' if i[0] == 'yes' else 'red' for i in y]
        elif self.phenotype in ['pH']:
            colors = []
            for i in y:
                print(i) # Don't know why these PRINTS are needed for the colors to work properly. Don't remove!
                print(type(i))
                print(i[0])
                print(type(i[0]))
                if int(i[0]) < 5:
                    colors.append('red')
                elif int(i[0]) > 5 and int(i[1]) < 9:
                    colors.append('green')
                else:
                    colors.append('blue')
        plt.scatter(x_mds[:,0], x_mds[:,1],c=colors)
        self.SavePlot(plt, 'MDS')
        
        stress = mds.stress_
        return stress
    
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
        
        elif self.phenotype == 'metabolism':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes':
                    y_interpretated.append(['facultative of microaerobic'])
                elif y[i][0] == 'yes' and y[i][1] == 'no':
                    y_interpretated.append(['aerobic'])
                elif y[i][0] == 'no' and y[i][1] == 'yes':
                    y_interpretated.append(['anaerobic'])
                elif y[i][0] == 'no' and y[i][1] == 'no':
                    y_interpretated.append(['none'])
        
        elif self.phenotype == 'Salt conc.':
            y_interpretated = numpy.asarray([[round(2*j/1000)/2 
                                              for j in i]
                                             for i in y])
            
        elif self.phenotype == 'pH':
            y_interpretated = numpy.asarray([[ round( (2*j/1000) + 2*7.0)/2 for j in i] for i in y])
            """
            y_interpretated = numpy.asarray([[if j > 0 round( (-2*log(j)) + 2*7.0)/2 
                                              else round( (-2*log(-j)) + 2*7.0)/2 
                                              for j in i] 
                                             for i in y])
            """
        elif self.phenotype == 'Temperature':
            y_interpretated = numpy.asarray([[round( 2*((i[0]/100) + 25.0)/2), 
                                              round( 2*((i[1]/100) + 35.0)/2) ] 
                                             for i in y])
        
        else:
            return y
        return y_interpretated

    @jit
    def Confusion_matrix(self, y_test, y_pred):
        if self.phenotype in ['range_tmp','range_salinity']: #multi-label phenotypes
            self.WriteLog( multilabel_confusion_matrix( y_test, y_pred ) )
        else:
            self.WriteLog( confusion_matrix( y_test, y_pred ) )

    @jit
    def Print_bacteria_with_wrong_predictions(self, model_name, y_true, y_pred):
        genomes = numpy.array(load(self.save_folder+'test_genomes.joblib'))

        file = open(self.save_folder + model_name + '_predictions.txt', 'w')
        for idx in range(len(genomes)):
            file.write(str(y_true[idx][0])+' '+str(y_pred[idx][0])+' '+str(genomes[idx])+'\n')
        
        return None
        
    @jit
    def Metrics_Classification(self, y_true, y_pred, y_pred_prob, model_name):
        y_true_interpretated = self.Interpretate_output(y_true)
        y_pred_interpretated = self.Interpretate_output(y_pred)
        
        self.Confusion_matrix(y_true_interpretated, y_pred_interpretated)
        self.WriteLog( classification_report( y_true_interpretated, y_pred_interpretated ) )
        if model_name in ['LGBMClassifier','XGBClassifier']:
            self.ROC(y_true, y_pred_prob, model_name)
        else:
            self.ROC(y_true, y_pred_prob[0], model_name)
        self.Print_bacteria_with_wrong_predictions(model_name,y_true,y_pred)

    def ROC(self, y_true, y_pred_prob, name):
        #if len(y_pred_prob) == :
        if y_pred_prob is None:
            self.WriteLog( "Method doesn't have probability output." )
        else:
            for idx in range(0,len(y_true[0])):
                y_t = [[0,1] if i[idx] == 'yes' else [1,0] for i in y_true]
                y_t = [row[1] for row in y_t]
                y_p = [row[1] for row in y_pred_prob]
                self.WriteLog( 'ROC-AUC: %.2f' % roc_auc_score(y_t, y_p) )
                fpr, tpr, _ = roc_curve(y_t, y_p)
                plt.figure( str(fpr + tpr) )
                roc_display = RocCurveDisplay(fpr = fpr, tpr = tpr).plot()
                self.SavePlot(plt, name + str(idx) + '_ROC')
                
                precision, recall, _ = precision_recall_curve(y_t, y_p)
                self.WriteLog( 'Precision-Recall AUC: %.2f' % auc(recall, precision) )
                self.WriteLog('')
                plt.figure( str(precision + recall) )
                pr_display = PrecisionRecallDisplay.from_predictions(y_t, y_p).plot()
                self.SavePlot(plt, name + str(idx) + '_PR')
    
    @jit
    def Split_y_low_high(self, y_true, y_pred):
        y_true_low, y_true_hi = numpy.hsplit(y_true,[1])
        y_pred_low, y_pred_hi = numpy.hsplit(y_pred,[1])
        
        unpredicted_rows_low = numpy.argwhere(isnull(y_pred_low))[:,:1]
        unpredicted_rows_hi = numpy.argwhere(isnull(y_pred_hi))[:,:1]
        
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
        self.WriteLog('')
    
    @jit
    def Graph_Identity(self, y_true_hi, y_pred_hi, y_true_lo, y_pred_lo, name):
        plt.figure( name + '_identity' )
        plt.xlabel('Valor real')
        plt.ylabel('Predição')
        plt.scatter(y_true_hi, y_pred_hi, marker='.', color='blue', label='Limite superior')
        plt.scatter(y_true_lo, y_pred_lo, marker='.', color='red', label='Limite inferior')
        plt.legend(loc='upper right', bbox_to_anchor=(1.7, 0.0), fancybox=True, ncol=1)
        plt.axline((0, 0), slope=1) # identity line for reference
        
        identity_line = numpy.arange(0, 14, 0.5, dtype=numpy.float16)
        lower_error_band = [x - 0.5 for x in identity_line]
        higher_error_band = [x + 0.5 for x in identity_line]
        
        plt.fill_between(identity_line, lower_error_band, higher_error_band, alpha=0.2, 
                         #edgecolor='#1B2ACC', facecolor='#089FFF', 
                         linewidth=0, linestyle='dashdot', antialiased=True)
        plt.xlim(min(y_true_lo)-1, max(y_true_hi)+1)
        plt.ylim(min(y_true_lo)-1, max(y_true_hi)+1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        # Calculate the regression line
        a_hi, b_hi = numpy.polyfit(y_true_hi, y_pred_hi, 1)
        # Plot the regression line
        plt.plot(y_true_hi, a_hi*y_true_hi + b_hi, color='blue')
        
        # Calculate the regression line
        a_lo, b_lo = numpy.polyfit(y_true_lo, y_pred_lo, 1)
        # Plot the regression line
        plt.plot(y_true_lo, a_lo*y_true_lo + b_lo, color='red')
        
        self.SavePlot(plt, name + 'identity')
    
    @jit
    def Graph_Error_Hist(self, y_true_hi, y_pred_hi, y_true_lo, y_pred_lo, name):
        plt.figure( name + '_errors' )
        errors = numpy.concatenate( (numpy.subtract( y_pred_hi, y_true_hi ), numpy.subtract( y_pred_lo, y_true_lo ) ))
        plt.hist(errors, discrete = True)
        plt.xlabel('Erro')
        plt.ylabel('Quantidade de casos')
        self.SavePlot(plt, name + '_errors')