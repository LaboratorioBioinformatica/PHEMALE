import sys
from .data_classes import DataIO
from numba import jit
from copy import deepcopy
import numpy
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import warnings
warnings.filterwarnings('ignore')

#Sklearn models used for classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegressionCV

#Sklearn models used for regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, LassoLars, ElasticNet, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, LogisticRegression, TweedieRegressor, SGDRegressor, PassiveAggressiveRegressor, QuantileRegressor, TheilSenRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

"""
Class: training of sklearn models.
Observations: Can train classification, regression and multilabel models.
Parameters: phenotype - Phenotype to be trained and predicted
            classification_or_regression - inform if it's classification/multilabel or regression problem
            multioutput - boolean for multioutput data
"""
class Sklearn:

    """
    Function: Finds optimal hyper-parameters of a sklearn model.
    Observations: Can deal with single outputs (classification or simple regression)
                  or multiple outputs (multilabels or multioutput regressions).
    Parameters: model - sklearn machine learning model to optimize
                params - space of parameters to explore in optimization
                multioutput_ensemble - boolean for models that need ensemble to be multioutput
    """
    def GridSearch( self, model, params, multioutput_ensemble = False ):

        x_train = deepcopy(self.x_train)
        y_train = deepcopy(self.y_train)
        x_test = deepcopy(self.x_test)
        y_test = deepcopy(self.y_test)
        
        #For simple classification or simple regression or natively multiouput models
        if len( y_train[0] ) == 1 or multioutput_ensemble == True:
            y_train = numpy.ravel(y_train)
            y_test = numpy.ravel(y_test)
        
            HGS = HalvingGridSearchCV( estimator = model, 
                                       param_grid=params, 
                                       cv=6,
                                       factor=4,
                                       n_jobs=-1,
                                       min_resources='exhaust')
            
            HGS.fit( x_train, y_train )

            self.dataIO.WriteLog( str( HGS.best_estimator_ ).split( '(' )[0] )
            self.dataIO.WriteLog( str( HGS.best_params_ ) )
            
            y_pred = HGS.best_estimator_.predict(x_test)

        #For multioutput models not natively multiouput, meaning it needs the classes MultiOutputClassifier/Regressor
        elif len( y_train[0] ) > 1 and multioutput_ensemble == False:
            
            y_train_ = numpy.hsplit(y_train, len(y_train[0]))[0]
            y_train_ = numpy.ravel(y_train_)

            HGS = HalvingGridSearchCV( estimator = deepcopy(model),
                                       param_grid=params,
                                       cv=6,
                                       factor=4,
                                       n_jobs=-1, 
                                       min_resources='exhaust')

            HGS.fit( x_train, y_train_ )

            self.dataIO.WriteLog( str( HGS.best_estimator_ ).split( '(' )[0] ) #write model name in log file
            self.dataIO.WriteLog( str( HGS.best_params_ ) ) #write best hyper-parameters in log file

            params = self.dataIO.ParseParameters( HGS.best_params_ )

            if self.classification_or_regression == 'classification':
                model = MultiOutputClassifier( model, n_jobs = -1 ).set_params(**params)

            elif self.classification_or_regression == 'regression':
                model = MultiOutputRegressor( model, n_jobs = -1 ).set_params(**params)
            else:
                sys.exit('Wrong classification_or_regression input')

            model.fit( x_train, y_train )
            y_pred = model.predict( x_test )

        self.dataIO.Metrics( y_test, y_pred, str( HGS.best_estimator_ ).split( '(' )[0] )
        
        #self.dataIO.SaveModel(final_model, str( HGS.best_estimator_ ).split( '(' )[0])
    
    """
    Function: Implements GridSearch in Sklearn models.
    Observations: If a sklearn model link has in its fit() method the description 
                  "y : array-like, shape = [n_samples] or [n_samples, n_outputs]"
                  it means it supports a 2-d array for targets (y). Meaning it is inherently multioutput.
    Parameters:
    """
    @jit
    def Sklearn_Exploratory( self ):

        if self.classification_or_regression == 'classification':
            
            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l2','l1'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':['balanced'],
                               'solver':['liblinear'],
                               'max_iter':[20,50,100,500,1000,3000]} )
            
            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l1','l2','none','elasticnet'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':['balanced'],
                               'solver':['saga'],
                               'max_iter':[20,50,100,500,1000,3000],
                               'multi_class':['ovr','multinomial'],
                               'l1_ratio':[0.2,0.5,0.7]} )
            
            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l2','none'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':['balanced'],
                               'solver':['newton-cg','lbfgs','sag'],
                               'max_iter':[20,50,100,500,1000,3000],
                               'multi_class':['ovr','multinomial']} )
            
            self.GridSearch(KNN(), 
                             {'n_neighbors':[2,4,7,10],
                              'weights':['uniform','distance'], 
                              'algorithm':['ball_tree','kd_tree','brute'], 
                              'leaf_size':[50,100,1000,10000],
                              'p':[1,2]})

            self.GridSearch(LogisticRegressionCV(),
                              {'Cs':[1,5,10,20], 
                               'dual':[False], 
                               'penalty':['l2'], 
                               'solver':['newton-cg'], 
                               'tol':[0.001, 0.01, 0.1], 
                               'max_iter':[100,200,500], 
                               'class_weight':['balanced'], 
                               'multi_class':['ovr'], 
                               'l1_ratios':[0.2,0.5,0.7]})
<<<<<<< HEAD
            
=======

>>>>>>> b17d59e18d3be5947d2977478b910871adcc0169
            self.GridSearch(LogisticRegressionCV(),
                              {'Cs':[1,5,10,20], 
                               'dual':[False], 
                               'penalty':['l2','l1','none'], 
                               'solver':['liblinear','saga'], 
                               'tol':[0.001, 0.01, 0.1], 
                               'max_iter':[100,200,500], 
                               'class_weight':['balanced'], 
                               'multi_class':['ovr'], 
                               'l1_ratios':[0.2,0.5,0.7]})
            
            self.GridSearch(LogisticRegressionCV(),
                              {'Cs':[1,5,10,20], 
                               'dual':[False], 
                               'penalty':['l2','none'], 
                               'solver':['sag'], 
                               'tol':[0.001, 0.01, 0.1], 
                               'max_iter':[100,200,500], 
                               'class_weight':['balanced'], 
                               'multi_class':['ovr'], 
                               'l1_ratios':[0.2,0.5,0.7]})
            
            self.GridSearch(LogisticRegressionCV(),
                              {'Cs':[1,5,10,20], 
                               'dual':[False], 
                               'penalty':['elasticnet'], 
                               'solver':['saga'], 
                               'tol':[0.001, 0.01, 0.1], 
                               'max_iter':[100,200,500], 
                               'class_weight':['balanced'], 
                               'multi_class':['ovr'], 
                               'l1_ratios':[0.2,0.5,0.7]})

            self.GridSearch(RandomForestClassifier(), 
                              {'n_estimators':[100,300,500,700,1000,5000,10000],
                               'criterion':['gini','entropy'],
                               'oob_score':[True,False],
                               'class_weight':['balanced','balanced_subsample'],
                               'max_depth':[None,10,50,100,300],
                               'max_features':['auto','sqrt','log2'],
                               'bootstrap':['False','True'],
                               'ccp_alpha':[0.0,0.005,0.01,0.03,0.05,0.1,0.2]})

            self.GridSearch(SVC(), 
                              {'C':[0.2,0.5,1.0,3.0,5.0,10.0],
                               'kernel':['linear','poly','rbf','sigmoid'],
                               'degree':[1,2,3,4,5,6,7,8,9],
                               'coef0':[0.0, 0.01, 0.1, 1.0],
                               'tol':[0.001, 0.05],
                               'cache_size':[200000],
                               'class_weight':['balanced'],
                               'shrinking':[True,False],
                               'decision_function_shape':['ovo', 'ovr']}, 
                              multioutput_ensemble = self.multioutput )

            self.GridSearch(GaussianProcessClassifier(),
                              {'max_iter_predict':[100,200,500],
                               'warm_start':[True, False],
                               'multi_class':['one_vs_rest','one_vs_one']} )

            self.GridSearch(RidgeClassifier(),
                              {'class_weight':['balanced'],
                               'alpha':[1.,2.,3.,4.,5.,10.],
                               'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                               'tol':[0.001, 0.01, 0.1],
                               'class_weight':['balanced'],
                               'max_iter':[100,200,500]})
            
            self.GridSearch(BernoulliNB(), 
                              {'alpha':[1.0e-10,0.1,0.5,1.0,5.0],
                               'binarize':[0.0,0.2,2.0,4.0], 
                               'fit_prior':[True], 
                               'class_prior':[None]},
                              multioutput_ensemble = self.multioutput )
    
        elif self.classification_or_regression == 'regression':

            
            self.GridSearch(LinearRegression(),
                              {'fit_intercept':[True]})
            
            self.GridSearch(PLSRegression(), 
                              {'n_components':[5,10,20,100],
                               'max_iter':[100,500,2000],
                               'tol':[0.01,0.1,0.5]})
            
            self.GridSearch(KNeighborsRegressor(),
                              {'n_neighbors':[2,5], 
                               'weights':['uniform','distance'], 
                               'algorithm':['ball_tree','kd_tree','brute'], 
                               'leaf_size':[50,100,1000,10000]})
            
            self.GridSearch(LassoLars(), 
                              {'alpha':[1.0,3.0,5.0,10.0],
                               'max_iter':[200,500,2000], 
                               'eps':[2.2e-16,0.1,1.0],
                               'fit_path':[False],
                               'jitter':[None,0.1,0.5,1.0],
                               'positive':[False,True]})
            
            self.GridSearch(OrthogonalMatchingPursuit(), 
                              {'tol':[0.01,0.1,1.0]})
            
            self.GridSearch(Ridge(), 
                              {'alpha':[0.5,1.0,5.0,10.0,15.0,20.0],
                               'max_iter':[200,500,800,1000,2000,4000,8000,13000,20000],
                               'tol':[0.001,0.01,0.1,1.0],
                               'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga']})

            self.GridSearch(SVR(), 
                              {'kernel':['linear','poly','rbf','sigmoid'],
                               'degree':[2,3,4,5,6,7], 
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'coef0':[0.0, 0.01, 0.1, 1.0],
                               'tol':[0.001],
                               'cache_size':[200000]})

            self.GridSearch(Lasso(), 
                              {'alpha':[1,3,5,10],
                               'max_iter':[200,500,2000],
                               'tol':[0.01,0.1,1.0],
                               'positive':[False,True]})
            
            self.GridSearch(ElasticNet(), 
                              {'alpha':[1.0,3.0,5.0,10.0],
                               'l1_ratio':[0.2,0.5,0.8],
                               'max_iter':[200,500,2000],
                               'tol':[0.01,0.1,1.0],
                               'positive':[False,True]})

            
            self.GridSearch(AdaBoostRegressor(), 
                              {'n_estimators':[20,50,100,500],
                               'learning_rate':[0.01,0.1,0.5],
                               'loss':['linear','square','exponential']})
            
            self.GridSearch(TweedieRegressor(), 
                              {'power':[0,1,2,3],
                               'alpha':[1.0e-4,0.1,0.5,1.0,2.0,5.0],
                               'max_iter':[200,500,2000],
                               'tol':[0.01,0.1,0.5]})

            self.GridSearch(PassiveAggressiveRegressor(), 
                              {'C':[0.1,0.5,1.0,2.0,5.0],
                               'fit_intercept':[True,False],
                               'max_iter':[200,500,2000],
                               'tol':[0.01,0.1,0.5],
                               'early_stopping':[True],
                               'validation_fraction':[0.2,0.3]})

            self.GridSearch(QuantileRegressor(), 
                              {'fit_intercept':[True],
                               'quantile':[0.15,0.25,0.5,0.8,0.99], 
                               'alpha':[0,1.0e-3,0.1,1.0,5.0], 
                               'solver':['interior-point','highs-ds','highs-ipm','highs','revised simplex']})

            self.GridSearch(TheilSenRegressor(), 
                              {'fit_intercept':[True],
                               'max_subpopulation':[200,500,1000,5000],
                               'max_iter':[200,500,2000],
                               'tol':[0.03,0.1,0.5]})

            self.GridSearch(KernelRidge(), 
                              {'kernel':['additive_chi2','chi2','linear','poly','polynomial','rbf',
                                         'laplacian','sigmoid','cosine'],
                               'alpha':[1.0e-4,0.2,1.0,5.0],
                               'degree':[2,3,4,5,6,7],
                               'gamma':[0,1.0e-4,0.3,1.0,5.0],
                               'coef0':[0.0,0.03,0.1,1.0]})
            
            self.GridSearch(RandomForestRegressor(), 
                              {'n_estimators':[100,300,500,700,1000,5000,10000], 
                               'max_depth':[None,10,50,100,200],
                               'max_features':['auto','sqrt','log2'],
                               'bootstrap':['False','True'],
                               'ccp_alpha':[0.0,0.003,0.005,0.01,0.03,0.05,0.1,0.2]})
            
            """
            # removed due to being too slow (+744 hours of training and no convergence)
            self.GridSearch(SGDRegressor(), 
                              {'loss':['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'], 
                               'penalty':['l1','l2','elasticnet'], 
                               'alpha':[1.0e-3,0.1,0.5,1.0],
                               'l1_ratio':[0,0.3,1.0],
                               'fit_intercept':[True,False],
                               'max_iter':[500,1000],
                               'tol':[0.2,0.5,1.0,2.0],
                               'early_stopping':[True],
                               'validation_fraction':[0.2],
                               'eta0':[0.02,0.5],
                               'epsilon':[0.1,0.5,1.0],
                               'learning_rate':['optimal','adaptative']})
                               
            # needs lots of memory for training +800gb
            self.GridSearch(ARDRegression(),
                              {'tol':[0.01,0.1,0.5],
                               'n_iter':[200,500,2000], 
                               'alpha_1':[1.0,0.1,1e-06],
                               'alpha_2':[1.0,1.0e-3,1e-06],
                               'lambda_1':[1.0,0.1,1e-06],
                               'lambda_2':[1.0,0.1,1e-06],
                               'compute_score':[True,False],
                               'fit_intercept':[True],
                               'threshold_lambda':[10,100.0,1000.0,10000.0]})
                               
            # needs lots of memory for training +800gb
            self.GridSearch(Lars(), 
                              {'n_nonzero_coefs':[numpy.inf,1000,10000],
                               'eps':[2.2e-16,0.1,1.0],
                               'fit_path':[False],
                               'jitter':[None,0.1,0.5,1.0]})
            
            # needs lots of memory for training +800gb                   
            self.GridSearch(BayesianRidge(), 
                              {'tol':[0.01,0.1,0.5],
                               'n_iter':[200,500,2000], 
                               'alpha_1':[1.0,1.0e-2,1e-06],
                               'alpha_2':[1.0,1.0e-3,1e-06],
                               'lambda_1':[1.0,1.0e-2,1e-06],
                               'lambda_2':[1.0,1.0e-2,1e-06],
                               'alpha_init':[1.0,1.0e-2,1e-06],
                               'lambda_init':[1.0,1.0e-2,1e-06],
                               'compute_score':[True,False],
                               'fit_intercept':[True],
                               'normalize':[True,False]})
            """

    def __init__(self, phenotype, classification_or_regression, multioutput = False):
        
        self.phenotype = phenotype
        self.classification_or_regression = classification_or_regression
        self.multioutput = multioutput
        
        self.dataIO = DataIO( phenotype, classification_or_regression )
        self.x_train, self.y_train, self.x_test, self.y_test, self.labelize = self.dataIO.GetTrainingData('data.joblib', splitTrainTest = 0.2, labelize = False)
        
        self.Sklearn_Exploratory()
        
#################################################################################################################################

from flaml import AutoML #https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML
from lightgbm import LGBMClassifier, LGBMRegressor

class LGBM:

    def __init__(self, phenotype, classification_or_regression):

        self.phenotype = phenotype
        self.classification_or_regression = classification_or_regression

        self.dataIO = DataIO( self.phenotype, self.classification_or_regression )
        
        x_train, y_train, x_test, y_test, labelize = self.dataIO.GetTrainingData('data.joblib', splitTrainTest = 0.2)
        
        best_config = self.GridSearch(x_train, y_train, x_test, y_test)
        
        x_train, y_train, labelize = self.dataIO.GetTrainingData('data.joblib')
        final_model = self.FinalModel(best_config, x_train, y_train)
        
        #self.dataIO.SaveModel(final_model, 'lgbm')

    def GridSearch(self, x_train, y_train, x_test, y_test):
        
        lgbm = AutoML()

        if self.classification_or_regression == 'classification':
            metric = 'log_loss'
            #metric = 'macro_f1'
        elif self.classification_or_regression == 'regression':
            metric = 'r2'
        
        settings = {'time_budget':240*60*60,
                    'task':self.classification_or_regression,
                    'estimator_list':['lgbm'],
                    'metric':metric,
                    'early_stop':'True'}

        y_train = numpy.hsplit(y_train, len(y_train[0]))[0]
        y_train = numpy.ravel(y_train)
        y_test = numpy.hsplit(y_test, len(y_test[0]))[0]
        y_test = numpy.ravel(y_test)
        
        lgbm.fit(X_train=x_train, y_train=y_train, **settings)

        self.dataIO.WriteLog(lgbm.model.estimator)

        y_pred = lgbm.predict(x_test)
        
        self.dataIO.Metrics(y_test, y_pred)
        
        self.dataIO.Graphs(y_test, y_pred, 'lgbm')

        return lgbm.best_config

    def FinalModel(self, config, x, y):

        if self.classification_or_regression == 'classification':
            lgbm = MultiOutputClassifier( LGBMClassifier(**config), n_jobs = -1 )

        elif self.classification_or_regression == 'regression':
            lgbm = MultiOutputRegressor( LGBMRegressor(**config), n_jobs = -1 )

        lgbm.fit(x, y)
        return lgbm
    
#################################################################################################################################
"""
import torch
import optuna

class ANN:

    def __init__(self, phenotype, classification_or_regression):

        self.phenotype = phenotype
        self.classification_or_regression = classification_or_regression

        self.dataIO = DataIO( self.phenotype, self.classification_or_regression )
        
        x_train, y_train, x_test, y_test, labelize = self.dataIO.GetTrainingData('data.joblib',labelize=True,splitTrainTest=0.2)
        
        best_config = self.GridSearch(x_train, y_train, x_test, y_test)
        
        x_train, y_train, labelize = self.dataIO.GetTrainingData('data.joblib')
        
        final_model = self.FinalModel(best_config, x_train, y_train)
        
        #self.dataIO.SaveModel(final_model, 'lgbm')

    def GridSearch(self, x_train, y_train, x_test, y_test):
        pass

    # 1. Define an objective function to be maximized.
    def objective(trial):

        # 2. Suggest values of the hyperparameters using a trial object.
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []

        in_features = 28 * 28
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, 10))
        layers.append(torch.nn.LogSoftmax(dim=1))
        model = torch.nn.Sequential(*layers).to(torch.device('gpu'))
        return accuracy

    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
"""

"""
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
import keras_tuner as kt        
from tensorflow import distribute
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.models import load_model

class ANN():
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
    except:
        print('Invalid device or cannot modify virtual devices once initialized.')

    x_train, y_train, x_val, y_val, LB = GetTrainingData(phenotypeFolder+'data.joblib',labelize = True, splitTrainTest = 0.3)
    batchSize = 20
    train_dataset = Dataset.from_tensor_slices((x_train,y_train)).batch(batchSize,drop_remainder=True)
    val_dataset = Dataset.from_tensor_slices((x_val,y_val)).batch(batchSize,drop_remainder=True)

    def build_model( hp = kt.HyperParameters ):
        model = keras.Sequential()
        model.add( keras.layers.Dropout( rate = hp.Choice(name='dropout', values=[0.0,0.15,0.3,0.4])))
        model.add( keras.layers.Dense(units = hp.Choice('1_neurons', values=[200,500,800,1000,2000,5000,10000]),
                                      activation = hp.Choice('1_activation', values=['relu','sigmoid','softmax','softplus',
                                                                                     'softsign','tanh','selu','elu']),
                                      kernel_constraint = keras.constraints.NonNeg(), bias_constraint = keras.constraints.NonNeg(), 
                                      kernel_regularizer = keras.regularizers.l2()))
        model.add( keras.layers.Dense(units=len(LB.classes_),
                                      activation= hp.Choice('output_activation',values = ['softmax','relu','sigmoid'])))
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp.Float("learning_rate", min_value=1e-4,
                                                                                 max_value=1e-1, sampling="log")),
                      loss = hp.Choice('loss',values=['categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence']),
                      metrics=['accuracy'])
        return model

    def ClassWeights( data ):
        from sklearn.utils import class_weight
        data = numpy.argmax(data, axis=1)
        data = class_weight.compute_class_weight(class_weight = 'balanced', classes = numpy.unique(data), y = data)
        data = {i : data[i] for i in range(len(data))}
        return data

    tuner = kt.BayesianOptimization(hypermodel = build_model,
                                    max_trials = 50,
                                    objective = 'val_accuracy',
                                    directory = phenotypeFolder+'TuningTests',
                                    distribution_strategy = distribute.MultiWorkerMirroredStrategy())

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, min_delta=5.0e-2)

    tuner.search(train_dataset, validation_data=val_dataset,
                 callbacks = [early_stop], class_weight = ClassWeights(y_train),
                 use_multiprocessing=True, epochs=200, verbose=2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    Log('Artificial Neural Network')
    Log("Dropout: {:.4f}".format(best_hps.get("dropout")))
    Log("\nNeurons 1st layer: {}".format(best_hps.get("1_neurons")))
    Log("\nActivation 1st layer: {}".format(best_hps.get("1_activation")))
    Log("\nActivation Output layer: {}".format(best_hps.get("output_activation")))
    Log("\nLearning rate: {:.4f}".format(best_hps.get("learning_rate")))
    Log("\nLoss: {}".format(best_hps.get("loss")))

    # Build the model with the optimal hyperparameters and train it to check best epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, callbacks=[early_stop], class_weight=ClassWeights(y_train), verbose=2, use_multiprocessing=True)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 10
    Log("\nBest epoch: {}".format(best_epoch))

    # Build the model with the optimal hyperparameters and train with best epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, epochs=best_epoch,
                        callbacks = [early_stop], verbose=2,
                        class_weight = ClassWeights(y_train),
                        use_multiprocessing=True)

    model.save(phenotypeFolder+proccessID+'ANN')

    #https://stackoverflow.com/questions/14645684/filtering-a-list-of-lists-in-a-smart-way

    y_pred_tmp = model.predict(x_val)
    y_pred = numpy.zeros_like(y_pred_tmp)
    for i in range(len(y_pred_tmp)):
        y_pred[i][numpy.argmax(y_pred_tmp[i])] = 1

    y_pred = LB.inverse_transform( y_pred )
    y_val = LB.inverse_transform(y_val)
    Metrics(y_val,y_pred)
"""
