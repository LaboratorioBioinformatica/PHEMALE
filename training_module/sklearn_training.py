from data_io import DataIO_Classification, DataIO_Regression

from copy import deepcopy
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, LassoLars, ElasticNet, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, LogisticRegression, TweedieRegressor, SGDRegressor, Perceptron, PassiveAggressiveRegressor, QuantileRegressor, TheilSenRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')

"""
Class: training of sklearn models.
Observations: Can train classification, regression and multilabel models.
Parameters: phenotype - Phenotype to be trained and predicted
            classification_or_regression - inform if it's classification/multilabel or regression problem
            multioutput - boolean for multioutput data
"""
class Sklearn_Training:

    """
    Function: Finds optimal hyper-parameters of a sklearn model.
    Observations: Can deal with single outputs (classification or simple regression)
                  or multiple outputs (multilabels or multioutput regressions).
    Parameters: model - sklearn machine learning model to optimize
                params - space of parameters to explore in optimization
                multioutput_ensemble - boolean for models that need ensemble to be multioutput
    """
    @jit
    def GridSearch( self, model, params, multioutput_ensemble = False ):

        x_train = deepcopy(self.x_train)
        y_train = deepcopy(self.y_train)
        x_test = deepcopy(self.x_test)
        y_test = deepcopy(self.y_test)
        
        #For simple classification or simple regression or natively multiouput models
        if len( y_train[0] ) == 1 or multioutput_ensemble == True:
            y_train = numpy.ravel(y_train)
        
            HGS = HalvingGridSearchCV( estimator = model, param_grid=params, cv=3, factor=3, 
                                       n_jobs=-1, min_resources='exhaust')
            
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
                                       cv=3, 
                                       factor=3, 
                                       n_jobs=-1, 
                                       min_resources='exhaust')

            HGS.fit( x_train, y_train_ )

            self.dataIO.WriteLog( str( HGS.best_estimator_ ).split( '(' )[0] ) #write model name in log file
            self.dataIO.WriteLog( str( HGS.best_params_ ) ) #write best hyper-parameters in log file
            
            params = self.dataIO.ParseParameters( HGS.best_params_ )
            
            if self.classification_or_regression == 'classification'
                model = MultiOutputClassifier( model, n_jobs = -1 ).set_params(**params)
            
            elif self.classification_or_regression == 'regression'
                model = MultiOutputRegressor( model, n_jobs = -1 ).set_params(**params)
            
            model.fit( x_train, y_train )
            y_pred = model.predict( x_test )
            
        #self.dataIO.SaveModel(final_model, str( HGS.best_estimator_ ).split( '(' )[0])
        
        self.dataIO.Metrics( y_test, y_pred )
    
    """
    Function: Implements GridSearch in Sklearn models.
    Observations: If a sklearn model link has in its fit() method the description 
                  "y : array-like, shape = [n_samples] or [n_samples, n_outputs]"
                  it means it supports a 2-d array for targets (y). Meaning it is inherently multioutput.
    Parameters:
    """
    def Sklearn_Exploratory( self ):

        if self.classification_or_regression == 'classification':
            
            self.GridSearch(BernoulliNB(), 
                              {'alpha':[1.0e-10,0.1,0.5,1.0,5.0],
                               'binarize':[0.0,0.2,2.0,4.0], 
                               'fit_prior':[True], 
                               'class_prior':[None]},
                              multioutput_ensemble == self.multioutput )

            self.GridSearch(KNN(), 
                             {'n_neighbors':[2,4,7,10],
                              'weights':['uniform','distance'], 
                              'algorithm':['ball_tree','kd_tree','brute'], 
                              'leaf_size':[50,100,1000,10000],
                              'p':[1,2]})

            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l2','none'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':[None,'balanced'],
                               'solver':['newton-cg','lbfgs','sag'],
                               'max_iter':[20,50,100,500,1000,3000],
                               'multi_class':['ovr','multinomial']},
                              multioutput_ensemble == self.multioutput )

            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l2','l1'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':[None,'balanced'],
                               'solver':['liblinear'],
                               'max_iter':[20,50,100,500,1000,3000]}, 
                              multioutput_ensemble == self.multioutput )

            self.GridSearch(LogisticRegression(), 
                              {'penalty':['l1','l2','none','elasticnet'],
                               'tol':[0.001, 0.01, 0.05],
                               'C':[0.1,0.5,1.0,2.0,5.0],
                               'class_weight':[None,'balanced'],
                               'solver':['saga'],
                               'max_iter':[20,50,100,500,1000,3000],
                               'multi_class':['ovr','multinomial'],
                               'l1_ratio':[0.2,0.5,0.7]}, 
                              multioutput_ensemble == self.multioutput )

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
                              {'C':[0.5,1.0,3.0,5.0],
                               'kernel':['linear','poly','rbf','sigmoid'],
                               'degree':[1,2,3,4,5,6],
                               'coef0':[0.0, 0.01, 0.1, 1.0],
                               'tol':[0.001, 0.05],
                               'cache_size':[200000],
                               'class_weight':[None,'balanced'],
                               'shrinking':[True,False],
                               'decision_function_shape':['ovo', 'ovr']}, 
                              multioutput_ensemble == self.multioutput )

            self.GridSearch(GaussianProcessClassifier(),
                              {'max_iter_predict':[100,200,500],
                               'warm_start':[True, False],
                               'multi_class':['one_vs_rest','one_vs_one']} )

            self.GridSearch(NuSVC(), 
                              {'nu':[0.2,0.5,0.8],
                               'kernel':['linear','poly','rbf','sigmoid'],
                               'degree':[1,2,3,4,5,6],
                               'coef0':[0.0, 0.01, 0.1, 1.0],
                               'tol':[0.001, 0.01, 0.1],
                               'cache_size':[200000]} )

            self.GridSearch(RidgeClassifierCV(),
                              {'normalize':[True, False],
                               'class_weight':['balanced'],
                               'alphas':[(1.,2.,3.,4.,5.,10.)]})

            self.GridSearch(LogisticRegressionCV(),
                              {'Cs':[1,5,10,20], 
                               'dual':[False], 
                               'penalty':['l2','l1','elasticnet'], 
                               'solver':['newton-cg','lbfgs','liblinear','sag','saga'], 
                               'tol':[0.001, 0.01, 0.1], 
                               'max_iter':[100,200,500], 
                               'class_weight':['balanced'], 
                               'multi_class':['ovr'], 
                               'l1_ratios':[0.2,0.5,0.7]})

            self.GridSearch(RidgeClassifier(),
                              {'class_weight':['balanced'],
                               'alpha':[1.,2.,3.,4.,5.,10.],
                               'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                               'tol':[0.001, 0.01, 0.1],
                               'class_weight':['balanced'],
                               'max_iter':[100,200,500]})
    
        elif self.classification_or_regression == 'regression':
        
            self.GridSearch(LinearRegression(),
                              {'fit_intercept':[True]})

            self.GridSearch(Ridge(), 
                              {'alpha':[0.5,1.0,5.0,10.0,15.0,20.0],
                               'max_iter':[200,500,800,1000,2000,4000,8000,13000,20000],
                               'tol':[0.001,0.01,0.1,1.0],
                               'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga']})

            self.GridSearch(RandomForestRegressor(), 
                              {'n_estimators':[100,300,500,700,1000,5000,10000], 
                               'max_depth':[None,10,50,100,300],
                               'max_features':['auto','sqrt','log2'],
                               'bootstrap':['False','True'],
                               'ccp_alpha':[0.0,0.003,0.005,0.01,0.02,0.03,0.05,0.1,0.15,0.2]})

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

            self.GridSearch(OrthogonalMatchingPursuit(), 
                              {'tol':[0.01,0.1,1.0]})

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
                              {'kernel':['additive_chi2','chi2','linear','poly','polynomial','rbf','laplacian','sigmoid','cosine'],
                               'alpha':[1.0e-4,0.2,1.0,5.0],
                               'degree':[2,3,4,5,6,7],
                               'gamma':[0,1.0e-4,0.3,1.0,5.0],
                               'coef0':[0.0,0.03,0.1,1.0]})

            self.GridSearch(PLSRegression(), 
                              {'n_components':[5,10,20,100],
                               'max_iter':[100,500,2000],
                               'tol':[0.01,0.1,0.5]})

            self.GridSearch(KNeighborsRegressor(), 
                              {'n_neighbors':[2,5], 
                               'weights':['uniform','distance'], 
                               'algorithm':['ball_tree','kd_tree','brute'], 
                               'leaf_size':[50,100,1000,10000]})

            self.GridSearch(AdaBoostRegressor(), 
                              {'n_estimators':[20,50,100,500],
                               'learning_rate':[0False.01,0.1,0.5],
                               'loss':['linear','square','exponential']})

            self.GridSearch(SGDRegressor(), 
                              {'loss':['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'], 
                               'penalty':['l1','l2','elasticnet'], 
                               'alpha':[1.0e-3,0.2,1.0,5.0],
                               'l1_ratio':[0,0.01,0.1,0.5,1.0],
                               'fit_intercept':[True,False],
                               'max_iter':[200,500,2000],
                               'tol':[0.01,0.1,0.5],
                               'early_stopping':[True],
                               'validation_fraction':[0.2],
                               'eta0':[0.01,0.1,0.5],
                               'epsilon':[0.1,0.5,1.0],
                               'learning_rate':['optimal','adaptative'],
                               'power_t':[0.1,0.25,0.5,1.0,2.0]})

            self.GridSearch(LassoLars(), 
                              {'alpha':[1.0,3.0,5.0,10.0],
                               'max_iter':[200,500,2000], 
                               'eps':[2.2e-16,0.1,1.0],
                               'fit_path':[False],
                               'jitter':[None,0.1,0.5,1.0],
                               'positive':[False,True]})

            self.GridSearch(ARDRegression(), False
                              {'tol':[0.01,0.1,0.5],
                               'n_iter':[200,500,2000], 
                               'alpha_1':[1.0,0.1,1e-06],
                               'alpha_2':[1.0,1.0e-3,1e-06],
                               'lambda_1':[1.0,0.1,1e-06],
                               'lambda_2':[1.0,0.1,1e-06],
                               'compute_score':[True,False],
                               'fit_intercept':[True],
                               'normalize':[True,False],
                               'threshold_lambda':[10,100.0,1000.0,10000.0]})

            self.GridSearch(Lars(), 
                              {'n_nonzero_coefs':[numpy.inf,1000,10000],
                               'eps':[2.2e-16,0.1,1.0],
                               'fit_path':[False],
                               'jitter':[None,0.1,0.5,1.0]})

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

    def __init__(self, phenotype, classification_or_regression, multioutput = False):
        
        self.phenotype = phenotype
        self.classification_or_regression = classification_or_regression
        self.multioutput = multioutput
        
        self.dataIO = DataIO( self.phenotype, self.classification_or_regression )
        
        self.x_train, self.y_train, self.x_test, self.y_test, self.labelize = dataIO.GetTrainingData('data.joblib', splitTrainTest = True, labelize = False)
        
        self.Sklearn_Exploratory()