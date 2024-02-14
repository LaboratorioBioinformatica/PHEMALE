from os import sched_getaffinity
from numba import jit
import numpy
from copy import deepcopy
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings
filterwarnings('ignore')

#Sklearn models used for classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import roc_auc_score

#Sklearn models used for regression
from sklearn.linear_model import LinearRegression, Ridge, LassoLars, ElasticNet, LogisticRegression, TweedieRegressor, PassiveAggressiveRegressor, QuantileRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor

class Sklearn:

    """
    Function: Finds optimal hyper-parameters of a sklearn model.
    Observations: Can deal with single outputs (classification or simple regression)
                  or multiple outputs (multilabels or multioutput regressions).
    Parameters: model - sklearn machine learning model to optimize
                params - space of parameters to explore in optimization
                multioutput_ensemble - boolean for models that need ensemble to be multioutput
    """
    def gridsearch( self, model, params, native_multioutput = False ):

        y_train = deepcopy(self.y_train)
        y_test = deepcopy(self.y_test)
        
        #Binary cases
        if len( y_train[0] ) == 1:
            y_train = numpy.ravel(y_train)
        #For multioutput models not natively multiouput
        elif len( y_train[0] ) > 1 and native_multioutput == False:
            needs_multioutput_model = True
            y_train = numpy.hsplit(y_train, len(y_train[0]))[0]
        if native_multioutput == True:
            needs_multioutput_model = False

        #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            score = 'r2'
        else:
            score = 'roc_auc'
        GScv = GridSearchCV( estimator = deepcopy(model), param_grid = params, cv = 5, n_jobs = self.n_jobs, scoring=score)

        try:
            GScv.fit( self.x_train, y_train )
            model_name = str(GScv.best_estimator_).split( '(' )[0]
            self.io.WriteLog( model_name ) #write model name in log file
            self.io.WriteLog( str( GScv.best_params_ ) ) #write best hyper-parameters in log file
        except Exception as exception:
            self.io.WriteLog( exception )
            self.io.WriteLog( str( model )+' failed to converge.' )
            return None

        if needs_multioutput_model is True:
            params = self.io.ParseParameters( GScv.best_params_ )
            if self.phenotype in ['optimum_tmp', 'optimum_ph']:
                model = MultiOutputRegressor( model, n_jobs = self.n_jobs ).set_params(**params)
            else:
                model = MultiOutputClassifier( model, n_jobs = self.n_jobs ).set_params(**params)
            model.fit( self.x_train, self.y_train ) 
            y_pred = model.predict( self.x_test )
        else:
            y_pred = GScv.best_estimator_.predict(self.x_test)

        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            self.io.Metrics_Regression(y_test, y_pred, model_name)
        else:
            self.io.Metrics_Classification(y_test, y_pred)

    ##########################################
    
    def newGridSearch( self, model, params, native_multioutput = False ):

        y_train = deepcopy(self.y_train)
        y_test = deepcopy(self.y_test)
        
        #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
                score = 'r2'
            else:
                score = 'roc_auc'
        
        #Binary cases
        if len( y_train[0] ) == 1:
            y_train = numpy.ravel(y_train)
        
        #For multioutput models
        elif len( y_train[0] ) > 1 and native_multioutput == False:
            params = self.io.ParseParameters( params )
            if self.phenotype in ['optimum_tmp', 'optimum_ph']:
                learner = MultiOutputRegressor( deepcopy(model) )
            else:
                learner = MultiOutputClassifier( deepcopy(model) )

        GScv = GridSearchCV( estimator = learner, param_grid = params, cv = 5, n_jobs = self.n_jobs, scoring=score)

        try:
            GScv.fit( self.x_train, y_train )
            self.io.WriteLog( str(model).split( '(' )[0] )
            self.io.WriteLog( str( GScv.best_params_ ) ) #write best hyper-parameters in log file
        except Exception as exception:
            self.io.WriteLog( exception )
            self.io.WriteLog( str( model )+' failed to converge.' )
            return None

        if needs_multioutput_model is True:
            params = self.io.ParseParameters( GScv.best_params_ )
            
            learner.fit( self.x_train, self.y_train ) 
            y_pred = learner.predict( self.x_test )
        else:
            y_pred = GScv.best_estimator_.predict(self.x_test)

        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            self.io.Metrics_Regression(y_test, y_pred, model_name)
        else:
            self.io.Metrics_Classification(y_test, y_pred)

    
    ##########################################
    
    from sklearn.ensemble import VotingClassifier
    models = {'lr1':LogisticRegression(),
              'lr2':LogisticRegression(),
              'lr3':LogisticRegression(),
              'knn':KNN(),
              'rf':RandomForestClassifier(),
              'svc':SVC(),
              'gpc':GaussianProcessClassifier,
              'rc':RidgeClassifier(),
              'bnb':BernoulliNB()}

    #Use the key for the classifier followed by __ and the attribute
    parameters = {'lr1__penalty':['l2','l1'],
                  'lr1__tol':[0.001, 0.01, 0.05],
                  'lr1__C':[0.1,0.5,1.0,2.0,5.0],
                  'lr1__class_weight':['balanced'],
                  'lr1__solver':['liblinear'],
                  'lr1__max_iter':[50,100,200,500,1000,2000,5000],
                  'lr2__penalty':['l1','l2','none','elasticnet'],
                  'lr2__tol':[0.001, 0.01, 0.05],
                  'lr2__C':[0.1,0.5,1.0,2.0,5.0],
                  'lr2__class_weight':['balanced'],
                  'lr2__solver':['saga'],
                  'lr2__max_iter':[50,100,500,1000,3000],
                  'lr2__l1_ratio':[0.2,0.5,0.7],
                  'lr3__penalty':['l2','none'],
                  'lr3__tol':[0.001, 0.01, 0.05],
                  'lr3__C':[0.1,0.5,1.0,2.0,5.0],
                  'lr3__class_weight':['balanced'],
                  'lr3__solver':['newton-cg','lbfgs','sag'],
                  'lr3__max_iter':[50,100,500,1000,3000],
                  'knn__n_neighbors':[2,4,7,10],
                  'knn__weights':['uniform','distance'], 
                  'knn__algorithm':['ball_tree','kd_tree','brute'], 
                  'knn__leaf_size':[50,100,500,1000,10000],
                  'knn__p':[1,2],
                  'rf__n_estimators':[100,300,500,700,1000,2000],
                  'rf__criterion':['gini','entropy'],
                  'rf__oob_score':[True,False],
                  'rf__class_weight':['balanced','balanced_subsample'],
                  'rf__max_depth':[None,10,50,100,300,500],
                  'rf__max_features':['auto','sqrt','log2'],
                  'rf__bootstrap':['False','True'],
                  'rf__ccp_alpha':[0.0,0.005,0.01,0.03,0.05,0.1,0.2],
                  'svc__C':[0.2,0.5,1.0,3.0,5.0,10.0],
                  'svc__kernel':['linear','poly','rbf','sigmoid'],
                  'svc__degree':[1,2,3,4,5,6,7,8,9],
                  'svc__coef0':[0.0, 0.01, 0.1, 1.0],
                  'svc__tol':[0.001, 0.05],
                  'svc__cache_size':[200000],
                  'svc__class_weight':['balanced'],
                  'svc__shrinking':[True,False],
                  'gpc__max_iter_predict':[100,200,500,1000,5000],
                  'gpc__warm_start':[True, False],
                  'rc__class_weight':['balanced'],
                  'rc__alpha':[1.,2.,3.,4.,5.,10.],
                  'rc__solver':['svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                  'rc__tol':[0.001, 0.01, 0.1],
                  'rc__class_weight':['balanced'],
                  'rc__max_iter':[50,100,500,1000,3000],
                  'bnb__alpha':[1.0e-10,0.1,0.5,1.0,5.0],
                  'bnb__binarize':[0.0,0.2,2.0,4.0],
                  'bnb__fit_prior':[True],
                  'bnb__class_prior':[None]}
    
    vot_classifier = VotingClassifier(estimators=[ (key, models[key]) for key in models.keys() ], voting='soft')
    grid = GridSearchCV(estimator=vot_classifier, param_grid=parameters, cv=5)
    grid.fit(X,y)
    print (grid.best_params_)
    
    """
    Function: Implements GridSearch in Sklearn regression models.
    """
    @jit
    def Exploratory_regression( self ):
        self.gridsearch(LinearRegression(),
                          {'fit_intercept':[True]}, native_multioutput = True)

        self.gridsearch(PLSRegression(), 
                          {'n_components':[5,10,15,20,100,200,500,5000,10000],
                           'max_iter':[500,1000,3000,5000],
                           'tol':[0.001,0.005,0.01,0.1]}, native_multioutput = True)

        self.gridsearch(KNeighborsRegressor(),
                          {'n_neighbors':[2,5,7,10], 
                           'weights':['uniform','distance'], 
                           'algorithm':['ball_tree','kd_tree','brute'], 
                           'leaf_size':[50,100,1000,10000]}, native_multioutput = True)
        
        self.gridsearch(RandomForestRegressor(), 
                          {'n_estimators':[100,300,500,700,1000,5000,10000], 
                           'max_depth':[None,10,50,100,200],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0.0,0.003,0.005,0.01,0.03,0.05,0.1,0.2]},
                           native_multioutput = True)

        self.gridsearch(SVR(), 
                          {'kernel':['linear','poly','rbf','sigmoid'],
                           'degree':[2,3,4,5,6,7], 
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'coef0':[0.0, 0.01, 0.1, 1.0],
                           'tol':[0.001],
                           'cache_size':[200000]}, native_multioutput = True)

        self.gridsearch(Ridge(), 
                          {'alpha':[0.5,1.0,5.0,10.0,15.0,20.0],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.001,0.01,0.1,1.0],
                           'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga']}, native_multioutput = True)

        self.gridsearch(ElasticNet(), 
                          {'alpha':[1.0,3.0,5.0,10.0],
                           'l1_ratio':[0.2,0.5,0.8],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,1.0],
                           'positive':[False,True]}, native_multioutput = True)

        self.gridsearch(AdaBoostRegressor(), 
                          {'n_estimators':[20,50,100,500],
                           'learning_rate':[0.01,0.1,0.5],
                           'loss':['linear','square','exponential']},
                           native_multioutput = True, native_multioutput = True)

        self.gridsearch(TweedieRegressor(), 
                          {'power':[0,1,2,3],
                           'alpha':[1.0e-4,0.1,0.5,1.0,2.0,5.0],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,0.5]}, native_multioutput = True)

        self.gridsearch(PassiveAggressiveRegressor(), 
                          {'C':[0.1,0.5,1.0,2.0,5.0],
                           'fit_intercept':[True,False],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,0.5],
                           'early_stopping':[True],
                           'validation_fraction':[0.2,0.3]}, native_multioutput = True)

        self.gridsearch(QuantileRegressor(), 
                          {'fit_intercept':[True],
                           'quantile':[0.15,0.25,0.5,0.8,0.99], 
                           'alpha':[0,1.0e-3,0.1,1.0,5.0], 
                           'solver':['interior-point','highs-ds','highs-ipm','highs','revised simplex']}, native_multioutput = True)

        self.gridsearch(TheilSenRegressor(), 
                          {'fit_intercept':[True],
                           'max_subpopulation':[200,500,1000,5000],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.03,0.1,0.5]}, native_multioutput = True)

        self.gridsearch(KernelRidge(), 
                          {'kernel':['additive_chi2','chi2','linear','poly','polynomial','rbf',
                                     'laplacian','sigmoid','cosine'],
                           'alpha':[1.0e-4,0.2,1.0,5.0],
                           'degree':[2,3,4,5,6,7],
                           'gamma':[0,1.0e-4,0.3,1.0,5.0],
                           'coef0':[0.0,0.03,0.1,1.0]}, native_multioutput = True)

    def Ensemble_Classifier(self):
        pass
    
    def Ensemble_Regressor(self, models):
        ensemble = StackingRegressor(estimators=[(str(model), model) for model in models])
        ensemble = ensemble.fit(X, y)
    
    def __init__(self, phenotype, x_train, y_train, x_test, y_test, io ):
        
        self.phenotype = phenotype
        self.n_jobs = len(sched_getaffinity(0))
        self.io = io
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        if phenotype in ['range_tmp', 'range_salinity', 'optimum_tmp', 'optimum_ph']:
            self.multioutput = True
        else:
            self.multioutput = False
        
        if phenotype in ['optimum_tmp', 'optimum_ph']:
            self.Exploratory_regression()
        else:
            self.Exploratory_classification()