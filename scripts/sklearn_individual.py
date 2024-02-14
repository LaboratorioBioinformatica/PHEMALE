from os import sched_getaffinity, path
from joblib import dump, load
from numba import jit
import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, GradientBoostingRegressor

class Sklearn:

    def Feature_importance_SHAP(self, model, x ):
        
        OGs = load('./results/'+self.phenotype+'/data/OGColumns.joblib')
        
        for i in range(len(model.estimators_)):
            explainer = shap.KernelExplainer(model.estimators_[i])
            shap_values = explainer.shap_values(x)
            plt.figure( str(i) + '_SHAP' )
            shap.summary_plot(shap_values, feature_names=OGs, plot_type='bar', max_display = 10, show=False)
            plt.savefig(self.io.save_folder + self.model_name + str(i) + '_shap.png',dpi=300)
    
    def Feature_importance_intrinsic(self, model ):
        OGs = load('./results/'+self.phenotype+'/data/OGColumns.joblib')
        
        feature_importance = pd.DataFrame(index = OGs)
        
        for idx in range(len(model.estimators_)):
            feat = None
            try:
                feat = pd.DataFrame(model.estimators_[idx].feature_importances_, index = OGs, columns=['class_'+str(idx)])
            except:
                try:
                    features = model.estimators_[idx].coef_
                    if features[0][0][0].ndim == 0:
                        feat = pd.DataFrame(features[0][0], index = OGs, columns=['class_'+str(idx)])
                except IndexError:
                    try: 
                        if features[0][0].ndim == 0:
                            feat = pd.DataFrame(features[0], index = OGs, columns=['class_'+str(idx)])
                    except IndexError:
                        if features[0].ndim == 0:
                            feat = pd.DataFrame(features, index = OGs, columns=['class_'+str(idx)])
                        else:
                            self.io.WriteLog( 'No intrinsic direct way of infering feature importance in the model.' )
                            return
            if feat is not None:
                feature_importance = pd.concat([feature_importance, feat], axis=1)
            else:
                return
            
        #if self.phenotype in ['optimum_tmp', 'optimum_ph']:
        feature_importance = feature_importance.nlargest(10, feature_importance.columns.values.tolist(), keep='all')
        #else:
            #feature_importance = feature_importance[feature_importance.apply(lambda row: (abs(row) > 0.1).any(), axis=1)]

        if feature_importance.empty:
            self.io.WriteLog( 'No relevant features detected in the model.' )
        else:
            plt.figure( str(idx)+'_intrinsic' )
            feature_importance.plot(kind='bar')
            plt.tight_layout()

            n = 0
            fig_name = self.io.save_folder + self.model_name + '_intrinsic.png'
            while path.isfile(fig_name):
                n = n + 1
                fig_name = self.io.save_folder + self.model_name + str(n) + '_intrinsic.png'
            plt.savefig(fig_name, dpi=300)
        self.io.WriteLog( '' )

    """
    Function: Finds optimal hyper-parameters of a sklearn model.
    Observations: Can deal with single outputs (classification or simple regression)
                  or multiple outputs (multilabels or multioutput regressions).
    Parameters: model - sklearn machine learning model to optimize
                params - space of parameters to explore in optimization
                multioutput_ensemble - boolean for models that need ensemble to be multioutput
    """
    def gridsearch( self, model, params ):

        self.model_name = str(model).split( '(' )[0]
        y_train = self.y_train
        parameters = self.io.ParseParameters( params )
        
        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            models = MultiOutputRegressor( model )
        else:
            models = MultiOutputClassifier( model )

        #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            score = 'r2'
        else:
            score = 'roc_auc_ovo_weighted'
        
        GScv = GridSearchCV( estimator = models, param_grid = parameters, cv = 5, n_jobs = self.n_jobs, scoring=score)
        
        try:
            GScv.fit( self.x_train, y_train )
            self.io.WriteLog( self.model_name )
            # write best hyper-parameters for each model in log file
            for estimator in GScv.best_estimator_.estimators_:
                self.io.WriteLog( str( estimator.get_params() ) )
        
        except Exception as exception:
            self.io.WriteLog( str( model )+' failed to converge. Reason below: ' )
            self.io.WriteLog( exception )
            return None
        
        y_pred = GScv.best_estimator_.predict(self.x_test)

        if self.phenotype in ['optimum_tmp', 'optimum_ph']:
            self.io.Metrics_Regression(self.y_test, y_pred, self.model_name)
        else:
            try:
                y_pred_prob = GScv.best_estimator_.predict_proba(self.x_test)
            except Exception as exception:
                y_pred_prob = None
                self.io.WriteLog( str( model )+" couldn't predict probabilities. Reason below: " )
                self.io.WriteLog( exception )
            self.io.Metrics_Classification(self.y_test, y_pred, y_pred_prob, self.model_name)
            
        self.Feature_importance_intrinsic(GScv.best_estimator_)
            
    """
    Function: Implements GridSearch in Sklearn classification models.
    Observations: If a sklearn model link has in its fit() method the description 
                  "y : array-like, shape = [n_samples] or [n_samples, n_outputs]"
                  it means it supports a 2-d array for targets (y). Meaning it is inherently multioutput.
    """
    @jit
    def Exploratory_classification( self ):

        self.gridsearch(LogisticRegression(),
                          {'penalty':['l2','l1'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['liblinear'],
                           'max_iter':[50,100,200,500,1000,2000,5000]})

        self.gridsearch(LogisticRegression(), 
                          {'penalty':['l1','l2','none','elasticnet'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['saga'],
                           'max_iter':[50,100,500,1000,3000],
                           'multi_class':['ovr','multinomial'],
                           'l1_ratio':[0.2,0.5,0.7]})
        
        self.gridsearch(LogisticRegression(), 
                          {'penalty':['l2','none'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['newton-cg','lbfgs','sag'],
                           'max_iter':[50,100,500,1000,3000],
                           'multi_class':['ovr','multinomial']})

        self.gridsearch(KNN(), 
                         {'n_neighbors':[2,4,7,10],
                          'weights':['uniform','distance'], 
                          'algorithm':['ball_tree','kd_tree','brute'], 
                          'leaf_size':[50,100,500,1000,10000],
                          'p':[1,2]})

        self.gridsearch(RandomForestClassifier(), 
                          {'n_estimators':[100,300,500,700,1000,2000],
                           'criterion':['gini','entropy'],
                           'oob_score':[True,False],
                           'class_weight':['balanced','balanced_subsample'],
                           'max_depth':[None,10,50,100,300,500],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0.0,0.005,0.05,0.1,0.2]})

        self.gridsearch(SVC(), 
                          {'C':[0.2,0.5,1.0,3.0,5.0,10.0],
                           'kernel':['linear','poly','rbf','sigmoid'],
                           'degree':[1,2,3,4,5,6,7,8,9],
                           'coef0':[0.0, 0.01, 0.1, 1.0],
                           'tol':[0.001, 0.05],
                           'cache_size':[200000],
                           'class_weight':['balanced'],
                           'shrinking':[True,False],
                           'decision_function_shape':['ovo', 'ovr'],
                           #'probability':[True, False],
                           'probability':[True]})

        self.gridsearch(GaussianProcessClassifier(),
                          {'max_iter_predict':[100,200,500,1000,5000],
                           'warm_start':[True, False],
                           'multi_class':['one_vs_rest','one_vs_one']})

        self.gridsearch(RidgeClassifier(),
                          {'class_weight':['balanced'],
                           'alpha':[1.,2.,3.,4.,5.,10.],
                           'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                           'tol':[0.001, 0.01, 0.1],
                           'class_weight':['balanced'],
                           'max_iter':[50,100,500,1000,3000]})
        
        self.gridsearch(BernoulliNB(), 
                          {'alpha':[1.0e-10,0.1,0.5,1.0,5.0],
                           'binarize':[0.0,0.2,2.0,4.0], 
                           'fit_prior':[True], 
                           'class_prior':[None]})

    """
    Function: Implements GridSearch in Sklearn regression models.
    """
    @jit
    def Exploratory_regression( self ):
        self.gridsearch(LinearRegression(),
                          {'fit_intercept':[True]})

        self.gridsearch(PLSRegression(), 
                          {'n_components':[5,10,15,20,100,200,500,5000,10000],
                           'max_iter':[500,1000,3000,5000],
                           'tol':[0.001,0.005,0.01,0.1]})

        self.gridsearch(KNeighborsRegressor(),
                          {'n_neighbors':[2,5,7,10], 
                           'weights':['uniform','distance'], 
                           'algorithm':['ball_tree','kd_tree','brute'], 
                           'leaf_size':[50,100,1000,10000]})
        
        self.gridsearch(RandomForestRegressor(), 
                          {'n_estimators':[100,300,500,700,1000,5000,10000], 
                           'max_depth':[None,10,50,100,200],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0.0,0.005,0.01,0.05,0.1,0.2]})

        self.gridsearch(SVR(), 
                          {'kernel':['linear','poly','rbf','sigmoid'],
                           'degree':[2,3,4,5,6,7], 
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'coef0':[0.0, 0.01, 0.1, 1.0],
                           'tol':[0.001],
                           'cache_size':[200000]})

        self.gridsearch(Ridge(), 
                          {'alpha':[0.5,1.0,5.0,10.0,15.0,20.0],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.001,0.01,0.1,1.0],
                           'solver':['svd','cholesky','lsqr','sparse_cg','sag','saga']})

        self.gridsearch(ElasticNet(), 
                          {'alpha':[1.0,3.0,5.0,10.0],
                           'l1_ratio':[0.2,0.5,0.8],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,1.0],
                           'positive':[False,True]})

        self.gridsearch(AdaBoostRegressor(), 
                          {'n_estimators':[20,50,100,500],
                           'learning_rate':[0.01,0.1,0.5],
                           'loss':['linear','square','exponential']})

        self.gridsearch(TweedieRegressor(), 
                          {'power':[0,1,2,3],
                           'alpha':[1.0e-4,0.1,0.5,1.0,2.0,5.0],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,0.5]})

        self.gridsearch(PassiveAggressiveRegressor(), 
                          {'C':[0.1,0.5,1.0,2.0,5.0],
                           'fit_intercept':[True,False],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.01,0.1,0.5],
                           'early_stopping':[True],
                           'validation_fraction':[0.2,0.3]})

        self.gridsearch(QuantileRegressor(), 
                          {'fit_intercept':[True],
                           'quantile':[0.15,0.25,0.5,0.8,0.99], 
                           'alpha':[0,1.0e-3,0.1,1.0,5.0], 
                           'solver':['interior-point','highs-ds','highs-ipm','highs','revised simplex']})

        self.gridsearch(TheilSenRegressor(), 
                          {'fit_intercept':[True],
                           'max_subpopulation':[200,500,1000,5000],
                           'max_iter':[50,100,500,1000,3000],
                           'tol':[0.03,0.1,0.5]})

        self.gridsearch(KernelRidge(), 
                          {'kernel':['additive_chi2','chi2','linear','poly','polynomial','rbf',
                                     'laplacian','sigmoid','cosine'],
                           'alpha':[1.0e-4,0.2,1.0,5.0],
                           'degree':[2,3,4,5,6,7],
                           'gamma':[0,1.0e-4,0.3,1.0,5.0],
                           'coef0':[0.0,0.03,0.1,1.0]})

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
        
        self.model_name = None
        
        if phenotype in ['range_tmp', 'range_salinity', 'optimum_tmp', 'optimum_ph']:
            self.multioutput = True
        else:
            self.multioutput = False
        
        if phenotype in ['optimum_tmp', 'optimum_ph']:
            self.Exploratory_regression()
        else:
            self.Exploratory_classification()