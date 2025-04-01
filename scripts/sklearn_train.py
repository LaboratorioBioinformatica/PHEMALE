#https://medium.com/@ia.bb/bem-vindo-cuml-adeus-scikit-learn-4d3498f5db15
#https://medium.com/rapids-ai/100x-faster-machine-learning-model-ensembling-with-rapids-cuml-and-scikit-learn-meta-estimators-d869788ee6b1

from os import sched_getaffinity, path
from numba import jit
from numpy import array
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings
filterwarnings('ignore')

#Sklearn models used for classification
from sklearn.linear_model import LogisticRegression
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
        for i in range(len(model.estimators_)):
            explainer = shap.KernelExplainer(model.estimators_[i])
            shap_values = explainer.shap_values(x)
            plt.figure( str(i) + '_SHAP' )
            shap.summary_plot(shap_values, feature_names=self.OG, plot_type='bar', max_display = 10, show=False)
            plt.savefig(self.io.save_folder + self.model_name + str(i) + '_shap.png',dpi=300)
    
    def Feature_importance_intrinsic(self, model, n_features = 10 ):
        feats = []
        for idx in range(len(model.estimators_)):
            try:
                feat_importance = pd.DataFrame(model.estimators_[idx].feature_importances_, index=self.OG)
            except:
                try:
                    features = model.estimators_[idx].coef_
                    if features[0].ndim == 0:
                        feat_importance = pd.DataFrame(features, index=self.OG)
                    elif features[0][0].ndim == 0:
                        feat_importance = pd.DataFrame(features[0], index=self.OG)
                    elif features[0][0][0].ndim == 0:
                        feat_importance = pd.DataFrame(features[0][0], index=self.OG)
                except:
                    self.io.WriteLog( 'No intrinsic direct way of infering feature importance in the model.\n' )
                    return
                        
            if self.model_name in ['RandomForestClassifier','RandomForestRegressor',
                                   'BernoulliNB',
                                   'XGBClassifier','XGBRegressor',
                                   'LGBMClassifier','LGBMRegressor']:
                feat_importance = feat_importance.loc[~(feat_importance == 0).all(axis=1)]
                feat_importance = feat_importance.nlargest(n_features,feat_importance.columns.values.tolist(),keep='first')
            elif self.phenotype in ['nitrogen_fixation', 'nitrate_reduction','range_salinity','sporulation','fermentation']:
                feats1 = feat_importance.nlargest(n_features, feat_importance.columns.values.tolist(),keep='all')
                feats2 = feat_importance.nsmallest(n_features, feat_importance.columns.values.tolist(), keep='all')
                feat_importance = pd.concat([feats1, feats2], axis=0)

            if feat_importance.empty:
                self.io.WriteLog( 'No relevant features detected in the model.\n' )
            else:
                fig_name=self.model_name+str(idx)+'_intrinsic'
                plt.figure( fig_name )
                feat_importance.plot(kind='bar',legend=False)
                plt.tight_layout()
                self.io.SavePlot(plt,fig_name)
                if n_features > 10:
                    feat_importance.to_csv(self.io.save_folder + fig_name+'.csv')
            feats.append(feat_importance.index.to_list())
        feats = [item for row in feats for item in row]
        return feats

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
        parameters = self.io.ParseParameters( params )
        
        if self.phenotype in ['pH', 'Salt conc.', 'Temperature']:
            models = MultiOutputRegressor( model )
        else:
            models = MultiOutputClassifier( model )

        #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self.phenotype in ['pH', 'Salt conc.', 'Temperature']:
            scores = ['mse',
                      'r2',
                      'max_error',
                      'neg_median_absolute_error',
                      'neg_mean_absolute_error',
                      'neg_mean_squared_error']
            score = scores[3]
        else:
            #score = 'roc_auc_ovo_weighted'
            score = 'f1_macro'
        
        self.io.WriteLog( 'Using as score: '+score )
        
        GScv = GridSearchCV(estimator=models, param_grid=parameters, cv=5, n_jobs=self.n_jobs, 
                            pre_dispatch=2*self.n_jobs, scoring=score, refit=True)

        try:
            GScv.fit( self.x_train, self.y_train )
            self.io.WriteLog( self.model_name )
            # write best hyper-parameters for each model in log file
            for estimator in GScv.best_estimator_.estimators_:
                self.io.WriteLog( str( estimator.get_params() ) )
        
        except Exception as exception:
            self.io.WriteLog( str( model )+' failed to converge. Reason below: ' )
            self.io.WriteLog( exception )
            return None
        
        y_pred = GScv.best_estimator_.predict(self.x_test)

        if self.phenotype in ['pH', 'Salt conc.', 'Temperature']:
            try:
                self.io.Metrics_Regression(array(self.y_test), y_pred, self.model_name)
            except Exception as exception:
                self.io.WriteLog( str( model )+" error. Reason below: " )
                self.io.WriteLog( exception )
        else:
            try:
                y_pred_prob = GScv.best_estimator_.predict_proba(self.x_test)
                self.io.Metrics_Classification(self.y_test, y_pred, y_pred_prob, self.model_name)
            except Exception as exception:
                y_pred_prob = None
                self.io.WriteLog( str( model )+" couldn't predict probabilities. Reason below: " )
                self.io.WriteLog( exception )
            
        self.Feature_importance_intrinsic(GScv.best_estimator_)
        
        self.io.Save(GScv.best_estimator_, self.model_name)
            
    """
    Function: Implements GridSearch in Sklearn classification models.
    Observations: If a sklearn model link has in its fit() method the description 
                  "y : array-like, shape = [n_samples] or [n_samples, n_outputs]"
                  it means it supports a 2-d array for targets (y). Meaning it is inherently multioutput.
    """
    @jit
    def Exploratory_classification( self ):

        self.gridsearch(LogisticRegression(),
                          {'penalty':['l1'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['liblinear','saga'],
                           'max_iter':[25,50,100,300,400,600]})

        self.gridsearch(LogisticRegression(), 
                          {'penalty':['elasticnet'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['saga'],
                           'max_iter':[25,50,100,300,600],
                           'multi_class':['ovr','multinomial'],
                           'l1_ratio':[0.2,0.5,0.7]})
        
        self.gridsearch(LogisticRegression(), 
                          {'penalty':['l2','none'],
                           'tol':[0.001, 0.01, 0.05],
                           'C':[0.1,0.5,1.0,2.0,5.0],
                           'class_weight':['balanced'],
                           'solver':['newton-cg','lbfgs','sag',
                                     'liblinear','saga'],
                           'max_iter':[25,50,100,300,600],
                           'multi_class':['ovr','multinomial']})

        self.gridsearch(KNN(), 
                         {'n_neighbors':[2,4,7,10,30,50,60],
                          'weights':['uniform','distance'], 
                          'algorithm':['ball_tree','kd_tree','brute'], 
                          'leaf_size':[5,10,50,100,500,1000,4000],
                          'p':[1,2,3,4,5,6,7,8,9,10],
                          'metric':['cityblock','cosine','euclidean','haversine','l1','l2','manhattan','nan_euclidean']
                         }
                       )

        self.gridsearch(RandomForestClassifier(), 
                          {'n_estimators':[100,300,500,700,1000,2000],
                           'criterion':['gini','entropy'],
                           'oob_score':[True,False],
                           'class_weight':['balanced','balanced_subsample'],
                           'max_depth':[None,10,50,100,300,600],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0.0,0.01,0.1,1.0]})
        
        self.gridsearch(SVC(), 
                          {'C':[.1,.5,1.0,3.0],
                           'kernel':['linear','rbf','sigmoid'],
                           #'degree':[2,3,4],
                           'coef0':[0.0, 0.1, 1.0, 3.0],
                           'tol':[0.001, 0.1, 1.0, 3.0],
                           'gamma':['scale','auto',0.1,1.0,3.0],
                           'cache_size':[100*1000],
                           'class_weight':['balanced'],
                           'shrinking':[True,False],
                           'decision_function_shape':['ovr'],
                           'probability':[True]})
        
        self.gridsearch(BernoulliNB(), 
                          {'alpha':[1.0e-10,0.1,0.5,1.0,5.0],
                           'binarize':[0.0,0.2,2.0,4.0], 
                           'fit_prior':[True], 
                           'class_prior':[None]})
        
    @jit
    def Exploratory_classification_hypothetical( self ):

        self.gridsearch(LogisticRegression(),
                          {'penalty':['l1','l2'],
                           'tol':[0.0001,0.001, 0.01],
                           'C':[0.1,0.5,1.0],
                           'class_weight':['balanced'],
                           'solver':['sag','saga'],
                           'max_iter':[25,50,75,100,600,800],
                           'multi_class':['multinomial'],
                           'l1_ratio':[None,0.01,0.05]})

        self.gridsearch(KNN(), 
                         {'n_neighbors':[3,4,5],
                          'weights':['uniform','distance'], 
                          'algorithm':['brute','ball_tree'], 
                          'leaf_size':[4,5,6],
                          'p':[1],
                          'metric':['cosine','cityblock']})

        self.gridsearch(RandomForestClassifier(), 
                          {'bootstrap':['True','False'],
                           'ccp_alpha':[0.01,0.02],
                           'class_weight':['balanced_subsample','balance'],
                           'criterion':['entropy'],
                           'max_features':['sqrt'],
                           'max_leaf_nodes':[None],
                           'max_samples': [None],
                           'min_impurity_decrease':[0.0],
                           'min_samples_leaf':[1,2],
                           'min_samples_split':[1,2],
                           'min_weight_fraction_leaf':[0.0],
                           'n_estimators':[80,100,120,250,300,400],
                           'oob_score':[True],
                           'max_depth':[None,50]})

        self.gridsearch(SVC(), 
                          {'C':[3.0],
                           'class_weight':['balanced'],
                           'coef0':[0.0],
                           'decision_function_shape':['ovr'],
                           'degree':[3],
                           'gamma':['scale'],
                           'kernel':['rbf'],
                           'probability':[True],
                           'tol':[0.001],
                           'cache_size':[100*1000],
                           'shrinking':[True]})
        
        self.gridsearch(BernoulliNB(), 
                          {'alpha':[1.0e-10],
                           'binarize':[0.0],
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
                          {'n_components':[1,5,10],
                           'max_iter':[300,500,1000],
                           'tol':[.005,.01,.1]})
                           
        self.gridsearch(KNeighborsRegressor(),
                          {'n_neighbors':[2,4,7,10,15,20], 
                           'weights':['uniform','distance'], 
                           'algorithm':['ball_tree','kd_tree','brute'], 
                           'leaf_size':[5,10,25,40,70,100],
                           'p':[1,2]})
        
        self.gridsearch(RandomForestRegressor(), 
                          {'n_estimators':[25,50,100,250], 
                           'max_depth':[None,5,10,50,100,200],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0,.01,.1,.2]})

        self.gridsearch(SVR(), 
                          {'kernel':['linear','poly',
                                     'rbf','sigmoid'],
                           'degree':[2,3,4,5,6,7], 
                           'C':[.5,1,2,5,7,9],
                           'coef0':[0, .01, .1, 1],
                           'tol':[.001,.01,.1],
                           'cache_size':[200000]})

        self.gridsearch(Ridge(), 
                          {'alpha':[1,5,10,15,20],
                           'max_iter':[50,100,500,1000,3000,5000],
                           'tol':[.001,.01,.1,1],
                           'solver':['svd','cholesky','lsqr',
                                     'sparse_cg','sag','saga']})

        self.gridsearch(ElasticNet(), 
                          {'alpha':[1,3,5,10,15],
                           'l1_ratio':[.2,.5,.8,1],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,1],
                           'positive':[False,True]})

        self.gridsearch(AdaBoostRegressor(), 
                          {'n_estimators':[25,50,100,250],
                           'learning_rate':[.01,.1,.5],
                           'loss':['linear','square','exponential']})

        self.gridsearch(TweedieRegressor(), 
                          {'power':[0,1,2,3],
                           'alpha':[.005,.1,.5,1],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,.5,1]})

        self.gridsearch(PassiveAggressiveRegressor(), 
                          {'C':[.1,.5,1,2,5],
                           'fit_intercept':[True,False],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,.5],
                           'early_stopping':[True],
                           'validation_fraction':[.2,.3]})
        self.gridsearch(KernelRidge(), 
                          {'kernel':['additive_chi2','chi2','linear',
                                     'poly','polynomial','rbf',
                                     'laplacian','sigmoid','cosine'],
                           'alpha':[.05,.2,1,2],
                           'degree':[2,3,4,5,6,7],
                           'gamma':[0,.05,.3,1,5],
                           'coef0':[.0,.03,.1,1]})

    @jit
    def Exploratory_regression_hypothetical( self ):
        
        self.gridsearch(LinearRegression(),
                          {'fit_intercept':[True]})

        self.gridsearch(PLSRegression(), 
                          {'n_components':[1,5,10],
                           'max_iter':[300,500,1000],
                           'tol':[.005,.01,.1]})
                           
        self.gridsearch(KNeighborsRegressor(),
                          {'n_neighbors':[2,4,7,10,15,20], 
                           'weights':['uniform','distance'], 
                           'algorithm':['ball_tree','kd_tree','brute'], 
                           'leaf_size':[5,10,25,40,70,100],
                           'p':[1,2]})
        
        self.gridsearch(RandomForestRegressor(), 
                          {'n_estimators':[25,50,100,250], 
                           'max_depth':[None,5,10,50,100,200],
                           'max_features':['auto','sqrt','log2'],
                           'bootstrap':['False','True'],
                           'ccp_alpha':[0,.01,.1,.2]})

        self.gridsearch(SVR(), 
                          {'kernel':['linear','poly',
                                     'rbf','sigmoid'],
                           'degree':[2,3,4,5,6,7], 
                           'C':[.5,1,2,5,7,9],
                           'coef0':[0, .01, .1, 1],
                           'tol':[.001,.01,.1],
                           'cache_size':[200000]})

        self.gridsearch(Ridge(), 
                          {'alpha':[1,5,10,15,20],
                           'max_iter':[50,100,500,1000,3000,5000],
                           'tol':[.001,.01,.1,1],
                           'solver':['svd','cholesky','lsqr',
                                     'sparse_cg','sag','saga']})

        self.gridsearch(ElasticNet(), 
                          {'alpha':[1,3,5,10,15],
                           'l1_ratio':[.2,.5,.8,1],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,1],
                           'positive':[False,True]})

        self.gridsearch(AdaBoostRegressor(), 
                          {'n_estimators':[25,50,100,250],
                           'learning_rate':[.01,.1,.5],
                           'loss':['linear','square','exponential']})

        self.gridsearch(TweedieRegressor(), 
                          {'power':[0,1,2,3],
                           'alpha':[.005,.1,.5,1],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,.5,1]})

        self.gridsearch(PassiveAggressiveRegressor(), 
                          {'C':[.1,.5,1,2,5],
                           'fit_intercept':[True,False],
                           'max_iter':[10,25,50,100,300,1000,3000],
                           'tol':[.01,.1,.5],
                           'early_stopping':[True],
                           'validation_fraction':[.2,.3]})

        self.gridsearch(KernelRidge(), 
                          {'kernel':['cosine'],
                           'alpha':[.2],
                           'degree':[1,2],
                           'gamma':[0],
                           'coef0':[.0]})
        
    def Exploratory_phase(self):
        if self.phenotype in ['pH',
                              'Salt conc.',
                              'Temperature']:
            if self.io.hypothetical == True:
                self.Exploratory_regression_hypothetical()
            else:
                self.Exploratory_regression()
        else:
            if self.io.hypothetical == True:
                self.Exploratory_classification_hypothetical()
            else:
                self.Exploratory_classification()
        
    def SetData(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def __init__(self, phenotype, io, COGs ):
        
        self.phenotype = phenotype
        self.n_jobs = len(sched_getaffinity(0))
        self.io = io
        self.OG = COGs
        
        self.model_name = None
        
        if phenotype in ['range_tmp',
                         'range_salinity',
                         'pH',
                         'Salt conc.',
                         'Temperature']:
            self.multioutput = True
        else:
            self.multioutput = False
