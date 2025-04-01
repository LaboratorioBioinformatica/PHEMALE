from flaml import AutoML
from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor, plot_importance
from xgboost import XGBModel, XGBRegressor, XGBClassifier, plot_importance
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from os import sched_getaffinity
from copy import deepcopy
import numpy
from joblib import load
from pandas import DataFrame, concat, to_numeric
import matplotlib.pyplot as plt
import shap

from warnings import filterwarnings
filterwarnings('ignore')

class TreeBoost:

    def __init__(self, phenotype, io, COGs):

        self.phenotype = phenotype
        self.n_jobs = len(sched_getaffinity(0))
        self.io = io
        self.OGs = COGs

        if self.phenotype in ['pH', 'Salt conc.', 'Temperature']:
            self.classification_or_regression = 'regression'
            metrics = ['mse','r2','max_error','neg_median_absolute_error','neg_mean_absolute_error','neg_mean_squared_error']
            self.metric = metrics[3]
        else:
            self.classification_or_regression = 'classification'
            metrics = ['roc_auc_ovo_weighted','f1_macro']
            self.metric = metrics[1]
            
        self.io.WriteLog('Using as score: ' + self.metric)
            
    def GridSearch(self, x, y, x_test, y_test):
        
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        
        models = []
        
        # METRICS - https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters

        if self.classification_or_regression == 'classification':
            #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
            models.append( LGBMClassifier() )
            #https://xgboost.readthedocs.io/en/latest/python/python_api.html
            models.append( XGBClassifier() )
        elif self.classification_or_regression == 'regression':
            #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
            models.append( LGBMRegressor() )
            models.append( XGBRegressor() )

        for model in models:
            file_name = str(model).split('(')[0]
            self.io.WriteLog( file_name )
            model = self._GridSearch_(model, x, y)
            
            y_pred = model.predict(x_test)
            if self.phenotype in ['pH','Temperature','Salt conc.']:
                self.io.Metrics_Regression(y_test, y_pred, file_name)
            else:
                #for predictor in model:
                for idx in range(len(model.estimators_)):
                    y_pred_prob = model.estimators_[idx].predict_proba(x_test)
                    self.io.Metrics_Classification(y_test, y_pred, y_pred_prob, file_name)
    
    def _GridSearch_(self, model, x, y):
        if isinstance(model,LGBMModel):
            param_grid = {"estimator__num_leaves": [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30],
                          "estimator__max_depth": [-1,2,3,4,5,6,7,8,9,10,15,20],
                          "estimator__subsample": [0.2, 0.5, 0.8, 1.0],
                          "estimator__subsample_freq":[0,1],
                          "estimator__colsample_bytree": [0.1,0.2,0.4,0.6,0.8,1.0],
                          "estimator__n_estimators": [1,2,5,7,10,20,30,40,50,60,70,100,300],
                          'estimator__boosting_type': ['gbdt', 'dart', 'rf'], # for better accuracy -> try dart
                          'estimator__reg_alpha' : [0,0.5,1,1.5,2],
                          'estimator__reg_lambda' : [0,0.5,1,1.5,2],
                          'estimator__min_split_gain':[0.0,0.2,0.5], 
                          'estimator__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5],
                          'estimator__min_child_samples':[2,3,4,5,6,10],
                          'estimator__min_child_weight':[0.0001, 0.001, 0.01, 0.5, 1, 3, 5], 
                          'estimator__importance_type':['split','gain'],
                          'estimator__class_weight':['balanced', None],
                          'estimator__max_bin': [1,2,5,10,50,100,200,300,500,1000], # large max_bin helps improve accuracy but slow training
                          'estimator__min_data_in_leaf': [1,2,3,8,10,15,50,100,300]
                         }
        elif isinstance(model,XGBModel):
            param_grid = {'estimator__grow_policy':['depthwise','lossguide'],
                          'estimator__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5,1.0],
                          'estimator__verbosity':[0],
                          'estimator__booster':['gbtree', 'gblinear','dart'],
                          'estimator__sampling_method':['gradient_based','uniform'],
                          "estimator__subsample": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          'estimator__colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          'estimator__colsample_bylevel': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          'estimator__colsample_bynode': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          'estimator__gamma': [0.1,0.2,0.5, 1, 1.5, 2, 5],
                          'estimator__reg_alpha' : [0,0.5,1,1.5,2,3,5],
                          'estimator__reg_lambda' : [0,0.5,1,1.5,2,3,5],
                          'estimator__min_child_weight':[0.0001, 0.001, 0.01, 1, 3, 5], 
                          'estimator__max_leaves': [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,50,70],
                          'estimator__max_delta_step':[0.1,0.5,1,1.5,2],
                          'estimator__scale_pos_weight':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          'estimator__max_bin': [1,2,5,10,50,100,200,300,500,1000],
                          "estimator__max_depth": [0,1,2,3,4,5,6,7,8,9,10,15],
                          'estimator__learning_rate': [0.005,0.01,0.05,0.1,0.5,1.0,2.0],
                          'estimator__n_estimators': [1,2,5,7,10,20,30,40,50,60,70,100,300],
                          'estimator__tree_method':['exact', 'approx', 'hist']}

        grid = RandomizedSearchCV(estimator=MultiOutputRegressor(model), param_distributions=param_grid, 
                                  n_iter=50, cv=4, scoring=self.metric, n_jobs=-1, verbose=0)
        grid.fit(X=x, y=y)
        
        for estimator in grid.best_estimator_.estimators_:
            self.io.WriteLog( str( estimator.get_params() ) )

        self.Feature_importance_intrinsic(model, grid.best_estimator_)
        self.io.Save(grid.best_estimator_, str(model).split('(')[0])
        return grid.best_estimator_

    def Feature_importance_intrinsic(self, model, predictor, n_features = 20):

        for idx in range(len(predictor.estimators_)):
            if isinstance(model,LGBMModel):
                feat_importance = DataFrame(predictor.estimators_[idx].booster_.feature_importance(importance_type='gain'),
                                            index=self.OGs)
            elif isinstance(model,XGBModel):
                feat_importance = DataFrame(predictor.estimators_[idx].feature_importances_, index=self.OGs)
                model = str(model).split('(')[0]
                
            feat_importance = feat_importance.loc[~(feat_importance == 0).all(axis=1)]
            feat_importance = feat_importance.nlargest(n_features, feat_importance.columns.values.tolist(),keep='first')

            if feat_importance.empty:
                self.io.WriteLog( 'No relevant features detected in the model.' )
            else:
                fig_name = self.io.save_folder+str(model)+'_'+str(idx)+'_intrinsic.png'
                plt.figure( fig_name )
                feat_importance.plot(kind='bar',legend=False)
                plt.tight_layout()
                plt.savefig(fig_name, dpi=300)

    def Feature_importance_SHAP(self, model, x, name ):
        for i in range(len(model.estimators_)):
            explainer = shap.TreeExplainer(model.estimators_[i])
            shap_values = explainer.shap_values(x)
            plt.figure( str(i) )
            shap.summary_plot(shap_values, feature_names=self.OGs, plot_type='bar', max_display = 10, show=False)
            plt.savefig(self.io.save_folder+str(i)+'_'+name+'_SHAP.png',dpi=300)
