from flaml import AutoML
from lightgbm import LGBMClassifier, LGBMRegressor, plot_importance
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
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

    def __init__(self, phenotype, io):

        self.phenotype = phenotype
        self.n_jobs = len(sched_getaffinity(0))
        self.io = io
        self.OGs = load('./results/'+self.phenotype+'/data/OGColumns.joblib')
        if self.phenotype == 'optimum_ph' or self.phenotype == 'optimum_tmp':
            self.classification_or_regression = 'regression'
        else:
            self.classification_or_regression = 'classification'
        
    def GridSearch(self, x, y, model = 'lgbm'):

        if self.classification_or_regression == 'classification':
            metric = 'log_loss'
            #metric = 'macro_f1'
        elif self.classification_or_regression == 'regression':
            metric = 'r2'
        
        settings = {'time_budget':1.0*24*60*60,
                    'task':self.classification_or_regression,
                    'estimator_list':[model],
                    'metric':metric,
                    'early_stop':'True',
                    'eval_method':'cv'}

        y = numpy.hsplit(y, len(y[0]))[0]
        y = numpy.ravel(y)
        
        automl = AutoML()
        automl.fit(X_train=x, y_train=y, **settings)
        
        self.io.WriteLog(automl.model.estimator)

        return automl.best_config

    def FinalModel(self, model, config, x, y):

        if self.classification_or_regression == 'classification':
            if model is 'lgbm':
                predictor = MultiOutputClassifier( LGBMClassifier(**config, class_weight ='balanced'), n_jobs = self.n_jobs )
            elif model is 'xgboost':
                predictor = MultiOutputClassifier( XGBClassifier(**config), n_jobs = self.n_jobs )

        elif self.classification_or_regression == 'regression':
            if model is 'lgbm':
                predictor = MultiOutputRegressor( LGBMRegressor(**config), n_jobs = self.n_jobs )
            elif model is 'xgboost':
                predictor = MultiOutputRegressor( XGBRegressor(**config), n_jobs = self.n_jobs )

        predictor.fit(x, y)
        
        self.Feature_importance_intrinsic(model, predictor)
        self.io.Save(predictor, model)
        return predictor

    def Feature_importance_intrinsic(self, model, predictor, n_features = 20):
        for idx in range(len(predictor.estimators_)):
            if model is 'lgbm':
                feat_importance = DataFrame(predictor.estimators_[idx].booster_.feature_importance(importance_type='gain'),
                                            index=self.OGs)
            elif model is 'xgboost':
                feat_importance = DataFrame(predictor.estimators_[idx].feature_importances_, index=self.OGs)

            feat_importance = feat_importance.loc[~(feat_importance == 0).all(axis=1)]
            feat_importance = feat_importance.nlargest(n_features, feat_importance.columns.values.tolist(),keep='first')

            if feat_importance.empty:
                self.io.WriteLog( 'No relevant features detected in the model.' )
            else:
                fig_name = self.io.save_folder+model+'_'+str(idx)+'_intrinsic.png'
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