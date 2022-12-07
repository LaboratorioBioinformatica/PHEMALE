from data_io import DataIO_Classification, DataIO_Regression

import warnings
warnings.filterwarnings('ignore')

from flaml import AutoML #https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML
from lightgbm import LGBMClassifier, LGBMRegressor

class LGBM:

    def __init__(self, phenotype, classification_or_regression):

        self.phenotype = phenotype
        self.classification_or_regression = classification_or_regression

        self.dataIO = DataIO( self.phenotype, self.classification_or_regression )
        x_train, y_train, x_test, y_test, labelize = dataIO.GetTrainingData('data.joblib', splitTrainTest = True, labelize = False)
        
        best_config = self.GridSearch(x_train, y_train, x_test, y_test)
        
        x_train, y_train, labelize = dataIO.GetTrainingData('data.joblib', splitTrainTest = False)
        final_model = self.FinalModel(best_config, x_train, y_train)
        
        self.dataIO.SaveModel(final_model, 'lgbm')

    def GridSearch(self, x_train, y_train, x_test, y_test):
        
        lgbm = AutoML()

        if self.classification_or_regression == 'classification':
            metric = 'macro_F1'
        
        elif self.classification_or_regression == 'regression':
            metric = 'mse'
        
        settings = {'time_budget':4*60*60,
                    'task':self.classification_or_regression,
                    'estimator_list':['lgbm'],
                    'metric':metric,
                    'early_stop':'True'}

        lgbm.fit( X_train = x_train,
                  y_train = y_train,
                  **settings)

        self.dataIO.WriteLog(lgbm.model.estimator)

        y_pred = lgbm.predict(x_test)

        self.dataIO.Metrics(y_test, y_pred)

        return lgbm.best_config

    def FinalModel(self, config, x, y):

        if self.classification_or_regression == 'classification':
            lgbm = LGBMClassifier(**config)

        elif self.classification_or_regression == 'regression':
            lgbm = LGBMRegressor(**config)

        lgbm.fit(x, y)
        return lgbm