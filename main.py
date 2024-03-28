"""
TODO list:
range_salinity 
optimum_ph
nitrogen_fixation

List of possible phenotypes:range_tmp
                            range_salinity
                            optimum_tmp
                            optimum_ph
                            metabolism = oxygen use
                            sporulation
                            nitrogen_fixation
                            nitrate_reduction
                            fermentation
                            sulfate_reduction
"""
import sys
phenotype = str(sys.argv[1])

try:
    if str(sys.argv[2]) == 'hypothetical' or str(sys.argv[2]) == '2':
        hypothetical_OG = True
except IndexError:
    hypothetical_OG = False

######################## Data Preparation ########################
from scripts.collect_data import CollectData
#CollectData( phenotype, genomes_per_species = 1 )

from scripts.eggnog_mapping import EggnogMapping
#EggnogMapping( phenotype )

from scripts.standardization import DataStandardization
#DataStandardization( phenotype, hypothetical_OG)

from scripts.create_dataset import CreateDataset
#CreateDataset( phenotype )

from scripts.data_IO import IO
io = IO( phenotype )

io.WriteLog( 'Phenotype: ' + phenotype )
io.WriteLog( 'Hypothetical COGs: ' + str(hypothetical_OG) )

x = io.GetX()
y = io.GetY()

x_train, y_train, x_test, y_test = io.SplitData(x, y, 0.2)

############################ Training ############################
from scripts.sklearn_train import Sklearn
Sklearn( phenotype, x_train, y_train, x_test, y_test, io )

#https://jovian.com/poduguvenu/xgboost-lightgbm-catboost-sklearn-gradientboosting-comparision
from scripts.treeboost_train import TreeBoost
tree = TreeBoost( phenotype, io )
for model in ['lgbm','xgboost']:
    best_config = tree.GridSearch(x_train, y_train, model)
    predictor = tree.FinalModel(model, best_config, x_test, y_test)
    y_pred = predictor.predict(x_test)
    if phenotype in ['optimum_tmp','optimum_ph']:
        io.Metrics_Regression(y_test, y_pred, model)
    else:
        y_pred_prob = predictor.predict_proba(x_test)
        io.Metrics_Classification(y_test, y_pred, y_pred_prob, model)

########################## Final models ###########################

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
#https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting