"""
Madin { range_tmp, range_salinity,
        metabolism = oxygen use, nitrogen_fixation, nitrate_reduction, 
        fermentation, sulfate_reduction }
BacDive { pH, Temperature, Salt conc., Oxygen tolerance }

"""
import sys
phenotype = str(sys.argv[1])

try:
    if str(sys.argv[2]) in ['hypothetical','hypothetic']:
        hyp = True
    elif str(sys.argv[2]) == 'curated':
        hyp = False
    elif str(sys.argv[2]) == 'binary':
        binary = True
        hyp = False
    elif str(sys.argv[2]) in ['numeric','numerical']:
        binary = False
        hyp = False
except IndexError:
    hyp = False
    
try:
    if str(sys.argv[3]) == 'binary':
        binary = True
    elif str(sys.argv[3]) == 'numeric':
        binary = False
except IndexError:
    binary = False
    
######################### Data Preparation ########################
"""
from scripts.collect_data import CollectData
cd = CollectData( phenotype, genomes_per_species = 1 )
del(cd)

from scripts.eggnog_mapping import EggnogMapping
em = EggnogMapping( phenotype )
del(em)
"""

from scripts.data_IO import IO
io = IO( phenotype, hypothetical = hyp)

from scripts.standardization import DataStandardization
ds = DataStandardization( phenotype, hyp, io)
del(ds)

from scripts.create_dataset import CreateDataset
filtercog = 0
cd = CreateDataset(phenotype, io, hypothetical = hyp, filter_cogs = filtercog)
del(cd)

io.WriteLog( 'Phenotype: ' + phenotype )
io.WriteLog( 'Using hypothetical cogs: '+str(hyp))

io.LoadOGcolumns()

x = io.GetX()
y = io.GetY()

stress = io.MDS_plot(x,y)
io.WriteLog('Stress:'+ str(stress))
    
dataset = None
#dataset = 'enum'
if dataset == None:
    x_train, y_train, x_test, y_test = io.SplitData(x, y, 0.2)
    io.WriteLog( 'Training samples: ' + str(len(y_train)) )
    io.WriteLog( 'Testing samples: ' + str(len(y_test)) )
else:
    x_train, y_train, x_test, y_test = io.GetAlreadySplitData(dataset)
    io.WriteLog( 'Using dataset: ' + dataset )
    io.WriteLog( 'Training samples: ' + str(len(x_train)) )
    io.WriteLog( 'Testing samples: ' + str(len(x_test)) )

if binary == True:
    x_train = io.ContinuousToBinary(x_train)
    x_test = io.ContinuousToBinary(x_test)
    io.WriteLog( 'Using presence/absence of Ortholog Groups as parameters' )
io.WriteLog('')

from scripts.sklearn_train import Sklearn
skl = Sklearn( phenotype, io, io.OG )

from scripts.treeboost_train import TreeBoost
tree = TreeBoost( phenotype, io, io.OG )

phase=1

############################ 1st Exploratory phase ############################    
if phase == 1:
    skl.SetData(x_train, y_train, x_test, y_test)
    skl.Exploratory_phase()

    tree.GridSearch(x_train, y_train, x_test, y_test)            

############################ 2nd Exploratory phase ############################
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier

if phase == 2:
    io.WriteLog( 'Committee ensemble models' )

    if phenotype == 'nitrogen_fixation':
        models = {'LR1':LogisticRegression(C=0.1, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, 
                                           l1_ratio=None, max_iter=100, multi_class='multinomial', n_jobs=None, tol=0.01,
                                           penalty='none', random_state=None, solver='sag', warm_start=False),
                  'LR2':LogisticRegression(C=5.0, class_weight='balanced',dual=False,fit_intercept=True,intercept_scaling=1,
                                           l1_ratio=0.7, max_iter=50, multi_class='multinomial', n_jobs=None, tol=0.001,
                                           penalty='none', random_state=None, solver='saga', warm_start=False),
                  'SVC':SVC(C=0.1, break_ties=False, cache_size=100000, class_weight='balanced', coef0=3.0, 
                            decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear', max_iter=-1, 
                            probability=True, random_state=None, shrinking=True, tol=0.001)
                 }

    if phenotype in ['optimum_tmp','optimum_ph']:
        council = VotingRegressor(estimators=[(key, models[key]) for key in models.keys()], weights=None)
        council = council.fit(x_train, y_train)
        y_pred = council.predict(x_test)
        io.Metrics_Regression(y_test, y_pred, 'council')
    else:
        council = VotingClassifier(estimators=[(key, models[key]) for key in models.keys()], voting='soft')
        council = council.fit(x_train, y_train)
        y_pred = council.predict(x_test)
        y_pred_prob = council.predict_proba(x_test)
        io.Metrics_Classification(y_test, y_pred, y_pred_prob, 'council')
    io.Save(council, 'council')
    
#########################################################################################################################################

if phase == 3:
    model = load('results/'+phenotype+'/cog/LogisticRegression.joblib')
    genomes = load('results/'+phenotype+'/cog/test_genomes.joblib')
    for idx in range(len(x_test)):
        while len(x_test[idx]) > 4638:
            x_test[idx] = x_test[idx][:-1]
        x = [x_test[idx]]
        y_pred = model.predict(x)
        io.WriteLog(str(y_test[idx]) +' '+ str(y_pred) + ' '+genomes[idx])
    quit()
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    io.Metrics_Classification(y_test, y_pred, y_pred_prob, 'LogisticRegression_cog')
    
#########################################################################################################################################

from joblib import load
from numpy import array
import random
if phase == 4:
    model = load('results/'+phenotype+'/cog/LogisticRegression.joblib')
    genomes = load('results/'+phenotype+'/cog/test_genomes.joblib')
    for idx in range(len(x_test)):
        while len(x_test[idx]) > 4638:
            x_test[idx] = x_test[idx][:-1]
        x = [x_test[idx]]
        y_pred = model.predict(x)
        io.WriteLog(str(y_test[idx]) +' '+ str(y_pred) + ' '+genomes[idx])
    quit()
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    io.Metrics_Classification(y_test, y_pred, y_pred_prob, 'LogisticRegression_cog')
