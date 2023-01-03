################################################################################

                            # parameters #

import sys
try:
    #cpus used with eggnog-mapper
    n_cpus = int(sys.argv[1])
except IndexError:
    n_cpus = 1
    
phenotype = ['pathways',
             'range_tmp',
             'range_salinity',
             'optimum_tmp',
             'optimum_ph'][4]

pathway_specification = ['nitrogen_fixation',
                         'nitrate_reduction',
                         'fermentation',
                         'sulfate_reduction'][0]

#if it's a classification/label or regression task
classification_or_regression = ['classification','regression'][1]

#if is multioutput
is_multioutput = True

#the lower the threshold the more stern the filter
redundancy_threshold = 0.9

# 1 if the ortholog groups are curated or 2 if they are hypothetical
curated_or_hypothetical = 1

################################################################################

                     # data collection and organization #

from scripts.data_classes import CollectData, TransformData, MountDataset

CollectData( phenotype, n_cpus, specific_pathway = pathway_specification )
TransformData( phenotype, ortholog_groups_DB = curated_or_hypothetical )
MountDataset( phenotype, redundancy_threshold )

################################################################################

             # hyper-parameters search: training & evaluation #

from scripts.training import LGBM, Sklearn
LGBM( phenotype, classification_or_regression )
Sklearn( phenotype, classification_or_regression, multioutput = is_multioutput )

#from scripts.training import ANN
#ANN(phenotype, classification_or_regression, multioutput = is_multioutput)

################################################################################