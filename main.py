                            # parameters #

import sys

try:
    n_cpus = int(sys.argv[1])
except IndexError:
    n_cpus = 1
    
phenotype = ['pathways',       #0
             'range_tmp',      #1
             'range_salinity', #2
             'optimum_tmp',    #3
             'optimum_ph',     #4
             'sporulation'][4]

pathway_specification = ['nitrogen_fixation',
                         'nitrate_reduction',
                         'fermentation',
                         'sulfate_reduction'][0]

#if it's a classification/label or regression task
classification_or_regression = ['classification','regression'][1]

#if is multioutput
is_multioutput = True

# 1 if the ortholog groups are curated or 2 if they are hypothetical
curated_or_hypothetical = 2

################################################################################

                     # data collection and organization #

from scripts.data_classes import CollectData, TransformData, MountDataset

CollectData( phenotype, n_cpus, specific_pathway = pathway_specification, number_genomes_per_species = 4 )
TransformData( phenotype, ortholog_groups_DB = curated_or_hypothetical )
MountDataset( phenotype, specimens_per_species = 2 )

################################################################################

             # hyper-parameters search: training & evaluation #

from scripts.training import LGBM, Sklearn
#from scripts.training import ANN
Sklearn( phenotype, classification_or_regression, multioutput = is_multioutput )
LGBM( phenotype, classification_or_regression )
#ANN(phenotype, classification_or_regression, multioutput = is_multioutput)

################################################################################