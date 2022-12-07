import os
os.system('rm phemale-cpu.o*')

################################################################################

                            # parameters #
    
#phenotype of interest
phenotype = 'pathways'
pathway_specification = 'nitrogen_fixation'

#if it's a classification/label or regression task
classification_or_regression = 'classification'
#classification_or_regression = 'regression'

is_multioutput = False

#cpus used with eggnog-mapper
n_cpus = 50

#the lower the threshold the more stern the filter
redundancy_threshold = 0.9

# 1 if the ortholog groups are curated or 2 if they are hypothetical
curated_or_hypothetical = 1

################################################################################

                     # data collection and organization #

from data_module.collect_data import CollectData
from data_module.mount_data import MountDataset
from data_module.transform_data import TransformData

CollectData( phenotype, n_cpus, specific_pathway = pathway_specification )

TransformData( phenotype, ortholog_groups_DB = curated_or_hypothetical )

MountDataset( phenotype, redundancy_threshold )

################################################################################

             # hyper-parameters search: training & evaluation #

from training_module.sklearn_training import Sklearn_Training
from training_module.lgbm_training import LGBM
from training_module.ann_training import ANN

Sklearn_Training( phenotype, classification_or_regression, multioutput = is_multioutput )
LGBM( phenotype, classification_or_regression, multioutput = is_multioutput )

################################################################################