phenotype = ['range_tmp',         #0
             'range_salinity',    #1
             'optimum_tmp',       #2
             'optimum_ph',        #3
             'sporulation',       #4
             'nitrogen_fixation', #5
             'nitrate_reduction', #6
             'fermentation',      #7
             'sulfate_reduction' #8
            ][5]

################################################################################

                     # data collection and organization #

from scripts.data_classes import CollectData, SelectData, MountData

CollectData( phenotype, number_genomes_per_species = 10 )
SelectData( phenotype, specimens_per_species = 1 )
MountData( phenotype, only_COGs = False, sampling = 1.0 )

################################################################################

             # hyper-parameters search: training & evaluation #

from scripts.training import LGBM, Sklearn
Sklearn(phenotype)
LGBM(phenotype)

################################################################################