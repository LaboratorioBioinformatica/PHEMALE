import os
from glob import glob
from numba import jit
from joblib import dump, load
from random import shuffle, choices
import matplotlib.colors as mcolors

from pandas import DataFrame, concat, read_csv, options
options.display.max_colwidth = 2000  #higher data threshold limit for pandas

from warnings import filterwarnings
filterwarnings('ignore')

def Taxonomy(metadata, saveFolder):
    
    order = metadata.order.value_counts().rename_axis('data').reset_index(name='counts')
    families = len( metadata.family.value_counts().rename_axis('data').reset_index(name='counts') )
    genus = len( metadata.genus.value_counts().rename_axis('data').reset_index(name='counts') )
    species = len( metadata.species.value_counts().rename_axis('data').reset_index(name='counts') )

    with open(saveFolder+'distribution.log', 'w') as file:
        file.write('Number of orders: ' + str(len(order)) + '\n')
        file.write('Number of families: ' + str(families) + '\n')
        file.write('Number of genus: ' + str(genus) + '\n')
        file.write('Number of species: ' + str(species) + '\n')

    colors = choices( list(mcolors.CSS4_COLORS.values()) , k = len(order) )
    orderGraph = order.plot(y='counts', kind='pie', figsize=(5, 5), labels = order.data, labeldistance=None, colors=colors)
    orderGraph.legend(loc="right", bbox_to_anchor=(3.0,0.5), fontsize=8, ncol=4)
    orderGraph.figure.savefig(saveFolder + 'order_distribution.png', bbox_inches="tight", dpi=800)

class CreateDataset:
    
    """
    Function: generate metadata header
    """
    def MetadataHeader( self ):
        
        if self.phenotype in ['range_salinity','optimum_tmp','optimum_ph']:
            return ['file', 
                    'class', 
                    'order', 
                    'family', 
                    'genus', 
                    'species', 
                    'phenotype1',
                    'phenotype2']
        
        elif self.phenotype == 'range_tmp':
            return ['file', 
                    'class', 
                    'order', 
                    'family', 
                    'genus',
                    'species', 
                    'phenotype1',
                    'phenotype2',
                    'phenotype3']
        else:
            return ['file', 
                    'class', 
                    'order', 
                    'family', 
                    'genus', 
                    'species', 
                    'phenotype1']
    
    """
    Function: generate metadata of a specific genome file
    metadatum: row from a pandas dataframe with informations on class, order, family, genus and species
    """
    def ProccessMetadatum( self, datum_path, metadatum):
        
        if self.phenotype in ['range_salinity','optimum_tmp']:
            return [datum_path, 
                    metadatum.classes, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1, 
                    metadatum.phenotype2]

        elif self.phenotype == 'optimum_ph':
            return [datum_path, 
                    metadatum.classes, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    (10**3)*(2**(-metadatum.phenotype1)),
                    (10**3)*(2**(-metadatum.phenotype2)) ]

        elif self.phenotype == 'range_tmp':
            return [datum_path, 
                    metadatum.classes, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1, 
                    metadatum.phenotype2, 
                    metadatum.phenotype3]
        else:
            return [datum_path, 
                    metadatum.classes, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1]

    @jit
    def GenerateDatum( self, file ):
        
        genome = load( file )
        genome = genome + [0]*( self.OG_columns - len( genome ))
        return genome
        
    """
    Function: calculates ANI between file1 and file2 with FastANI
    file1: genome file (.fna)
    file2: genome file (.fna)
    """
    def ANI(self, file1, file2):
        n_cpus = len(os.sched_getaffinity(0))
        os.system('./FastANI/fastANI -q ' + file1 + ' -r '+ file2 + ' -t ' + str(n_cpus) + ' -o ./FastANI/ani.out')
        if os.stat('./FastANI/ani_'+self.phenotype+'.out').st_size != 0:
            ani = open( './FastANI/ani_'+self.phenotype+'.out','r').read()
            os.system('rm ./FastANI/ani.out')
            return float(ani.split()[2])
        else:
            return float(0.5)
        
    @jit
    def CheckRedundancy(self, file, phenotypes, ANI = 0.0, Pearson = 0.0):
        
        for idx in range(0,len(self.metadata)):
            
            if self.metadata[idx][len(phenotypes):] == phenotypes: # check if phenotype is the same
                
                if Pearson > 0.0:
                    datum = self.GenerateDatum( file )
                    identity = pearsonr( self.data[idx] , datum )[0]
                    threshold = Pearson
                
                if ANI > 0.0:
                    identity = self.ANI( self.metadata[idx][0] , file )[0]
                    threshold = ANI

                if abs(identity) > threshold:
                    return False

        return True
        
    """
    Function: generates metadata for training containing
    max_files_per_species: max number of files requested per folder
    """
    def MountDataset(self, ANI_threshold, Pearson_threshold):

        for idx, row in self.madin.iterrows():
            
            folder = '../data/genomes/'+str(row['taxid'])+'/'
            files = glob(folder + '*' + self.phenotype + '.joblib')
            shuffle( files ) #shuffles the list in place, in other terms doesnt returns anything
            
            for file in files:

                metadatum = self.ProccessMetadatum( file, row )
                datum = self.GenerateDatum( file )
                
                if len(datum) > self.OG_columns:
                    continue
                
                if ANI_threshold > 0.0:
                    if self.CheckRedundancy(file, metadatum[6:], ANI = ANI_threshold) == False:
                        continue
                
                if Pearson_threshold > 0.0:
                    if self.CheckRedundancy(file, metadatum[6:], Pearson = Pearson_threshold) == False:
                        continue
                    
                self.metadata.append( metadatum )
                self.data.append( datum )
                
                if ANI_threshold == 0.0 and Pearson_threshold == 0.0:
                    break

    def __init__(self, phenotype, threshold_ANI = 0, threshold_Pearson = 0):
        
        self.phenotype = phenotype
        self.metadata = []
        self.data = []
        self.OG_columns = len( load('./results/'+self.phenotype+'/data/'+'OGColumns.joblib') )
        
        phenotype_folder = './results/'+phenotype+'/data/'
        self.madin = read_csv( phenotype_folder+'madin.csv')
        
        ### In case of additional file with phenotype information about species that Madin didn't have
        if os.path.isfile( 'madin_'+self.phenotype+'_additional.csv' ) == True:
            self.madin = concat([self.madin, read_csv( 'madin_'+self.phenotype+'_additional.csv')])
        
        self.MountDataset( threshold_ANI, threshold_Pearson )
        
        dump( self.data, phenotype_folder + 'data.joblib' )
        
        self.metadata = DataFrame(self.metadata, columns=self.MetadataHeader())
        self.metadata.to_csv(phenotype_folder + 'metadata.csv', index=None)
        
        Taxonomy(self.metadata, phenotype_folder )
        
        print('Selected data and metadata saved in directory: ' + phenotype_folder)