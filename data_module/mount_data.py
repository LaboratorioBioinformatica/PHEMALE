from numba import jit
from joblib import load, dump
from scipy.stats.stats import pearsonr

import pandas
pandas.options.display.max_colwidth = 2000  #aumenta limite de dados do pandas

import warnings
warnings.filterwarnings('ignore')

class MountDataset:
    
    def GenerateMetadata( self ):
        
        metadata = [['genome', 'class', 'order', 'family', 'genus', 'species']]
        
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_ph':
            metadata[0] = metadata[0] + ['phenotype1','phenotype2']
        elif self.phenotype == 'range_tmp':
            metadata[0] = metadata[0] + ['phenotype1','phenotype2','phenotype3']
        elif self.phenotype == 'pathways' or self.phenotype == 'motility':
            metadata[0] = metadata[0] + ['phenotype1']
            
        return metadata
    
    def GenerateDatum( self, file, row, number_of_OG_columns ):
        
        genome = load( file )
        genome = genome + [0]*( number_of_OG_columns - len( genome ))
        
        metadatum = [file, row.classes, row.order, row.family, row.genus, row.species]
        
        append = None
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_ph':
            append = [ row.phenotype1 , row.phenotype2 ]
        
        elif self.phenotype == 'range_tmp':
            append = [ row.phenotype1 , row.phenotype2 , row.phenotype3 ]
        
        elif self.phenotype == 'pathways' or self.phenotype == 'motility':
            append = [ row.phenotype1 ]
        
        genome = genome + append
        metadatum = metadatum + append
        
        return genome, metadatum
    
    @jit
    def CheckRedundancy(self, datum, data, threshold, number_of_OG_columns):
        for i in data:
            if i[number_of_OG_columns:] == datum[number_of_OG_columns:]:
                if pearsonr( i[:number_of_OG_columns] , datum[:number_of_OG_columns] )[0] > threshold:
                    return True
        return False

    @jit
    def MountDataset(self, madin, number_of_OG_columns, redundancy_threshold ):
        
        data = []
        metadata = self.GenerateMetadata()
        
        for idx, row in madin.iterrows():
            
            folder = '../data/genomes/'+str(row['taxid'])+'/'
            
            if os.path.isdir(folder):
                
                for file in os.listdir(folder):
                    
                    if file.endswith( '.annotations' + self.phenotype + '.joblib' ):
                        
                        genome, metadatum = self.GenerateDatum( folder + file, row, number_of_OG_columns )
                        
                        if self.CheckRedundancy( genome, data, redundancy_threshold, number_of_OG_columns ) == False:
                            
                            data.append( genome )
                            metadata.append( metadatum )

        return data, metadata
    
    def __init__(self, phenotype, redundancy_threshold):
        
        self.phenotype = phenotype
        phenotype_folder = './metadata/'+self.phenotype+'/'
        
        madin = pandas.read_csv( phenotype_folder+'madin_'+self.phenotype+'.csv')
        number_of_OG_columns = len( load(phenotype_folder+'OGColumns.joblib') )
        redundancy_threshold = redundancy_threshold

        data, metadata = self.MountDataset( madin, number_of_OG_columns, redundancy_threshold )

        dump( metadata, phenotype_folder + 'metadata.joblib' )
        dump( data, phenotype_folder + 'data.joblib' )

        print('Dataset & metadata constructed in directory: '+phenotype_folder)