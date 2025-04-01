import os
from glob import glob
from numba import jit
from joblib import dump, load
from random import shuffle, choices
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import itertools

from pandas import DataFrame, concat, read_csv, options
options.display.max_colwidth = 2000  #higher data threshold limit for pandas

from warnings import filterwarnings
filterwarnings('ignore')

def Taxonomy(metadata, saveFolder):
    
    order = metadata.order.value_counts().to_dict()
    families = len( metadata.family.value_counts().rename_axis('data').reset_index(name='counts') )
    genus = len( metadata.genus.value_counts().rename_axis('data').reset_index(name='counts') )
    species = len( metadata.species.value_counts().rename_axis('data').reset_index(name='counts') )

    with open(saveFolder+'distribution.log', 'w') as file:
        file.write('Number of orders: ' + str(len(order.keys())) + '\n')
        file.write('Number of families: ' + str(families) + '\n')
        file.write('Number of genus: ' + str(genus) + '\n')
        file.write('Number of species: ' + str(species) + '\n\n')
        file.write('Orders with at most 10 representatives:\n')
        for key, value in order.items():
            if value < 11:
                file.write(key+','+str(value)+'\n')

    colors = choices( list(mcolors.CSS4_COLORS.values()) , k = len(order) )
    
    # group together all elements in the dictionary whose value is less than 2
    newdic={}
    for key, group in itertools.groupby(order, lambda k: 'Ordens com até\n10 representantes' if (order[k]<16) else k):
         newdic[key] = sum([order[k] for k in list(group)])

    orderGraph, ax = plt.subplots()
    labels = [x+' ('+str(y)+')' for x, y in zip(list(newdic.keys()), newdic.values())]

    ax.pie(newdic.values(), 
           autopct='%1d%%', 
           startangle=0, 
           labeldistance=0.5, 
           textprops={'fontsize': 7}, 
           pctdistance=1.15,
           colors=colors)
    orderGraph.legend(loc="right", bbox_to_anchor=(1.2,0.5), fontsize=8, ncol=1, labels=labels)
    ax.axis('equal')
    orderGraph.tight_layout()
    orderGraph.savefig(saveFolder + 'order_distribution.png', bbox_inches="tight", dpi=800)
    #orderGraph = order.plot(y='counts', kind='pie', figsize=(5, 5), labels = order.data, labeldistance=None, colors=colors)

class CreateDataset:
    
    """
    Function: generate metadata header
    """
    def MetadataHeader( self ):
        
        if self.phenotype in ['range_salinity','pH','Temperature','Salt conc.']:
            return ['file', 
                    'order', 
                    'family', 
                    'genus', 
                    'species', 
                    'phenotype1',
                    'phenotype2']
        
        elif self.phenotype == 'range_tmp':
            return ['file', 
                    'order', 
                    'family', 
                    'genus',
                    'species', 
                    'phenotype1',
                    'phenotype2',
                    'phenotype3']
        else:
            return ['file', 
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
        
        if self.phenotype in ['range_salinity','optimum_tmp','optimum_ph']:
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1, 
                    metadatum.phenotype2]
        
        elif self.phenotype in ['pH']:
            
            ## teste
            phs = [float(metadatum.phenotype1-7),float(metadatum.phenotype2-7)]
            phs_post_proccess = []
            for ph in ph12:
                if ph >= 0:
                    phs_post_proccess.append(-10**ph)
                else:
                    phs_post_proccess.append(10**ph)
            ## fim do teste
            
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    #float(metadatum.phenotype1-7)*1000, 
                    #float(metadatum.phenotype2-7)*1000]
                    ## teste
                    phs_post_proccess[0], 
                    phs_post_proccess[1]]
                    ## fim do teste

        elif self.phenotype in ['Salt conc.']:
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    float(metadatum.phenotype1)*1000, 
                    float(metadatum.phenotype2)*1000]
        
        if self.phenotype in ['Temperature']:
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    (metadatum.phenotype1-25.0)*100, 
                    (metadatum.phenotype2-35.0)*100]
        
        elif self.phenotype == 'range_tmp':
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1, 
                    metadatum.phenotype2, 
                    metadatum.phenotype3]
        else:
            return [datum_path, 
                    metadatum.order, 
                    metadatum.family, 
                    metadatum.genus, 
                    metadatum.species, 
                    metadatum.phenotype1]

    @jit
    def GenerateDatum( self, file ):
        
        genome = load( file )
        genome = genome + [0]*( len(self.OG_columns) - len( genome ))
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

        for idx, row in self.phenDB.iterrows():
            
            folder = '../data/genomes/'+str(row['taxid'])+'/'
            if self.hypothetical == True:
                files = glob(folder + '*' + self.phenotype + '_hypothetical.joblib')
            else:
                files = glob(folder + '*' + self.phenotype + '.joblib')
            shuffle( files ) #shuffles the list in place, in other terms doesnt returns anything
            
            for file in files:

                metadatum = self.ProccessMetadatum( file, row )
                
                # filter for temperature in range
                #if metadatum[5] > 11*100 or metadatum[6] > 11*100:
                    #continue
                
                datum = self.GenerateDatum( file )
                
                # n sei pq
                #if len(datum) > len(self.OG_columns):
                    #continue
                
                if ANI_threshold > 0.0:
                    if self.CheckRedundancy(file, metadatum[5:], ANI = ANI_threshold) == False:
                        continue
                
                if Pearson_threshold > 0.0:
                    if self.CheckRedundancy(file, metadatum[5:], Pearson = Pearson_threshold) == False:
                        continue
                    
                self.metadata.append( metadatum )
                self.data.append( datum )
                
                if ANI_threshold == 0.0 and Pearson_threshold == 0.0:
                    break
    
    @jit              
    def Filter_cogs(self, threshold = 10.00):
        #1st filtration
        data = DataFrame(self.data,columns=self.OG_columns)
        #data = data[data.columns[data.sum() >= threshold]]

        #2nd filtration
        for column in data.columns:
            count = data[column].gt(0).sum()
            if count < threshold:
                data.drop(column, axis=1, inplace=True)
        
        self.data = data.to_numpy()
        self.OG_columns = list(data)

    def __init__(self, phenotype, io, hypothetical = False, filter_cogs = False, threshold_ANI = 0, threshold_Pearson = 0):
        
        self.phenotype = phenotype
        self.io = io
        self.metadata = []
        self.data = []
        self.hypothetical = hypothetical
        if self.hypothetical == True:
            self.OG_columns = load(self.io.save_folder+'/OGColumns_hypothetical.joblib')
        else:
            self.OG_columns = load(self.io.save_folder+'/OGColumns.joblib')
        
        phenotype_folder = './results/'+phenotype+'/data/'
        self.phenDB = read_csv( phenotype_folder+'phenDB.csv')
        
        self.MountDataset( threshold_ANI, threshold_Pearson )
        if filter_cogs != False:
            self.Filter_cogs(threshold = filter_cogs)
        
        """
        if self.hypothetical == True:
            dump( self.data, phenotype_folder + 'data_hypothetical.joblib' )
            dump( self.OG_columns, self.io.save_folder+'OGColumns_hypothetical.joblib')
        else:
            dump( self.data, phenotype_folder + 'data.joblib' )
            dump( self.OG_columns, self.io.save_folder+'OGColumns.joblib')
        """
        
        self.metadata = DataFrame(self.metadata, columns=self.MetadataHeader())
        self.metadata.to_csv(phenotype_folder + 'metadata.csv', index=None)
        
        Taxonomy(self.metadata, phenotype_folder )
        
        print('Selected data and metadata saved in directory: ' + phenotype_folder)