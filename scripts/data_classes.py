import os
import sys
from numba import jit
import pandas
pandas.options.display.max_colwidth = 2000  #higher data threshold limit for pandas
import warnings
warnings.filterwarnings('ignore')

"""
Class: collecting and organizing training data.
Observations: data from different phenotypes must be treated differently in the ParseMadin() method.
Parameters: phenotype - phenotype of interest
            n_jobs - number of CPU cores to use with eggnog-mapper
"""
class CollectData:

    """
    Function: downloads a genome fasta file from the NCBI server
    Parameters: downloadpath - path to fasta file in NCBI server
                savefolder - folder to save file
                file - genome fasta file name
    """
    def DownloadData(self, downloadpath, savefolder, file):
        
        try:
            os.system('wget -q -P ' + savefolder + ' ' + downloadpath)
            os.system('gzip -d ' + savefolder + '/' + file + '_genomic.fna.gz')
        
        except:
            DownloadData(downloadpath, savefolder, file)

    """
    Function: parse the file of bacteria phenotypes from the Madin et al. study
    Observations: data from Madin is manipulated to better adapt for multiclasses/multilabels
    Parameters: phenotype - phenotype of interest
                pathway - if phenotype of interest is 'pathway' this parameter is to speficy which pathway
    """
    def ParseMadin(self, phenotype, specific_pathway ):

        os.system('wget --tries=20 --quiet -P ' + self.data_folder + ' https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/master/output/condensed_species_NCBI.csv')
        madin = pandas.read_csv( self.data_folder + 'condensed_species_NCBI.csv', sep=',')
        os.system('rm -r ' + self.data_folder + 'condensed_species_NCBI.csv')
        
        madin = madin[madin.superkingdom != 'Archaea'] # remove archaea
        madin.rename(columns = {'species_tax_id':'taxid'}, inplace = True) # renames column
        madin.rename(columns = {'class':'classes'}, inplace = True) # renames column
        madin = madin.dropna(subset=[phenotype]) # drops lines with no information about the phenotype of interest
        madin.rename(columns = {phenotype:'phenotype1'}, inplace = True) # renames column
        
        # if phenotype is numeric
        if phenotype == 'optimum_tmp' or phenotype == 'optimum_ph':

            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0.0, 0.5)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(0.5)
            madin['phenotype2'] = madin['phenotype1'].add(madin[phenotype+'.stdev']) # phenotype2 = Higher numerical value = mean + stdev
            madin['phenotype1'] = madin['phenotype1'].sub(madin[phenotype+'.stdev']) # phenotype1 = Lower numerical value = mean - stdev
        
        elif phenotype == 'range_salinity':

            # phenotype1 = halophilic
            madin['phenotype2'] = 'no' # phenotype2 = non-halophilic

            for idx, row in madin.iterrows():
                if row.phenotype1 == 'extreme-halophilic' or row.phenotype1 == 'moderate-halophilic' or row.phenotype1 == 'halophilic':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'no'
                elif row.phenotype1 == 'non-halophilic':
                    madin['phenotype1'][idx] = 'no'
                    madin['phenotype2'][idx] = 'yes'
                elif row.phenotype1 == 'halotolerant':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'yes'

        elif phenotype == 'pathways':
            for idx, row in madin.iterrows():
                if specific_pathway in row.phenotype1:
                    madin['phenotype1'][idx] = 'yes'
                else:
                    madin['phenotype1'][idx] = 'no'
                    
        elif phenotype == 'range_tmp':

            # phenotype1 = mesophilic
            # phenotype2 = psychrophilic
            madin['phenotype2'] = 'no'
            # phenotype3 = thermophilic
            madin['phenotype3'] = 'no'

            for idx, row in madin.iterrows():
                if row.phenotype1 == 'mesophilic':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'no'
                    madin['phenotype3'][idx] = 'no'
                elif row.phenotype1 == 'extreme thermophilic' or row.phenotype1 == 'thermophilic':
                    madin['phenotype1'][idx] = 'no'
                    madin['phenotype2'][idx] = 'no'
                    madin['phenotype3'][idx] = 'yes'
                elif row.phenotype1 == 'facultative thermotolerant' or row.phenotype1 == 'thermotolerant':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'no'
                    madin['phenotype3'][idx] = 'yes'
                elif row.phenotype1 == 'facultative psychrophilic' or row.phenotype1 == 'psychrotolerant':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'yes'
                    madin['phenotype3'][idx] = 'no'
                elif row.phenotype1 == 'psychrophilic':
                    madin['phenotype1'][idx] = 'no'
                    madin['phenotype2'][idx] = 'yes'
                    madin['phenotype3'][idx] = 'no'

        madin.to_csv( self.data_folder + phenotype + '/data/madin_'+phenotype+'.csv', index=False)
        return madin

    """
    Function: Gets file with information of bacterias with complete genomes in NCBI server.
    """
    def ParseNCBI(self):
        os.system('wget --tries=20 --quiet -P ' + self.data_folder +
                      ' https://ftp.ncbi.nlm.nih.gov/genomes/genbank/bacteria/assembly_summary.txt')
        ncbi = open( self.data_folder + 'assembly_summary.txt','r').readlines()
        del ncbi[0] # remove first comment line
        ncbi[0] = ncbi[0][2:] # remove comment characters
        ncbi = [line.rstrip() for line in ncbi]
        ncbi = [line_component.split("\t") for line_component in ncbi]
        ncbi = pandas.DataFrame(data=ncbi[1:], columns=ncbi[0])
        ncbi = ncbi[ncbi.assembly_level != 'Contig'] # removes contig level of completeness
        ncbi = ncbi[ncbi.assembly_level != 'Scaffold'] # removes scaffold level of completeness
        ncbi = ncbi[['assembly_accession','assembly_level','taxid','organism_name','ftp_path']]
        os.system('rm -r ' + self.data_folder + 'assembly_summary.txt')
        
        return ncbi
    
    """
    Function: Runs eggnog-mapper on genome fasta file using prodigal
    Parameters: file - genome fasta file
                n_jobs - number of CPU cores dedicated to eggnog-mapper
    """
    def RunEggNOG(self, file, n_jobs):
        os.system('./eggnog-mapper/emapper.py -i ' + file + '_genomic.fna --itype genome --genepred prodigal -o '+ file + 
                  ' --cpu ' + str(n_jobs) + ' --tax_scope Bacteria --tax_scope_mode inner_broadest --temp_dir ./eggnog-mapper/temp --override')

    """
    Function: 
    Parameters: madin - 
                ncbi - file with information of bacterias genomes in NCBI server
                number_genomes_per_species - maximum number of genomes collected per species
                n_jobs - number of CPU cores to use with eggnog-mapper
    """
    def CreateDatabase(self, madin, ncbi, number_genomes_per_species, n_jobs):
    
        for idx_madin, row_madin in madin.iterrows():

            madin_ncbi = ncbi.loc[ncbi.taxid == str( row_madin.taxid )]
            madin_ncbi = madin_ncbi[:number_genomes_per_species]

            if len(madin_ncbi.index) > 0: # check if dataframe is not empty
                
                folder = '../data/genomes/'+str(row_madin.taxid)+'/'

                if os.path.isdir( folder ) == False:
                    os.mkdir( folder )
                
                for idx_madin_ncbi, row_madin_ncbi in madin_ncbi.iterrows():
                    accession = row_madin_ncbi.ftp_path.split('/',9)[-1]
                    
                    if os.path.isfile( folder + accession + '.emapper.annotations') == True:
                        continue #if eggnog-mapper file already exists skip
                    else:
                        self.DownloadData( row_madin_ncbi.ftp_path+'/'+accession+'_genomic.fna.gz', folder, accession)
                        self.RunEggNOG( folder + accession, n_jobs)
                
                for file in os.listdir( folder ):
                    if file.endswith('.emapper.annotations') == False and file.endswith('.joblib') == False:
                        os.system('rm -r ' + folder + file)
    
    def __init__(self, phenotype, n_jobs, specific_pathway = False):
        
        self.data_folder = './results/'
        
        # creates results folder
        if os.path.isdir( self.data_folder + phenotype + '/data' ) == False:
            os.makedirs( self.data_folder + phenotype + '/data' )
        
        madin = self.ParseMadin( phenotype, specific_pathway )
        ncbi = self.ParseNCBI()

        # maximum number of genomes collected per species
        number_genomes_per_species = 3
        
        self.CreateDatabase( madin, ncbi, number_genomes_per_species, n_jobs )

        print('Downloaded raw dataset')
        
#####################################################################################################################

from joblib import load, dump
import numpy
numpy.seterr(divide = 'ignore')

class TransformData:

    def ParseEggNOGFile(self, file):
        
        with open(file) as annotations:
            try:
                annotations = [line.rstrip() for line in annotations]
                annotations = [component.split("\t") for component in annotations]
                del annotations[0:4]                                     # deletando as 4 linhas do início
                del annotations[ len(annotations)-3 : len(annotations) ] # deletando as 3 linhas do fim
                annotations = pandas.DataFrame(data=annotations[1:], columns=annotations[0])
                return annotations
            except UnicodeDecodeError:
                self.problems = True
                os.system('rm ' + file)
            except IndexError:
                self.problems = True
                os.system('rm ' + file)
        return [0]
    
    #values suggested by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3820096/
    @jit
    def TransformEVALUE(self, evalue):

        maxValue = 1.0
        middle = 20.0
        curve = middle/3
        threshold = 10**(-6)
        
        if evalue > threshold:
            return 0
        elif evalue == 0:
            return maxValue
        else:
            evalue = -numpy.log10( float(evalue) )
            evalue = (evalue-middle)/curve
            return round( (maxValue / (1 + numpy.exp(-evalue))), 2 )
    
    #Parameters 
    #ortholog_groups_DB: 1 or 2; 1 for curated groups (COG and COG+); 2 for hypothetical groups (COG, COG+ and EGGNOG);
    def TransformEggNOGFile(self, annotation, ortholog_groups_DB):
        
        genome = [0]*len(self.OG_columns)
        
        for idx, row in annotation.iterrows():
        
            try:
                OG = row.eggNOG_OGs.split(',')[ortholog_groups_DB].split('@')[0]
                transformedEvalue = self.TransformEVALUE(float(row.evalue))

                try:
                    index = self.OG_columns.index(OG)
                    genome[index] = genome[index] + transformedEvalue

                except ValueError:
                    self.OG_columns.append(OG)
                    genome.append( transformedEvalue )
            
            except IndexError:
                continue
        
            except AttributeError:
                continue

        return genome

    def __init__(self, phenotype, ortholog_groups_DB = 1 ):
        
        self.problems = False
        self.OG_columns = []
        self.ortholog_groups_DB = ortholog_groups_DB
        
        madin = pandas.read_csv('./results/'+phenotype+'/data/'+'madin_'+phenotype+'.csv')
        
        for idx in madin.index:
            
            folder = '../data/genomes/'+str(madin['taxid'][idx])+'/'
            
            if os.path.isdir(folder):
                for eggnog_file in os.listdir(folder):
                    if eggnog_file.endswith('.annotations'):
                        annotation = self.ParseEggNOGFile( folder + eggnog_file )
                        if self.problems == False:
                            genome = self.TransformEggNOGFile( annotation, self.ortholog_groups_DB )
                            dump( genome, folder + eggnog_file + phenotype + '.joblib' )

        if self.problems == False:
            dump( self.OG_columns, './results/'+phenotype+'/data/OGColumns.joblib')
            print('Processed data')
        else:
            sys.exit('Error found and corrected on data collection. Run program again.')

#####################################################################################################################

import statistics
from scipy.stats import pearsonr
import math
import matplotlib.colors as mcolors
import random

class MountDataset:

    @jit
    def CheckRedundancy(self, datum, data, threshold):
        if len(data) != 0:
            for i in data:
                if i[self.number_of_OG_columns:] == datum[self.number_of_OG_columns:]:
                    if pearsonr( i[:self.number_of_OG_columns] , datum[:self.number_of_OG_columns] )[0] > threshold:
                        return False
        return True
    
    @jit
    def SelectFromFolderByMinPearson(self, folder, n_files):
        files = []
        pearson_values = []
        pearson_files = []

        for file in os.listdir(folder):
            if file.endswith( '.annotations' + self.phenotype + '.joblib' ):
                file = folder + file
                genome = load( file )
                genome = genome + [0]*( self.number_of_OG_columns - len( genome ))
                for file2 in files:
                    genome2 = load( file2 )
                    genome2 = genome2 + [0]*( self.number_of_OG_columns - len( genome2 ))
                    pearson_values.append( pearsonr( genome , genome2 )[0] )
                    pearson_files.append( (file2, file) )
                files.append( file )
        
        if len(files) > 1:
            min_pearson_indexes = numpy.argsort(pearson_values)[n_files:]
            files = numpy.unique( numpy.delete( numpy.array(pearson_files), min_pearson_indexes) )[:n_files]
        
        return files
    
    @jit
    def CalculatePearsonThreshold(self):
        pearsons = []

        for idx, row in self.madin.iterrows():
            pearson = []
            files = []
            folder = '../data/genomes/'+str(row['taxid'])+'/'

            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith( '.annotations' + self.phenotype + '.joblib' ):
                        file = load( folder+file )
                        file = file + [0]*( self.number_of_OG_columns - len( file ))
                        if len(files) > 0:
                            for i in files:
                                pearson.append( pearsonr( i , file )[0] )
                        files.append( file )

            if len(pearson) > 1:
                pearsons.append(statistics.mean(pearson))
            if len(pearson) == 1:
                pearsons.append(pearson[0])

        return math.ceil( statistics.mean(pearsons)*100 )/100
    
    def GenerateMetadataHeader( self ):
        
        metadata = [['genome', 'class', 'order', 'family', 'genus', 'species']]
        
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_ph':
            metadata[0] = metadata[0] + ['phenotype1','phenotype2']
        elif self.phenotype == 'range_tmp':
            metadata[0] = metadata[0] + ['phenotype1','phenotype2','phenotype3']
        elif self.phenotype == 'pathways' or self.phenotype == 'motility':
            metadata[0] = metadata[0] + ['phenotype1']
            
        return metadata
    
    @jit
    def GenerateDatum( self, file, row ):
        
        genome = load( file )
        genome = genome + [0]*( self.number_of_OG_columns - len( genome ))
        
        metadatum = [file, row.classes, row.order, row.family, row.genus, row.species]
        
        append = None
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_tmp':
            append = [ row.phenotype1 , row.phenotype2 ]
        
        if self.phenotype == 'optimum_ph':
            #append = [ 10**(8-row.phenotype1) , 10**(8-row.phenotype2) ] #other scale option for pH
            append = [ (10**3)*(2**(-row.phenotype1)) , (10**3)*(2**(-row.phenotype2)) ]
        elif self.phenotype == 'range_tmp':
            append = [ row.phenotype1 , row.phenotype2 , row.phenotype3 ]
        else:
            append = [ row.phenotype1 ]
        
        genome = genome + append
        metadatum = metadatum + append
        
        return genome, metadatum
    
    @jit
    def MountDataset(self, madin, redundancy_threshold = False, maximum_diversity = True, max_files_per_species = 3 ):
        
        data = []
        metadata = self.GenerateMetadataHeader()
        
        for idx, row in madin.iterrows():
            
            folder = '../data/genomes/'+str(row['taxid'])+'/'
            
            if os.path.isdir(folder):
                
                if maximum_diversity == True:
                    for file in self.SelectFromFolderByMinPearson( folder, max_files_per_species ):
                        genome, metadatum = self.GenerateDatum( file, row )
                        data.append( genome )
                        metadata.append( metadatum )
                
                else:
                    for file in os.listdir(folder):
                        n_files_species = 0
                        if file.endswith( '.annotations' + self.phenotype + '.joblib' ):
                            genome, metadatum = self.GenerateDatum( folder + file, row )
                            if (redundancy_threshold != False and self.CheckRedundancy( genome, data, redundancy_threshold ) == True) == True or redundancy_threshold == False and n_files_species < max_files_per_species:
                                n_files_species = n_files_species + 1
                                data.append( genome )
                                metadata.append( metadatum )

        return data, metadata
    
    def Taxonomy(self, metadata, saveFolder):
        
        metadata = pandas.DataFrame(data=metadata[1:], columns=metadata[0])
        order = metadata.order.value_counts().rename_axis('data').reset_index(name='counts')
        families = len( metadata.family.value_counts().rename_axis('data').reset_index(name='counts') )
        genus = len( metadata.genus.value_counts().rename_axis('data').reset_index(name='counts') )
        species = len( metadata.species.value_counts().rename_axis('data').reset_index(name='counts') )
                          
        with open(saveFolder+'metametadata.log', 'w') as file:
            file.write('Number of orders: ' + str(len(order)) + '\n')
            file.write('Number of families: ' + str(families) + '\n')
            file.write('Number of genus: ' + str(genus) + '\n')
            file.write('Number of species: ' + str(species) + '\n')
            file.write('Number of OGs: ' + str(self.number_of_OG_columns))
            
        colors = random.choices( list(mcolors.CSS4_COLORS.values()) , k = len(order) )
        orderGraph = order.plot(y='counts', kind='pie', figsize=(5, 5), labels = order.data, labeldistance=None, colors=colors)
        orderGraph.legend(loc="right", bbox_to_anchor=(3.0,0.5), fontsize=8, ncol=4)
        orderGraph.figure.savefig(saveFolder + 'order.png', bbox_inches="tight", dpi=800)
            
    def __init__(self, phenotype):
        
        self.phenotype = phenotype
        phenotype_folder = './results/'+phenotype+'/data/'
        
        self.madin = pandas.read_csv( phenotype_folder+'madin_'+self.phenotype+'.csv')
        self.number_of_OG_columns = len( load(phenotype_folder+'OGColumns.joblib') )

        """
        data, metadata = self.MountDataset( self.madin, 
                                           redundancy_threshold = self.CalculatePearsonThreshold(), 
                                           maximum_diversity = False, 
                                           max_files_per_species = 3 )
        """
        data, metadata = self.MountDataset( self.madin, 
                                           redundancy_threshold = False, 
                                           maximum_diversity = True, 
                                           max_files_per_species = 2 )
        
        dump( metadata, phenotype_folder + 'metadata.joblib' )
        dump( data, phenotype_folder + 'data.joblib' )
        self.Taxonomy(metadata, phenotype_folder )
        
        print('Dataset & metadata constructed in directory: ' + phenotype_folder)
        
#####################################################################################################################

from datetime import date, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score, multilabel_confusion_matrix, confusion_matrix, classification_report
import matplotlib.pyplot as plt

"""
Class: input and output of data
Parameters: phenotype - phenotype of interest
            classification_or_regression - if the data is numerical (regression) or label (classification)
"""
class DataIO:

    def __init__(self, phenotype, classification_or_regression):
        
        self.startTime = datetime.now()
        
        self.phenotype = phenotype
        self.phenotype_directory = './results/'+phenotype+'/data/'
        
        if os.path.isdir( './results/' ) == False:
            os.mkdir('./results/')
        
        self.results_directory = './results/'+self.phenotype+'/'
        if os.path.isdir( self.results_directory ) == False:
            os.mkdir(self.results_directory)
        
        self.classification_or_regression = classification_or_regression
    
    def WriteLog(self, text):
        
        if hasattr(self, 'results_ID_directory') == False:
            self.results_ID_directory = self.results_directory+'#'+ self.startTime.strftime("%Y-%m-%d_%H:%M") +'/'
            os.mkdir(self.results_ID_directory)

        with open(self.results_ID_directory+'log.txt', 'a+') as file:
            file.write(str(text)+'\n')

    def SaveModel(self, model, file_name):
        dump(model, self.results_ID_directory+file_name+'.joblib')

    @jit
    def GetTrainingData(self, file, splitTrainTest = False, labelize = False):

        number_of_OG_columns = len(load(self.phenotype_directory+'OGColumns.joblib') )
        
        x = numpy.asarray( load( self.phenotype_directory + file, mmap_mode='r+' ) )
        x, y = numpy.hsplit(x, [number_of_OG_columns])
        x = numpy.asarray( x, dtype=numpy.float16)
        x = numpy.nan_to_num(x, nan=0)

        if self.classification_or_regression == 'classification' and labelize == True:
            labelize = MultiLabelBinarizer()
            y = [[i] for i in y]
            y = labelize.fit_transform(y)
        
        elif self.classification_or_regression == 'regression':
            y = numpy.asarray( y, dtype=numpy.float16)

        if splitTrainTest != False:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=splitTrainTest)
            return x_train, y_train, x_test, y_test, labelize

        return x, y, labelize
        
    def ParseParameters(self, dictionary):
        return { "estimator__"+str(key) : (transform(value) 
                 if isinstance(value, dict) else value) 
                 for key, value in dictionary.items() }
    
    """
    Function transform output to something interpretable to humans
    """
    def YTransform(self, y):
        #para casos como o PLSRegression que por algum motivo devolve output como uma lista de lista de listas ao invés de apenas uma lsita de listas.
        if len(y) == 1:
            y = y[0]
            
        y_relabeled = []
        
        if self.phenotype == 'range_salinity':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes':
                    y_relabeled.append('halotolerant')
                elif y[i][0] == 'yes' and y[i][1] == 'no':
                    y_relabeled.append('halophilic')
                elif y[i][0] == 'no' and y[i][1] == 'yes':
                    y_relabeled.append('non-halophilic')
                elif y[i][0] == 'no' and y[i][1] == 'no':
                    y_relabeled.append('none')
        elif self.phenotype == 'range_tmp':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y_relabeled.append('mesophilic, psychrophilic and thermophilic')
                elif y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y_relabeled.append('mesophilic and thermophilic')
                elif y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y_relabeled.append('mesophilic and psychrophilic')
                elif y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y_relabeled.append('psychrophilic and thermophilic')
                if y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'no':
                    y_relabeled.append('mesophilic')
                if y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y_relabeled.append('psychrophilic')
                if y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y_relabeled.append('thermophilic')
                elif y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'no':
                    y_relabeled.append('none')
        elif self.phenotype == 'optimum_ph':
            try:
                y_relabeled = numpy.asarray([[ -numpy.log2(j/(10**3)) for j in i] for i in y], dtype=numpy.float16)
            except ValueError:
                sys.exit('Training went awry. Run program again.')
        else:
            return y

        return y_relabeled

    def Metrics(self, y_test, y_pred, name):
        
        y_test = self.YTransform(y_test)
        y_pred = self.YTransform(y_pred)
        
        if self.classification_or_regression == 'classification':
        
            self.WriteLog( multilabel_confusion_matrix( y_test, y_pred ) )
            self.WriteLog( classification_report( y_test, y_pred ) )

            #implementar retorno de alguma métrica (talvez AUC)
            #https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb
            #return

        elif self.classification_or_regression == 'regression':

            y_test_lower, y_test_higher = numpy.hsplit(y_test,[1])
            y_pred_lower, y_pred_higher = numpy.hsplit(y_pred,[1])
            
            y_test_lower = numpy.ravel(y_test_lower)
            y_test_higher = numpy.ravel(y_test_higher)
            y_pred_lower = numpy.ravel(y_pred_lower)
            y_pred_higher = numpy.ravel(y_pred_higher)

            number_of_not_predicted = 0
            
            index_nan = numpy.argwhere( numpy.isnan( y_pred_lower ) )
            number_of_not_predicted = len(index_nan)
            y_pred_lower = numpy.delete( y_pred_lower, index_nan )
            y_test_lower = numpy.delete( y_test_lower, index_nan )
            y_pred_higher = numpy.delete( y_pred_higher, index_nan )
            y_test_higher = numpy.delete( y_test_higher, index_nan )
            
            index_nan = numpy.argwhere( numpy.isnan( y_pred_higher ) )
            number_of_not_predicted = number_of_not_predicted + len(index_nan)
            y_pred_lower = numpy.delete( y_pred_lower, index_nan )
            y_test_lower = numpy.delete( y_test_lower, index_nan )
            y_pred_higher = numpy.delete( y_pred_higher, index_nan )
            y_test_higher = numpy.delete( y_test_higher, index_nan )
            
            r2_lower = r2_score(y_test_lower, y_pred_lower)
            r2_higher = r2_score(y_test_higher, y_pred_higher)

            self.WriteLog( 'Number of values not pedicted: ' + str(number_of_not_predicted))
            self.WriteLog( 'R2 score of lower values: %.2f' % r2_lower )
            self.WriteLog( 'R2 score of higher values: %.2f' % r2_higher )
            self.WriteLog( '' )
            self.Graphs(y_test, y_pred, name)

    @jit
    def Graphs(self, y_test, y_pred, name):

        y_test_lower, y_test_higher = numpy.hsplit(y_test,[1])
        y_pred_lower, y_pred_higher = numpy.hsplit(y_pred,[1])
        
        y_test_lower = numpy.ravel(y_test_lower)
        y_test_higher = numpy.ravel(y_test_higher)
        y_pred_lower = numpy.ravel(y_pred_lower)
        y_pred_higher = numpy.ravel(y_pred_higher)

        x = [(i + j) / 2 for i, j in zip(y_test_lower, y_test_higher)]
        x_uncertainty = [abs(i - j) / 2 for i, j in zip(y_test_lower, y_test_higher)]
        y = [(i + j) / 2 for i, j in zip(y_pred_lower, y_pred_higher)]
        y_uncertainty = [abs(i - j) / 2 for i, j in zip(y_pred_lower, y_pred_higher)]
        
        plt.figure( (datetime.now() - self.startTime).total_seconds() )
        
        plt.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty,
                     marker='.', markerfacecolor='red', markersize=6,
                     linestyle='none', capsize=0, elinewidth=0.2)
        #plt.plot([0, 1], [0, 1], transform=plt.transAxes)
        plt.axline((0, 0), slope=1) # identity line for reference
        plt.savefig(self.results_ID_directory + name + '.png', bbox_inches="tight", dpi=600, format='png')