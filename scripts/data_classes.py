import os
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

            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0.0, 1.0)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(1.0)
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
                n_jobs - number of CPU cores to parallel eggnog-mapper
    """
    def RunEggNOG(self, file, n_jobs):
        os.system('./eggnog-mapper/emapper.py -i ' + file + '_genomic.fna --itype genome --genepred prodigal -o '+ file + 
                  ' --cpu ' + str(n_jobs) + ' --tax_scope 2 --tax_scope_mode inner_broadest --temp_dir ./eggnog-mapper/temp --override')

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

            for idx_madin_ncbi, row_madin_ncbi in madin_ncbi.iterrows():
                
                accession = row_madin_ncbi.ftp_path.split('/',9)[-1]
                folder = '../data/genomes/'+str(row_madin_ncbi.taxid)+'/'

                if os.path.isdir( folder ) == False:
                    os.mkdir( folder )

                # if eggnog-mapper file already exists go to next for iteration
                if os.path.isfile( folder + accession + '.emapper.annotations') == True:
                    continue

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
        number_genomes_per_species = 5
        
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
    
    @jit
    def TransformEVALUE(self, evalue):
        
        middle = 9
        curve = middle/5
        maxValue = 10
        
        evalue = -numpy.log10( float(evalue) )
        
        if evalue < 0:
            return 0
        elif evalue == 0:
            return maxValue
        
        evalue = (evalue-middle)/curve
        return round( (maxValue / (1 + numpy.exp(-evalue))), 1 )
    
    #Parameters 
    #ortholog_groups_DB: 1 or 2; 1 for curated groups (COG and COG+); 2 for hypothetical groups (COG, COG+ and EGGNOG);
    def TransformEggNOGFile(self, annotation, ortholog_groups_DB):
        
        genome = [0]*len(self.OG_columns)
        
        for idx, row in annotation.iterrows():
        
            try:
                OG = row.eggNOG_OGs.split(',')[ortholog_groups_DB].split('@')[0]
                transformedEvalue = self.TransformEVALUE(row.evalue)

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
            print('Error on data collection. Rerun from start.')
            quit()
            
#####################################################################################################################

from scipy.stats.stats import pearsonr

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
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_ph' or self.phenotype == 'optimum_tmp':
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
        phenotype_folder = './results/'+phenotype+'/data/'
        
        madin = pandas.read_csv( phenotype_folder+'madin_'+self.phenotype+'.csv')
        number_of_OG_columns = len( load(phenotype_folder+'OGColumns.joblib') )

        data, metadata = self.MountDataset( madin, number_of_OG_columns, redundancy_threshold )

        dump( metadata, phenotype_folder + 'metadata.joblib' )
        dump( data, phenotype_folder + 'data.joblib' )

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
            #self.results_ID_directory = self.results_directory+'#'+str( date.today() )+'/'
            self.results_ID_directory = self.results_directory+'#'+str( datetime.now() )+'/'
            os.mkdir(self.results_ID_directory)

        with open(self.results_ID_directory+'log.txt', 'a+') as file:
            file.write(str(text)+'\n')

    def SaveModel(self, model, file_name):
        dump(self.results_ID_directory+file_name+'.joblib', model)

    @jit
    def GetTrainingData(self, file, splitTrainTest = False, labelize = False):

        number_of_OG_columns = len(load(self.phenotype_directory+'OGColumns.joblib') )
        
        x = numpy.asarray( load( self.phenotype_directory + file ) )
        x, y = numpy.hsplit(x, [number_of_OG_columns])
        x = numpy.asarray( x, dtype=numpy.float16)
        x = numpy.nan_to_num(x, nan=0)

        if labelize == True:
            labelize = MultiLabelBinarizer()
            y = [[i] for i in y]
            y = labelize.fit_transform(y)

        if splitTrainTest != False:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=splitTrainTest)
            return x_train, y_train, x_test, y_test, labelize

        return x, y, labelize
        
    def ParseParameters(self, dictionary):
        return { "estimator__"+str(key) : (transform(value) 
                 if isinstance(value, dict) else value) 
                 for key, value in dictionary.items() }
    
    def LabelClassifier(self, y):
        
        if self.phenotype == 'range_salinity':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes':
                    y[i] = 'halotolerant'
                elif y[i][0] == 'yes' and y[i][1] == 'no':
                    y[i] = 'halophilic'
                elif y[i][0] == 'no' and y[i][1] == 'yes':
                    y[i] = 'non-halophilic'
                elif y[i][0] == 'no' and y[i][1] == 'no':
                    y[i] = 'none'

        if self.phenotype == 'range_tmp':
            for i in range(len(y)):
                if y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y[i] = 'all'
                    #y[i] = 'mesophilic, psychrophilic and thermophilic'
                elif y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y[i] = 'thermotolerant'
                    #y[i] = 'mesophilic and thermophilic'
                elif y[i][0] == 'yes' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y[i] = 'psychrotolerant'
                    #y[i] = 'mesophilic and psychrophilic'
                elif y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'yes':
                    y[i] = 'SongOfIce&Fire'
                    #y[i] = 'psychrophilic and thermophilic'
                if y[i][0] == 'yes' and y[i][1] == 'no' and y[i][2] == 'no':
                    y[i] = 'mesophilic'
                if y[i][0] == 'no' and y[i][1] == 'yes' and y[i][2] == 'no':
                    y[i] = 'psychrophilic'
                if y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'yes':
                    y[i] = 'thermophilic'
                elif y[i][0] == 'no' and y[i][1] == 'no' and y[i][2] == 'no':
                    y[i] = 'none'
        
        # verificar se precisa desta linha
        #y = numpy.hsplit(y,[1])[0]
        return y
    
    @jit
    def Metrics(self, y_test, y_pred):

        if self.classification_or_regression == 'classification':
            y_test = self.LabelClassifier(y_test)
            y_pred = self.LabelClassifier(y_pred)

            self.WriteLog( confusion_matrix( y_test, y_pred ) )
            self.WriteLog( classification_report( y_test, y_pred ) )
            
            #implementar retorno de alguma métrica (talvez AUC)
            #return 
            
        elif self.classification_or_regression == 'regression':
            y_test_lower, y_test_higher = numpy.hsplit(y_test,[1])
            y_pred_lower, y_pred_higher = numpy.hsplit(y_pred,[1])
            
            r2_lower = r2_score(y_test_lower, y_pred_lower)
            r2_higher = r2_score(y_test_higher, y_pred_higher)
            
            self.WriteLog( 'R2 score of lower values: '+ str(r2_lower) )
            self.WriteLog( 'R2 score of higher values: '+ str(r2_higher) )
            self.WriteLog( '' )
            
            return ( r2_lower, r2_higher )

    @jit
    def Graphs(self, y_test, y_pred):
        y_test_lower, y_test_higher = numpy.hsplit(y_test,[1])
        y_pred_lower, y_pred_higher = numpy.hsplit(y_pred,[1])

        y_test_lower = numpy.ravel(y_test_lower)
        y_test_higher = numpy.ravel(y_test_higher)
        y_pred_lower = numpy.ravel(y_pred_lower)
        y_pred_higher = numpy.ravel(y_pred_higher)

        plt.fill_between(x, y_test_lower, y_test_higher, color = 'red', alpha=0.15)
        plt.fill_between(x, y_pred_lower, y_pred_higher, color = 'cyan', alpha=0.15)

        plt.savefig(self.results_ID_directory + 'graph.png', bbox_inches="tight", dpi=600, format='png')