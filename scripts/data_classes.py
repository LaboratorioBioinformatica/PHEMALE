import os
from numba import jit
import pandas
pandas.options.display.max_colwidth = 2000  #higher data threshold limit for pandas
from warnings import filterwarnings
filterwarnings('ignore')

"""
Class: collecting and organizing training data.
Observations: data from different phenotypes must be treated differently in the ParseMadin() method.
Parameters: phenotype - phenotype of interest
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
            os.system('gzip -d ' + savefolder + file)
        except:
            DownloadData(downloadpath, savefolder, file)
            
    """
    Function: parse the file of bacteria phenotypes from the Madin et al. study
    Observations: data from Madin is manipulated to better adapt for multiclasses/multilabels
    Parameters: phenotype - phenotype of interest
                pathway - if phenotype of interest is 'pathway' this parameter is to specify which is the pathway of interest
    """
    @jit
    def ParseMadin(self, phenotype ):
        os.system('wget --tries=20 --quiet -P ' + self.data_folder + ' https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/master/output/condensed_species_NCBI.csv')
        madin = pandas.read_csv( self.data_folder + 'condensed_species_NCBI.csv', sep=',')
        os.system('rm -r ' + self.data_folder + 'condensed_species_NCBI.csv')
        
        madin = madin[madin.superkingdom != 'Archaea'] # remove archaea
        madin.rename(columns = {'species_tax_id':'taxid'}, inplace = True) # renames column
        madin.rename(columns = {'class':'classes'}, inplace = True) # renames column
        
        if phenotype in ['nitrogen_fixation','nitrate_reduction','fermentation','sulfate_reduction']:
            madin = madin.dropna(subset=['pathways']) # drops lines with no information about the phenotype of interest
            madin.rename(columns = {'pathways':'phenotype1'}, inplace = True) # renames column
        else:
            madin = madin.dropna(subset=[phenotype]) # drops lines with no information about the phenotype of interest
            madin.rename(columns = {phenotype:'phenotype1'}, inplace = True) # renames column
        
        if phenotype == 'optimum_tmp':
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0.0, 2.0)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(2.0)
            madin['phenotype2'] = madin['phenotype1'].add(madin[phenotype+'.stdev']) #phenotype2 = higher value = mean + stdev
            madin['phenotype1'] = madin['phenotype1'].sub(madin[phenotype+'.stdev']) # phenotype1 = lower value = mean - stdev

        elif phenotype == 'optimum_ph':
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0, 0.5)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(0.5)
            madin['phenotype2'] = madin['phenotype1'].sub(madin[phenotype+'.stdev']) # phenotype2 = higher value = mean + stdev
            madin['phenotype1'] = madin['phenotype1'].add(madin[phenotype+'.stdev']) # phenotype1 = lower value = mean - stdev
        
        elif phenotype == 'range_salinity':
            # phenotype1 = halophilic
            madin['phenotype2'] = 'no' # phenotype2 = non-halophilic
            for idx, row in madin.iterrows():
                if row.phenotype1=='extreme-halophilic' or row.phenotype1=='moderate-halophilic' or row.phenotype1=='halophilic':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'no'
                elif row.phenotype1 == 'non-halophilic':
                    madin['phenotype1'][idx] = 'no'
                    madin['phenotype2'][idx] = 'yes'
                elif row.phenotype1 == 'halotolerant':
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'yes'

        elif phenotype in ['nitrogen_fixation','nitrate_reduction','fermentation','sulfate_reduction']:
            for idx, row in madin.iterrows():
                if phenotype in row.phenotype1:
                    madin['phenotype1'][idx] = 'yes'
                else:
                    madin['phenotype1'][idx] = 'no'
                    
        elif phenotype == 'range_tmp':
            # phenotype1 = mesophilic
            madin['phenotype2'] = 'no' # phenotype2 = psychrophilic
            madin['phenotype3'] = 'no' # phenotype3 = thermophilic

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
    @jit
    def ParseNCBI(self):
        os.system('wget --tries=40 --quiet -P ' + self.data_folder + 
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
    Parameters: madin - 
                ncbi - file with information of bacterias genomes in NCBI server
                number_genomes_per_species - maximum number of genomes collected per species
    """
    @jit
    def CreateGenomeDatabase(self, madin, ncbi, number_genomes_per_species):        
        for idx_madin, row_madin in madin.iterrows():
            madin_ncbi = ncbi.loc[ncbi.taxid == str( row_madin.taxid )]
            madin_ncbi = madin_ncbi[:number_genomes_per_species]            
            if len(madin_ncbi) > 0:
                folder = '../data/genomes/'+str(row_madin.taxid)+'/'
                if os.path.isdir( folder ) == False:
                    os.mkdir( folder )
                    for idx_madin_ncbi, row_madin_ncbi in madin_ncbi.iterrows():
                        file = row_madin_ncbi.ftp_path.split('/',9)[-1] + '_genomic.fna'
                        if os.path.isfile( folder + file ) == False:
                            file = file + '.gz'
                            self.DownloadData( row_madin_ncbi.ftp_path+'/'+file, folder, file)
                        else:
                            phi120 = 0

    def __init__(self, phenotype, number_genomes_per_species = 5):
        self.data_folder = './results/'
        if os.path.isdir( self.data_folder + phenotype + '/data' ) == False:
            os.makedirs( self.data_folder + phenotype + '/data' )
        madin = self.ParseMadin( phenotype )
        if os.path.isfile( self.data_folder + phenotype + '/data/madin_'+phenotype+'_append.csv' ) == True:
            madin_append = pandas.read_csv( self.data_folder + phenotype + '/data/madin_'+phenotype+'_append.csv')
            madin = pandas.concat([madin, madin_append])
        ncbi = self.ParseNCBI()        
        self.CreateGenomeDatabase( madin, ncbi, number_genomes_per_species )
        print('Downloaded genomes database')
        
#####################################################################################################################

from joblib import load, dump
from heapq import nsmallest
import glob
import random
import matplotlib.colors as mcolors

class SelectData:
    """
    Function: calculates ANI between file1 and file2 with FastANI
    file1: genome file (.fna)
    file2: genome file (.fna)
    """
    def ANI(self, file1, file2):
        n_cpus = len(os.sched_getaffinity(0))
        os.system('./FastANI/fastANI -q ' + file1 + ' -r '+ file2 + ' -t ' + str(n_cpus) + ' -o ./FastANI/ani_'+self.phenotype+'.out')
        if os.stat('./FastANI/ani_'+self.phenotype+'.out').st_size != 0:
            ani = open( './FastANI/ani_'+self.phenotype+'.out','r').read()
            return float(ani.split()[2])
        else:
            return float(0.5)

    """
    Function: generate metadata of a specific genome file or generate metadata header if no parameters is passed
    filepath: genome file
    metadatum: row from a pandas dataframe with informations on class, order, family, genus and species
    """
    def GenerateMetadata( self, filepath = False, metadatum = False):
        if filepath == False and metadatum == False: #generates header
            metadata = ['file', 'class', 'order', 'family', 'genus', 'species']
            append = ['phenotype1']
            if self.phenotype in ['range_salinity','optimum_tmp','optimum_ph']:
                append = append + ['phenotype2']
            elif self.phenotype == 'range_tmp':
                append = append + ['phenotype2','phenotype3']
            
        else: #generates metadata of file
            metadata = [filepath, metadatum.classes, metadatum.order, metadatum.family, metadatum.genus, metadatum.species]
            append = [metadatum.phenotype1]
            if self.phenotype in ['range_salinity','optimum_tmp','optimum_ph']:
                append = append + [metadatum.phenotype2]
            elif self.phenotype in ['range_tmp']:
                append = append + [metadatum.phenotype2, metadatum.phenotype3]

        return metadata + append
    
    """
    Function: selects n_files .fna files with highest diversity (lowest ANI scores) inside specific folder
    folder: folder of genome .fna files
    n_files: max number of files requested per folder
    """
    def FilesWithLowestANIByFolder(self, folder, n_files):
        files = []
        ANI_values = []
        ANI_files = []
        
        for file1 in os.listdir(folder):
            if file1.endswith( '.fna' ):
                file1 = folder + file1
                for file2 in files:
                    ANI_values.append( self.ANI( file1, file2 ) )
                    ANI_files.append( (file1, file2) )
                files.append( file1 )
        
        if len(ANI_values) > 0:
            ANI_values, ANI_files = ( list(t) for t in zip(*sorted(zip(ANI_values, ANI_files))) )
            ANI_files = [item for sublist in ANI_files for item in sublist]
            ANI_files = list(dict.fromkeys(ANI_files)) # to remove duplicate instances but maintain order of appearance in list
            files = ANI_files[:n_files]
            rejected_files = ANI_files[3:]
            for rejected in rejected_files:
                os.system('rm ' + rejected)

        return files
    
    """
    Function: generates metadata for training containing
    max_files_per_species: max number of files requested per folder
    """
    def MountDataset(self, max_files_per_species):
        metadata = [ self.GenerateMetadata() ]
        for idx, row in self.madin.iterrows():
            folder = '../data/genomes/'+str(row['taxid'])+'/'
            if (os.path.isdir(folder)) and (len(os.listdir(folder)) > 0):
                files = self.FilesWithLowestANIByFolder( folder, max_files_per_species )
                for file in files:
                    metadata.append( self.GenerateMetadata( filepath = file, metadatum = row ) )
        return metadata
          
    def Taxonomy(self, metadata, savefolder):
        
        metadata = pandas.DataFrame(data=metadata[1:], columns=metadata[0])
        order = metadata.order.value_counts().rename_axis('data').reset_index(name='counts')
        families = len( metadata.family.value_counts().rename_axis('data').reset_index(name='counts') )
        genus = len( metadata.genus.value_counts().rename_axis('data').reset_index(name='counts') )
        species = len( metadata.species.value_counts().rename_axis('data').reset_index(name='counts') )
        specimens = len( metadata )
                          
        with open(savefolder+'metametadata.log', 'w') as file:
            file.write('Number of orders: ' + str(len(order)) + '\n')
            file.write('Number of families: ' + str(families) + '\n')
            file.write('Number of genus: ' + str(genus) + '\n')
            file.write('Number of species: ' + str(species) + '\n')
            file.write('Number of specimens: ' + str(specimens) + '\n')

        colors = random.choices( list(mcolors.CSS4_COLORS.values()) , k = len(order) )
        orderGraph = order.plot(y='counts', kind='pie', figsize=(5, 5), labels = order.data, labeldistance=None, colors=colors)
        orderGraph.legend(loc="right", bbox_to_anchor=(3.0,0.5), fontsize=8, ncol=4)
        orderGraph.figure.savefig(savefolder + 'order.png', bbox_inches="tight", dpi=800)
        
    def __init__(self, phenotype, specimens_per_species):
        
        self.phenotype = phenotype
        phenotype_folder = './results/'+phenotype+'/data/'
        self.madin = pandas.read_csv( phenotype_folder+'madin_'+self.phenotype+'.csv')
        if os.path.isfile( phenotype_folder + 'madin_'+self.phenotype+'_append.csv' ) == True:
            madin_append = pandas.read_csv( phenotype_folder + 'madin_'+self.phenotype+'_append.csv')
            self.madin = pandas.concat([self.madin, madin_append])
        metadata = self.MountDataset( max_files_per_species = specimens_per_species )
        dump( metadata, phenotype_folder + 'metadata.joblib' )
        self.Taxonomy(metadata, phenotype_folder )
        print('Selected data and metadata saved in directory: ' + phenotype_folder)
        
#####################################################################################################################

import csv
from random import sample
import numpy
numpy.seterr(divide = 'ignore')

class MountData:

    def RunEggNOG(self, file):
        if os.path.isdir( './eggnog-mapper/temp' ) == False:
            os.mkdir( './eggnog-mapper/temp' )
        
        """
        --sensmode DIAMOND_SENS_MODE
            default, fast, mid-sensitive, sensitive, more-sensitive, very-sensitive or ultra-sensitive. 
            Default of eggnog is sensitive. 
            Default of diamond is a little faster than mid-sensitive.
            More information in https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41592-021-01101-x/MediaObjects/41592_2021_1101_Fig1_HTML.png
        --index_chunks C
            The number of chunks for processing the seed index. This option can be used to tune the performance. 
            The default value is 4. Setting this parameter to 1 will improve the performance at the cost of increased memory use.
        --block_size B
            Block size in billions of sequence letters to be processed at a time. 
            This is the main parameter for controlling the program’s memory and disk space usage. 
            Bigger numbers will increase the use of memory and temporary disk space, but also improve performance. 
            The program can be expected to use roughly x6 this number of memory (in GB).
            The default value is 2.0. The parameter can be decreased for reducing memory use, as well as increased 
            for better performance (values of >20 are not recommended).
        The memory needed is roughly 20*B/C.
        """
        n_cpus = str(len(os.sched_getaffinity(0)))
        os.system('./eggnog-mapper/emapper.py -i '+ file + ' --itype genome --genepred prodigal -o ' + file + 
                  ' --cpu ' + n_cpus + ' --tax_scope 2 --tax_scope_mode broadest --override --no_file_comments '+
                  '--temp_dir ./eggnog-mapper/temp --dmnd_db ../data/databases/bacteria.dmnd --data_dir ../data/databases '+
                  '--dmnd_iterate yes --index_chunks 1 --block_size 0.4 --sensmode very-sensitive --target_orthologs one2one '+
                  '--outfmt_short')
        
        os.system('rm ' + file + '.emapper.genepred.fasta')
        os.system('rm ' + file + '.emapper.genepred.gff')
        os.system('rm ' + file + '.emapper.hits')
        os.system('rm ' + file + '.emapper.seed_orthologs')

    @jit
    def TransformEVALUE(self, evalue):

        maxValue = numpy.float16(1.0)
        threshold = numpy.float16(10**(-5))
        middle = numpy.float16(15.0)
        curve = numpy.float16(middle/3)
        
        if evalue > threshold:
            return numpy.float16(0)
        elif evalue == 0:
            return maxValue
        else:
            evalue = -numpy.log10( float(evalue) )
            evalue = (evalue-middle)/curve
            return numpy.float16( round( ( maxValue / (1 + numpy.exp(-evalue))), 2) )
    
    #Transform the eggnog file into a vector with quantities of each ortholog group present in eggnog
    def TransformFile(self, file, only_COGs):

        if os.path.isfile( file + '.emapper.annotations' ) == False:
            self.RunEggNOG(file)

        annotation = list( csv.reader(open(file + '.emapper.annotations'), delimiter='\t') )
        genome = [0]*len(self.OG_columns)
        
        for row in annotation[1:]:
            OG = row[4].split(',')[0].split('@')[0]
            if (only_COGs == True and 'COG' in OG) or only_COGs == False:
                transformedEvalue = self.TransformEVALUE( numpy.float16(row[2]) )
                try:
                    index = self.OG_columns.index(OG)
                    genome[index] = genome[index] + transformedEvalue
                except ValueError:
                    self.OG_columns.append(OG)
                    genome.append(transformedEvalue)
        dump( genome, file + '.' + self.phenotype + '.joblib' )

    def GenerateDatum( self, metadatum ):
        
        genome = load( metadatum[0] +'.'+ self.phenotype + '.joblib')
        genome = genome + [0]*( len(self.OG_columns) - len( genome ))
        append = None
        
        if self.phenotype == 'range_salinity' or self.phenotype == 'optimum_tmp':
            append = [ metadatum[-2] , metadatum[-1] ]
        elif self.phenotype == 'optimum_ph':
            append = [ (10**3)*(2**(-metadatum[-2])) , (10**3)*(2**(-metadatum[-1])) ]
            #append = [ (10**(14-metadatum[-2])) , (10**(14-metadatum[-1])) ]
        elif self.phenotype == 'range_tmp':
            append = [ metadatum[-3] , metadatum[-2] , metadatum[-1] ]
        else:
            append = [ metadatum[-1] ]        
        return genome + append
    
    def __init__(self, phenotype, only_COGs = True, sampling = 1.0 ):

        self.OG_columns = []
        self.phenotype = phenotype
        metadata = load('./results/'+phenotype+'/data/metadata.joblib')
        metadata = metadata[1:]
        
        for file in metadata:
            genome = self.TransformFile( file[0], only_COGs )

        dump( self.OG_columns, './results/' + phenotype + '/data/OGColumns.joblib')
        
        with open('./results/' + phenotype + '/data/metametadata.log', 'a') as file:
            file.write('Number of OGs: ' + str(len(self.OG_columns)))
        
        data = []
        for metadatum in sample( metadata, int(sampling*len(metadata)) ):
            data.append( self.GenerateDatum( metadatum ) )
        dump( data, './results/' + phenotype + '/data/data.joblib')
        
        print('Mounted data for training with onlyCOGs set to '+str(only_COGs))

#####################################################################################################################

from datetime import date, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score, multilabel_confusion_matrix, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys

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
            y = numpy.asarray( y )

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
        #para casos como o PLSRegression que por algum motivo devolve output como uma lista de lista de listas ao invés de apenas uma lista de listas.
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
        
            if self.phenotype == 'range_tmp' or self.phenotype == 'range_salinity': #multi-label phenotypes
                self.WriteLog( multilabel_confusion_matrix( y_test, y_pred ) )
            else:
                self.WriteLog( confusion_matrix( y_test, y_pred ) )
                
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
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty,
                     marker='.', markerfacecolor='red', markersize=6,
                     linestyle='none', capsize=0, elinewidth=0.2)
        plt.axline((0, 0), slope=1) # identity line for reference
        
        plt.savefig(self.results_ID_directory + name + '.png', bbox_inches="tight", dpi=600, format='png')