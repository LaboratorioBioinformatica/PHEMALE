import os
import pandas
pandas.options.display.max_colwidth = 2000  #aumenta limite de dados do pandas
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

        madin = pandas.read_csv( self.data_folder + 'Madin_species.csv', sep=',')
        madin = madin[madin.superkingdom != 'Archaea'] # remove archaea
        madin.rename(columns = {'species_tax_id':'taxid'}, inplace = True) # renames column
        madin.rename(columns = {'class':'classes'}, inplace = True) # renames column
        madin = madin.dropna(subset=[phenotype]) # drops lines with no information about the phenotype of interest
        madin.rename(columns = {phenotype:'phenotype1'}, inplace = True) # renames column
        madin = madin[['taxid','data_source','genus','classes','order','family','species','phenotype1']]
        
        # if phenotype is numeric
        if str(phenotype+'.stdev') in madin.columns:
            
            # phenotype2 = Higher numerical value = mean + stdev
            madin['phenotype2'] = madin['phenotype1'].add(madin[phenotype+'.stdev'])
            # phenotype1 = Lower numerical value = mean - stdev
            madin['phenotype1'] = madin['phenotype1'].sub(madin[phenotype+'.stdev'])
            madin = madin[['taxid','data_source','genus','classes','order','family','species', 
                           'phenotype1','phenotype2']]
        
        elif phenotype == 'range_salinity':

            # phenotype1 = halophilic
            # phenotype2 = non-halophilic
            madin['phenotype2'] = 'no'

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

        madin.to_csv( self.data_folder + phenotype + '/madin_'+phenotype+'.csv', index=False)

        return madin

    """
    Function: Gets file with information of bacterias with complete genomes in NCBI server.
    """
    def ParseNCBI(self):
        if os.path.isfile( self.data_folder + 'assembly_summary.txt') == False:
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
        
        return ncbi
    
    """
    Function: Runs eggnog-mapper on genome fasta file using prodigal
    Parameters: file - genome fasta file
                n_jobs - number of CPU cores to parallel eggnog-mapper
    """
    def RunEggNOG(self, file, n_jobs):
        os.system('./eggnog-mapper/emapper.py -i ' + file + '_genomic.fna --itype genome --genepred prodigal -o '+ file + 
                  ' --cpu ' + str(n_jobs) + ' --tax_scope 2 --tax_scope_mode inner_broadest --override')

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
        
        self.data_folder = './metadata/'
        
        # creates results folder
        if os.path.isdir( self.data_folder ) == False:
            os.mkdir( self.data_folder )
        
        # creates folder for results of specific phenotype
        if os.path.isdir( self.data_folder + phenotype ) == False:
            os.mkdir( self.data_folder + phenotype )
        
        madin = self.ParseMadin( phenotype, specific_pathway )
        ncbi = self.ParseNCBI()

        # maximum number of genomes collected per species
        number_genomes_per_species = 5
        
        self.CreateDatabase( madin, ncbi, number_genomes_per_species, n_jobs )

        print('Downloaded raw dataset')