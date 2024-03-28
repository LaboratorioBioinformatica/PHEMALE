"""
Script to download genomes available in NCBI of interest to Madin
"""

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
        
        ### Get Madin's results ###
        os.system('wget --quiet -P ' + self.data_folder + ' https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/master/output/condensed_species_NCBI.csv')
        madin = pandas.read_csv( self.data_folder + 'condensed_species_NCBI.csv', sep=',')
        os.system('rm -r ' + self.data_folder + 'condensed_species_NCBI.csv')
        
        madin = madin[madin.superkingdom != 'Archaea'] # remove archaea
        madin.rename(columns = {'species_tax_id':'taxid'}, inplace = True)
        madin.rename(columns = {'class':'classes'}, inplace = True)

        # special case because these pheonotypes are information inside the column pathways
        if phenotype in ['nitrogen_fixation','nitrate_reduction','fermentation','sulfate_reduction']:
            madin = madin.dropna(subset=['pathways']) # drops lines with no information about the phenotype of interest
            madin.rename(columns = {'pathways':'phenotype1'}, inplace = True)
        else:
            madin = madin.dropna(subset=[phenotype]) # drops lines with no information about the phenotype of interest
            madin.rename(columns = {phenotype:'phenotype1'}, inplace = True)

        if phenotype == 'optimum_tmp':
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0.0, 5.0)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(5.0)
            madin['phenotype2'] = madin['phenotype1'].add(madin[phenotype+'.stdev']) # Higher value = mean + stdev
            madin['phenotype1'] = madin['phenotype1'].sub(madin[phenotype+'.stdev']) # Lower value = mean - stdev

        elif phenotype == 'optimum_ph':
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].replace( 0, 1.0)
            madin[phenotype+'.stdev'] = madin[phenotype+'.stdev'].fillna(1.0)
            madin['phenotype2'] = madin['phenotype1'].sub(madin[phenotype+'.stdev']) # mean + stdev
            madin['phenotype1'] = madin['phenotype1'].add(madin[phenotype+'.stdev']) # mean - stdev

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

        elif phenotype == 'metabolism':
            # phenotype1 = aerobic
            madin['phenotype2'] = 'no' # phenotype2 = anaerobic
            for idx, row in madin.iterrows():
                if row.phenotype1 in ['aerobic','obligate aerobic']:
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'no'
                elif row.phenotype1 in ['anaerobic','obligate anaerobic']:
                    madin['phenotype1'][idx] = 'no'
                    madin['phenotype2'][idx] = 'yes'
                elif row.phenotype1 in ['facultative','microaerophilic']:
                    madin['phenotype1'][idx] = 'yes'
                    madin['phenotype2'][idx] = 'yes'
                    
        madin.to_csv( self.data_folder + phenotype + '/data/madin.csv', index=False)
        return madin

    """
    Function: Gets file with information of bacterias with complete genomes in NCBI server.
    """
    @jit
    def ParseNCBI(self):
        os.system('wget --quiet -P '+self.data_folder+
                  ' https://ftp.ncbi.nlm.nih.gov/genomes/genbank/bacteria/assembly_summary.txt')
        ncbi = open( self.data_folder + 'assembly_summary.txt','r').readlines()
        del ncbi[0] # remove comment line
        ncbi[0] = ncbi[0][1:] # remove comment character
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
                genomes_per_species - maximum number of genomes collected per species
    """
    @jit
    def CreateGenomeDatabase(self, madin, ncbi, genomes_per_species):
        
        for idx_madin, row_madin in madin.iterrows():
            madin_ncbi = ncbi.loc[ncbi.taxid == str( row_madin.taxid )]
            madin_ncbi = madin_ncbi[:genomes_per_species]
            
            if len(madin_ncbi) > 0:
                folder = '../data/genomes/'+str(row_madin.taxid)+'/'
                os.makedirs(folder, exist_ok=True)
    
                for idx_madin_ncbi, row_madin_ncbi in madin_ncbi.iterrows():
                    file = row_madin_ncbi.ftp_path.split('/',9)[-1] + '_genomic.fna'

                    if os.path.isfile( folder + file ) == False:
                        file = file +'.gz'
                        self.DownloadData( row_madin_ncbi.ftp_path+'/'+file, folder, file)
                    
                    #else:
                        #phi120 = 0

    def __init__(self, phenotype, genomes_per_species = 1):
        
        self.data_folder = './results/'
        os.makedirs(self.data_folder + phenotype + '/data', exist_ok=True)
        
        madin = self.ParseMadin( phenotype )
        
        if os.path.isfile( 'madin_'+phenotype+'_append.csv' ) == True:
            madin_append = pandas.read_csv( 'madin_'+phenotype+'_append.csv')
            madin = pandas.concat([madin, madin_append])
        
        ncbi = self.ParseNCBI()        
        
        self.CreateGenomeDatabase( madin, ncbi, genomes_per_species )
        
        print('Downloaded genomes database.')