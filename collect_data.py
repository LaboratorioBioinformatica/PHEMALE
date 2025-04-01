"""
Script to download genomes available in NCBI of interest to Madin
"""

import os
from numba import jit
import pandas
import re
pandas.options.display.max_colwidth = 2000  #higher data threshold limit for pandas
from urllib.request import urlretrieve
from math import isnan
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

    def ParseBacDive(self,phenotype):
        if phenotype == 'pH':
            phenotype_file = 'https://bacdive.dsmz.de/advsearch/csv?fg%5B0%5D%5Bgc%5D=OR&fg%5B0%5D%5Bfl%5D%5B0%5D%5Bfd%5D=Domain&fg%5B0%5D%5Bfl%5D%5B0%5D%5Bfo%5D=exact&fg%5B0%5D%5Bfl%5D%5B0%5D%5Bfv%5D=Bacteria&fg%5B0%5D%5Bfl%5D%5B0%5D%5Bfvd%5D=strains-domain-1&fg%5B0%5D%5Bfl%5D%5B1%5D=AND&fg%5B0%5D%5Bfl%5D%5B2%5D%5Bfd%5D=16S+associated+NCBI+tax+ID&fg%5B0%5D%5Bfl%5D%5B2%5D%5Bfo%5D=smaller&fg%5B0%5D%5Bfl%5D%5B2%5D%5Bfv%5D=9999999999&fg%5B0%5D%5Bfl%5D%5B2%5D%5Bfvd%5D=sequence_16S-tax_id-7&fg%5B0%5D%5Bfl%5D%5B3%5D=AND&fg%5B0%5D%5Bfl%5D%5B4%5D%5Bfd%5D=Order&fg%5B0%5D%5Bfl%5D%5B4%5D%5Bfo%5D=contains&fg%5B0%5D%5Bfl%5D%5B4%5D%5Bfv%5D=%2A&fg%5B0%5D%5Bfl%5D%5B4%5D%5Bfvd%5D=strains-ordo-1&fg%5B0%5D%5Bfl%5D%5B5%5D=AND&fg%5B0%5D%5Bfl%5D%5B6%5D%5Bfd%5D=Family&fg%5B0%5D%5Bfl%5D%5B6%5D%5Bfo%5D=contains&fg%5B0%5D%5Bfl%5D%5B6%5D%5Bfv%5D=%2A&fg%5B0%5D%5Bfl%5D%5B6%5D%5Bfvd%5D=strains-family-1&fg%5B0%5D%5Bfl%5D%5B7%5D=AND&fg%5B0%5D%5Bfl%5D%5B8%5D%5Bfd%5D=pH&fg%5B0%5D%5Bfl%5D%5B8%5D%5Bfo%5D=smaller&fg%5B0%5D%5Bfl%5D%5B8%5D%5Bfv%5D=9999&fg%5B0%5D%5Bfl%5D%5B8%5D%5Bfvd%5D=culture_pH-pH-3'
        elif phenotype == 'Temperature':
            phenotype_file = 'https://bacdive.dsmz.de/advsearch/csv?fg%5B1%5D%5Bgc%5D=OR&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfd%5D=Domain&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfv%5D=Bacteria&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfvd%5D=strains-domain-1&fg%5B1%5D%5Bfl%5D%5B2%5D=AND&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfd%5D=16S+associated+NCBI+tax+ID&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfo%5D=NOT_equal&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfv%5D=0&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfvd%5D=sequence_16S-tax_id-7&fg%5B1%5D%5Bfl%5D%5B4%5D=AND&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfd%5D=Order&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfv%5D=%2A&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfvd%5D=strains-ordo-1&fg%5B1%5D%5Bfl%5D%5B6%5D=AND&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfd%5D=Temperature&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfo%5D=smaller&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfv%5D=99999999999&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfvd%5D=culture_temp-temp-3&fg%5B1%5D%5Bfl%5D%5B8%5D=AND&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfd%5D=Family&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfv%5D=%2A&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfvd%5D=strains-family-1'
        elif phenotype == 'Salt conc.':
            phenotype_file = 'https://bacdive.dsmz.de/advsearch/csv?fg%5B1%5D%5Bgc%5D=OR&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfd%5D=Domain&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfv%5D=Bacteria&fg%5B1%5D%5Bfl%5D%5B1%5D%5Bfvd%5D=strains-domain-1&fg%5B1%5D%5Bfl%5D%5B2%5D=AND&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfd%5D=Family&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfv%5D=%2A&fg%5B1%5D%5Bfl%5D%5B3%5D%5Bfvd%5D=strains-family-1&fg%5B1%5D%5Bfl%5D%5B4%5D=AND&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfd%5D=Order&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfo%5D=contains&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfv%5D=%2A&fg%5B1%5D%5Bfl%5D%5B5%5D%5Bfvd%5D=strains-ordo-1&fg%5B1%5D%5Bfl%5D%5B6%5D=AND&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfd%5D=Salt+conc.&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfo%5D=smaller&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfv%5D=99999999&fg%5B1%5D%5Bfl%5D%5B7%5D%5Bfvd%5D=halophily-salt_concentration-4&fg%5B1%5D%5Bfl%5D%5B8%5D=AND&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfd%5D=16S+associated+NCBI+tax+ID&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfo%5D=NOT_equal&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfv%5D=0&fg%5B1%5D%5Bfl%5D%5B9%5D%5Bfvd%5D=sequence_16S-tax_id-7'
                    
        if os.path.isfile( './results/'+phenotype+'/data/bacdive.csv' ) == False:
            urlretrieve(phenotype_file, './results/'+phenotype+'/data/bacdive.csv')
        bacdive = pandas.read_csv('./results/'+phenotype+'/data/bacdive.csv', skiprows=[0, 1], usecols=['ordo','species','family','16S associated NCBI tax ID',phenotype])
        #os.system('rm bacdive.csv')
        bacdive.rename(columns = {'16S associated NCBI tax ID':'taxid',phenotype:'phenotype1','ordo':'order'}, inplace = True)
        bacdive['taxid'] = bacdive['taxid'].astype('Int64')
        bacdive.dropna(subset=['phenotype1'], inplace = True)

        if phenotype in ['pH','Temperature','Salt conc.']:
            bacdive['phenotype2'] = bacdive.phenotype1
            for idx in bacdive.index:
                cell = bacdive.loc[idx].phenotype1
                if cell.startswith("-") == True and cell.count("-") > 1:
                    bacdive.at[idx,'phenotype1'] = '-'+re.findall(r"[+]?(?:\d*\.*\d+)", cell.split('-')[1])[0]
                    bacdive.at[idx,'phenotype2'] = re.findall(r"[+]?(?:\d*\.*\d+)", cell.split('-')[2])[0]
                elif cell.startswith("-") == False and '-' in str(cell):
                    bacdive.at[idx,'phenotype1'] = re.findall(r"[+]?(?:\d*\.*\d+)", cell.split('-')[0])[0]
                    bacdive.at[idx,'phenotype2'] = re.findall(r"[+]?(?:\d*\.*\d+)", cell.split('-')[1])[0]
                else:
                    bacdive.at[idx,'phenotype1'] = re.findall(r"[+]?(?:\d*\.*\d+)", cell)[0]
                    bacdive.at[idx,'phenotype2'] = re.findall(r"[+]?(?:\d*\.*\d+)", cell)[0]
        
            for idx in reversed( range( 0,len(bacdive.index) ) ):
                if type(bacdive.iloc[idx].species) == float:
                    if float(bacdive.loc[bacdive.index[idx-1]].phenotype1) > float(bacdive.iloc[idx].phenotype1):
                        bacdive.at[bacdive.index[idx-1],'phenotype1'] = bacdive.iloc[idx].phenotype1
                    if float(bacdive.loc[bacdive.index[idx-1]].phenotype2) < float(bacdive.iloc[idx].phenotype2):
                        bacdive.at[bacdive.index[idx-1],'phenotype2'] = bacdive.iloc[idx].phenotype2
                    bacdive.drop(index=bacdive.index[idx],inplace=True)
            
            for idx in bacdive.index:
                if bacdive.loc[idx].phenotype1 == bacdive.loc[idx].phenotype2:
                    value = float(bacdive.at[idx,'phenotype1'])
                    bacdive.at[idx,'phenotype1'] = value - 0.05*value
                    bacdive.at[idx,'phenotype2'] = value + 0.05*value

        if phenotype == 'Oxygen tolerance':
            # phenotype1 = aerobic
            bacdive['phenotype2'] = 'no' # phenotype2 = anaerobic
            for idx, row in bacdive.iterrows():
                if row.phenotype1 in ['aerobe','obligate aerobe']:
                    bacdive['phenotype1'][idx] = 'yes'
                    bacdive['phenotype2'][idx] = 'no'
                elif row.phenotype1 in ['anaerobe','obligate anaerobe']:
                    bacdive['phenotype1'][idx] = 'no'
                    bacdive['phenotype2'][idx] = 'yes'
                elif row.phenotype1 in ['aerotolerant','facultative aerobe','facultative anaerobe','microaerophile','microanaerobe','microaerotolerant']:
                    bacdive['phenotype1'][idx] = 'yes'
                    bacdive['phenotype2'][idx] = 'yes'
        
        bacdive['genus'] = bacdive['species'].str.split(' ', expand=True)[0]
        bacdive.to_csv( self.data_folder + phenotype + '/data/phenDB.csv', index=False)
        return bacdive
                    
    """
    Function: parse the file of bacteria phenotypes from the Madin et al. study
    Observations: data from Madin is manipulated to better adapt for multiclasses/multilabels
    Parameters: phenotype - phenotype of interest
                pathway - if phenotype of interest is 'pathway' this parameter is to specify which is the pathway of interest
    """

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
            for idx, row in madin.iterrows():
                if phenotype in row.phenotype1:
                    madin['phenotype1'][idx] = 'yes'
                else:
                    madin['phenotype1'][idx] = 'no'
        else:
            madin = madin.dropna(subset=[phenotype]) # drops lines with no information about the phenotype of interest
            madin.rename(columns = {phenotype:'phenotype1'}, inplace = True)

        if phenotype == 'range_salinity':            
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
                    
        madin.to_csv( self.data_folder + phenotype + '/data/phenDB.csv', index=False)
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
    def CreateGenomeDatabase(self, phenDB, ncbi, genomes_per_species):
        for idx_phenDB, row_phenDB in phenDB.iterrows():
            phenDB_ncbi = ncbi.loc[ncbi.taxid == str(row_phenDB.taxid)]
            phenDB_ncbi = phenDB_ncbi[:genomes_per_species]

            if len(phenDB_ncbi) > 0:
                folder = '../data/genomes/'+str(row_phenDB.taxid)+'/'
                os.makedirs(folder, exist_ok=True)

                for idx_phenDB_ncbi, row_phenDB_ncbi in phenDB_ncbi.iterrows():
                    file = row_phenDB_ncbi.ftp_path.split('/',9)[-1] + '_genomic.fna'
                    if os.path.isfile( folder + file ) == False:
                        file = file +'.gz'
                        self.DownloadData( row_phenDB_ncbi.ftp_path+'/'+file, folder, file)                        

    def __init__(self, phenotype, genomes_per_species = 1):
        
        self.data_folder = './results/'
        os.makedirs(self.data_folder + phenotype + '/data', exist_ok=True)
        
        phenDB = None
        if phenotype in ['pH',
                         'Temperature',
                         'Salt conc.',
                         'Oxygen tolerance']:
            phenDB = self.ParseBacDive( phenotype )
        elif phenotype in ['nitrogen_fixation',
                           'nitrate_reduction',
                           'fermentation',
                           'sulfate_reduction',
                           'range_salinity',
                           'range_tmp',
                           'metabolism',
                           'motility',
                           'sporulation']:
            phenDB = self.ParseMadin( phenotype )
        else:
            print('WARNING: There is no database for that phenotype.')
        
        ncbi = self.ParseNCBI()
        self.CreateGenomeDatabase( phenDB, ncbi, genomes_per_species )
        print('Downloaded genomes database.')