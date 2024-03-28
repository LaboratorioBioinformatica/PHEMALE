import os
from numba import jit
from glob import glob
import pandas
pandas.options.display.max_colwidth = 2000  #higher data threshold limit for pandas
from warnings import filterwarnings
filterwarnings('ignore')

"""
Class: Eggnog-mapper step. Mapping Ortholog groups present in every genome
"""
class EggnogMapping:

    def ParseEggNOGFile(self, file):
        
        with open(file) as eggnog:
            try:
                eggnog = [line.rstrip() for line in eggnog]
                eggnog = [component.split("\t") for component in eggnog]
                del eggnog[0:4] # deleting 4 initial useless lines
                del eggnog[ len(eggnog)-3 : len(eggnog) ] # deleting e ending useless lines
                eggnog = pandas.DataFrame(data=eggnog[1:], columns=eggnog[0])
                return eggnog
            
            except UnicodeDecodeError:
                self.problems = True
                os.system('rm ' + file)
            
            except IndexError:
                self.problems = True
                os.system('rm ' + file)
        
        return ['Problems detected']
    
    def RunEggNOG(self, file):
        
        n_cpus = str(len(os.sched_getaffinity(0)))
        os.makedirs('./eggnog-mapper/temp', exist_ok=True)
        
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

        os.system('./eggnog-mapper/emapper.py -i '+ file + ' --itype genome --genepred prodigal -o ' + file + 
                  ' --cpu ' + n_cpus + ' --tax_scope 2 --tax_scope_mode broadest --override --no_file_comments '+
                  '--temp_dir ./eggnog-mapper/temp --dmnd_db ../data/databases/bacteria.dmnd --data_dir ../data/databases '+
                  '--dmnd_iterate yes --index_chunks 1 --block_size 0.4 --sensmode very-sensitive --target_orthologs one2one '+
                  '--outfmt_short')
        
        os.system('mv ' + file + '.emapper.annotations ' + file + '.eggnog')
        os.system('rm ' + file + '.emapper.*')
        self.ParseEggNOGFile( file + '.eggnog')
    
    """
    Parameters: madin - 
                ncbi - file with information of bacterias genomes in NCBI server
                number_genomes_per_species - maximum number of genomes collected per species
    """
    @jit
    def RunEggNOGOnDatabase(self, madin):
        
        for idx_madin, row_madin in madin.iterrows():     
            
            folder = '../data/genomes/'+str(row_madin.taxid)+'/'

            if os.path.isdir( folder ) == True:
                
                for file in glob(folder+"*.fna"):
                    
                    if os.path.isfile( file + '.eggnog' ) == False:
                        
                        if os.path.isfile( file + '.emapper.annotations' ) == True:
                            os.system('mv ' + file + '.emapper.annotations ' + file + '.eggnog')
                        else:
                            self.RunEggNOG(file)
                            
            print('Mapped: ' + str(idx_madin) + ' of ' + str( len(madin) ) )

    def __init__(self, phenotype ):
        
        madin = pandas.read_csv( './results/' + phenotype + '/data/madin.csv')
        
        if os.path.isfile( 'madin_'+phenotype+'_append.csv' ) == True:
            madin_append = pandas.read_csv( 'madin_'+phenotype+'_append.csv')
            madin = pandas.concat([madin, madin_append])
        
        self.RunEggNOGOnDatabase( madin )
        
        print('Successfully ran eggnog-mapper over database.')