from numba import jit
from joblib import load, dump
import os

import pandas
pandas.options.display.max_colwidth = 2000  #aumenta limite de dados do pandas

import numpy
numpy.seterr(divide = 'ignore')

import warnings
warnings.filterwarnings('ignore')

class TransformData:

    def ParseEggNOGFile(self, file):
        
        with open(file) as annotations:
        
            annotations = [line.rstrip() for line in annotations]
            annotations = [component.split("\t") for component in annotations]
            del annotations[0:4]                                     # deletando as 4 linhas do início
            del annotations[ len(annotations)-3 : len(annotations) ] # deletando as 3 linhas do fim
            annotations = pandas.DataFrame(data=annotations[1:], columns=annotations[0])
        
        return annotations
    
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
        
        self.OG_columns = []
        self.ortholog_groups_DB = ortholog_groups_DB
        
        madin = pandas.read_csv('./results/'+phenotype+'/'+'madin_'+phenotype+'.csv')
        
        for idx in madin.index:
            
            folder = '../genomes/'+str(madin['taxid'][idx])+'/'
            
            if os.path.isdir(folder):
                
                for eggnog_file in os.listdir(folder):
                
                    if eggnog_file.endswith('.annotations'):
                    
                        annotation = self.ParseEggNOGFile( folder + eggnog_file )
                        genome = self.TransformEggNOGFile( annotation, self.ortholog_groups_DB )
                        dump( genome, folder + eggnog_file + phenotype + '.joblib' )

        dump( self.OG_columns, './results/' + phenotype + '/'+'OGColumns.joblib')
        
        print('Processed data for machine learning')