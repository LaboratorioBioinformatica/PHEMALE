import os
import pandas
from joblib import dump
import csv
import numpy
from glob import glob
numpy.seterr(divide = 'ignore')
from numba import jit

class DataStandardization:

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
    
    #Transform the eggnog file into a vector with quantities of each ortholog group present
    def TransformFile(self, eggnog, hypothetical_OG):

        eggnog = list( csv.reader(open(eggnog), delimiter='\t') )
        genome = [0]*len(self.OG_columns)
        
        for row in eggnog[1:]:
            OG = row[4].split(',')[0].split('@')[0]
            
            if (hypothetical_OG == False and 'COG' in OG) or hypothetical_OG == True:
                transformedEvalue = self.TransformEVALUE( numpy.float16(row[2]) )

                try:
                    index = self.OG_columns.index(OG)
                    genome[index] = genome[index] + transformedEvalue
                
                except ValueError:
                    self.OG_columns.append(OG)
                    genome.append(transformedEvalue)
        
        return genome
    
    def __init__(self, phenotype, hypothetical_OG ):

        madin = pandas.read_csv( './results/' + phenotype + '/data/madin.csv')
        
        if os.path.isfile( 'madin_'+phenotype+'_append.csv' ) == True:
            madin_append = pandas.read_csv( 'madin_'+phenotype+'_append.csv')
            madin = pandas.concat([madin, madin_append])
        
        self.OG_columns = []
        
        for idx_madin, row_madin in madin.iterrows():     
            
            folder = '../data/genomes/'+str(row_madin.taxid)+'/'

            if os.path.isdir( folder ) == True:
                for file in glob(folder+"*.eggnog"):
                    datum = self.TransformFile( file, hypothetical_OG )
                    dump( datum, file + '.' + phenotype + '.joblib' )

        dump( self.OG_columns, './results/' + phenotype + '/data/OGColumns.joblib')
        
        print('Standarized data.')