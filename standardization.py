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
        threshold = numpy.float16(10**(-6))
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
    def TransformFile(self, eggnog, hypothetical):

        eggnog = list( csv.reader(open(eggnog), delimiter='\t') )
        genome = [0]*len(self.OG_columns)
        
        for row in eggnog[1:]:
            OG = row[4].split(',')[0].split('@')[0]
            
            if (hypothetical == False and 'COG' in OG) or hypothetical == True:
                transformedEvalue = self.TransformEVALUE( numpy.float16(row[2]) )

                try:
                    index = self.OG_columns.index(OG)
                    genome[index] = genome[index] + transformedEvalue
                
                except ValueError:
                    self.OG_columns.append(OG)
                    genome.append(transformedEvalue)
        
        return genome
    
    def __init__(self, phenotype, hypothetical, io ):
        phenDB = pandas.read_csv( './results/' + phenotype + '/data/phenDB.csv')
        self.OG_columns = []
        self.io = io
        
        for idx_phenDB, row_phenDB in phenDB.iterrows():     
            
            folder = '../data/genomes/'+str(row_phenDB.taxid)+'/'

            if os.path.isdir( folder ) == True:
                for file in glob(folder+"*.eggnog"):
                    datum = self.TransformFile( file, hypothetical )
                    if hypothetical == True:
                        dump( datum, file + '.' + phenotype + '_hypothetical.joblib' )
                    else:
                        dump( datum, file + '.' + phenotype + '.joblib' )

        if hypothetical == True:
            self.io.Save( self.OG_columns, 'OGColumns_hypothetical')
        else:
            self.io.Save( self.OG_columns, 'OGColumns')
        
        print('Standarized data.')