import os
from urllib.request import urlretrieve
import pandas
pandas.options.display.max_colwidth = 2000  #higher data threshold limit for pandas
from math import isnan
from warnings import filterwarnings
filterwarnings('ignore')

phenotype = 'pH'

if phenotype == 'pH':
    phenotype_file = 'https://bacdive.dsmz.de/advsearch/csv?fg%5B0%5D%5Bgc%5D=OR&fg%5B0%5D%5Bfl%5D%5B3%5D%5Bfd%5D=Domain&fg%5B0%5D%5Bfl%5D%5B3%5D%5Bfo%5D=exact&fg%5B0%5D%5Bfl%5D%5B3%5D%5Bfv%5D=Bacteria&fg%5B0%5D%5Bfl%5D%5B3%5D%5Bfvd%5D=strains-domain-1&fg%5B0%5D%5Bfl%5D%5B4%5D=AND&fg%5B0%5D%5Bfl%5D%5B5%5D%5Bfd%5D=16S+associated+NCBI+tax+ID&fg%5B0%5D%5Bfl%5D%5B5%5D%5Bfo%5D=NOT_equal&fg%5B0%5D%5Bfl%5D%5B5%5D%5Bfv%5D=0&fg%5B0%5D%5Bfl%5D%5B5%5D%5Bfvd%5D=sequence_16S-tax_id-7&fg%5B0%5D%5Bfl%5D%5B6%5D=AND&fg%5B0%5D%5Bfl%5D%5B7%5D%5Bfd%5D=pH&fg%5B0%5D%5Bfl%5D%5B7%5D%5Bfo%5D=greater&fg%5B0%5D%5Bfl%5D%5B7%5D%5Bfv%5D=0&fg%5B0%5D%5Bfl%5D%5B7%5D%5Bfvd%5D=culture_pH-pH-3'

urlretrieve(phenotype_file, 'bacdive.csv')
bacdive = pandas.read_csv('bacdive.csv', skiprows=[0, 1], usecols=['species','16S associated NCBI tax ID',phenotype])
os.system('rm bacdive.csv')
bacdive.rename(columns = {'16S associated NCBI tax ID':'taxid'}, inplace = True)

if phenotype == 'pH':

    # Transformando em lista os valores de fenótipos
    bacdive.phenotype1 = bacdive[phenotype].values.reshape(-1, 1).tolist()
    
    for idx in reversed( range( 0,len(bacdive.index) ) ):
        if isnan(bacdive.iloc[idx].taxid) == True:
            for i in bacdive.iloc[idx].phenotype1:
                bacdive.iloc[idx-1][phenotype] = bacdive.iloc[idx-1][phenotype].append(i)
            bacdive.drop(index=idx,inplace=True)
    
    print(bacdive)
    
    bacdive['phenotype1'] = ''
    for idx in range(0,len(bacdive.index)):
        phenotype1 = []
        for item in bacdive.iloc[idx][phenotype]:
            if '-' in str(item):
                items = item.split('-')
                for i in items:
                    phenotype1.append(i)
            else:
                phenotype1_temp.append(item)
        bacdive.iloc[idx].phenotype1 = phenotype1
        
        if idx in [0,1,2]:
            print(phenotype1_temp)
            print(bacdive.iloc[[idx]].phenotype1)
    
    print(bacdive)

"""
        #bacdive.phenotype2 = max(bacdive.phenotype1) # Higher value
        #bacdive.phenotype1 = min(bacdive.phenotype1) # Lower value
"""