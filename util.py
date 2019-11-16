
'''
   utility functions for processing terms

    shared by both indexing and query processing
'''
#import nltk

#nltk.download()
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer,PorterStemmer
from random import random
from nltk.tokenize import RegexpTokenizer



def isStopWord(word):
    ''' using the NLTK functions, return true/false'''

    # ToDo
    'using NLTK functions to find stopwords and remove them'
    stoplist = set(stopwords.words('english')) 
    if word in stoplist :
            
            return True
    else:
            return False

def stemming(word):
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''
    # ToDo 
    #stemword = SnowballStemmer("english").stem(word)
    stemword = PorterStemmer().stem(word)
    return stemword
