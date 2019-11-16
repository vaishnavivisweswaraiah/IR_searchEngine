

'''

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

'''
from util import isStopWord,stemming
import doc
import json
from cran import CranFile
import os
from nltk.tokenize import RegexpTokenizer
import sys
from doc import Collection
import math


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)
    def term_freq(self,collectionfile,Queryterm):
        ''' return the term frequency in the document'''
        #ToDo
        termFrequency ={}
        IDF ={}
        x = InvertedIndex()
        tokenizer = RegexpTokenizer(r'\w+') 
        with open("index_file") as input_file:
            invertedindexfile = json.load(input_file)
            for qterm in Queryterm:
                for term in invertedindexfile:
                    if term==qterm:
                         postingslist=invertedindexfile[term]
                         for posting in postingslist:
                             for docid,termfrequency in posting.items():
                                '''calculating tf(t,d) = 
                                frequency of term in docuemnt / total number of tokens in document'''
                                weight = len(termfrequency)/float(len(tokenizer.tokenize(collectionfile.find(docid).body)))
                                if term not in termFrequency.keys():
                                    termFrequency[term]={docid:weight}
                                else:
                                    termFrequency[term].update({docid:weight})
                         'calling IDF function to calculate IDF of term'
                         val = x.idf(collectionfile,term)
                         IDF[term]=val
        return termFrequency,IDF

class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        self.sorted_postings= [] # may sort them by docID for easier query processing

    def add(self, docid, pos):
        ''' add a posting'''
        'changes has been made in below code as has_key been deprecated in python 3'
       # if not self.posting.has_key(docid): has_key removed in python 3
        if docid not in self.posting:
           self.posting[docid] = [pos]
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging'''
        # ToDo 
        for docterms in Doclist:
            for kdocterm,vdocterm in docterms.items():
                for Kvdocterm,Vvdocterm in vdocterm.items():
                    Vvdocterm.sort()

class InvertedIndex:
    

    def __init__(self):
        self.items = {} # list of IndexItems
        self.nDocs = 0  # the number of indexed documents


    def indexDoc(self, Pdoc): # indexing a Document object
            ''' indexing a document, using the simple SPIMI algorithm, but no need to store blocks 
            due to the small collection we are handling. Using save/load the whole index instead'''
            # ToDo: indexing only title and body; use some functions defined in util.py
            # (1) convert to lower cases,
            Predictionary ={}
            'Tokenizing the document'
            tokenizer = RegexpTokenizer(r'\w+') 
            tokens = tokenizer.tokenize(Pdoc.body)
            '''iterating over the tokens and converting it to lowercase'''
            for tokenpos,token in enumerate(tokens):
                IndexItemobj = IndexItem(token.lower())
                'checking if token is in the Predictionary'
                if (IndexItemobj.term not in Predictionary):
                    # (2) remove stopwords,
                    'checking the token is stop word or not.If it is stop word it will not be appended to Predictionary'
                    isStop = isStopWord(IndexItemobj.term)
                    if isStop == False:
                        'storing the token position of the document along with docid for the token in stem dictionary'
                        IndexItemobj.posting[int(Pdoc.docID)]=[tokenpos]
                        Predictionary[IndexItemobj.term]=IndexItemobj.posting
                        '''This below code is executed if predictionary already contains the token
                           just appened the value rather then replacing the term with new value'''
                else:
                    docIDlist = Predictionary[IndexItemobj.term]
                    if int(Pdoc.docID) not in docIDlist:
                        docIDlist[int(Pdoc.docID)]=[tokenpos]
                    else:
                        docIDlist[int(Pdoc.docID)].append(tokenpos)          
            'stemming the tokens in predictionary ' 
            'stem dictionary merging common terms and postion of token in a document while stemming'   
        # (3) stemming
            Stemdictionary = {}
            for keytoken,values in Predictionary.items():
                stem = stemming(keytoken)
                if stem not in Stemdictionary:
                        Stemdictionary[stem]=values
                else:
                        stemlist = Stemdictionary[stem]
                        for k in values.keys():
                            valuekey = k;
                        valueposition = values[valuekey]
                        for v in valueposition:
                            stemlist[valuekey].append(v)
                Doclist.append(Stemdictionary)
                IndexItemobj2 = IndexItem(keytoken)
                'sorting the token positions in stemDictionary'
                IndexItemobj2.sort()
            'single pass in memory indexing'
            '''Below code builds the inverted index using SPIMI-INVERT '''       
            for termdata in Doclist:
                for token ,posting in termdata.items():
                     for id , termpos in posting.items():
                        if token not in dictionary:
                            'add postings to dictionary if dictionary does not contain the token'
                            dictionary[token] = [posting]# add to dictionary
                        else:
                            'get Postings list if term existing in dictionary and append it to existing term'
                            Getpostinglist = dictionary[token]
                            if posting not in Getpostinglist:
                                dictionary[token].append(posting) #add to postings list
                                
    def sort(self):
        ''' sort all posting lists by docID'''
        #ToDo
        for terms,postings in dictionary.items():  
            print(terms,postings)
            sorteddiclist = sorted(postings, key=lambda dic:dic, reverse=True)
            return sorteddiclist                        
        
    def find(self, term):
        return self.items[term]

    def save(self, obj,filename):
        ''' save to disk :reusable function to save file to disk using JSON'''
        # ToDo: using your preferred method to serialize/deserialize the index
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename,'w') as output_file:
            json.dump(obj,output_file,indent=3)
            output_file.close()

    def load(self, filename):
        ''' load from disk'''
        # ToDo
        'reusable function to Load saved JSON file from disk'
        o_filename = open(filename)
        input = json.load(o_filename)
        for x in input:
            print(x,':', input[x])

    def idf(self, collectionfile,term):
        ''' compute the inverted document frequency for a given term using formula log(N/df) 
                                                                                where N is total number of documents in collection'''
        #ToDo: return the IDF of the term
        N = len(collectionfile.docs) #total number of documents
        '''sys.argv[1] is index file name given from command prompt'''
        with open("index_file") as input_file:
            invertedindexfile = json.load(input_file)
            dfrequency=invertedindexfile[term] #df number of documents containing term 
            inverseDF =math.log10(N/len(dfrequency))
            'returns IDF to the calling function'
        return inverseDF
                
    # more methods if needed
    def getPostingsList(queryterm):
        '''function to get postings list of query terms in index file'''
        postingList = []
        if queryterm in InvertedIndex.items:
            postingposList = InvertedIndex.items[queryterm]
            for postingpos in postingposList:
                for posting in postingpos.keys():
                    postingList.append(posting)
        return postingList
    
    def mergeList(result1,result2):
        'merge two sets of values and return merged sorted set as output .This function is mainly used in Boolean Retrival method'
        mergepostings = set(result1).intersection(result2)
        sorted(mergepostings,key=int)
        return mergepostings

def test():
    ''' test your code thoroughly. put the testing cases here'''
    '''test code to test wheather the NLTK process the document/sentence and return tokens without punctuation'''
    tokenizer = RegexpTokenizer(r'\w+')    
    string = tokenizer.tokenize("this is me checking , . (tokenization ' "")/\ ")
    print(string)
    i=['you','i','me','my','myself','bad','good']
    'checking for if function returns true for real stop words'
    for i in i:
        value = isStopWord(i)
        if( value == True):
            print("stopword",i)
        else:
            print("Not stopword",i)
    stem =['stemming','cars','experimental','coming']
    rootwords=[]
    for s in stem:
        stemword=stemming(s)
        rootwords.append(stemword)
    print("words post stemming")
    print(rootwords)
    print ('Pass')

def indexingCranfield():
    #ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file"
    # the index is saved to index_file
    input_filename = sys.argv[1]#"cran.all" #sys.argv[1]
    ouput_filename = sys.argv[2] #"Index.json" as INDEX_file #sys.argv[2]  
    'creating Cranfile and inverted index objects'
    cf=CranFile(input_filename)
    x = InvertedIndex()
    'Iterating over crancollection to process and create index for all documents in collection'
    for i,doc in enumerate(cf.docs):
        if i<1:
            'call to build index'
            x.indexDoc(doc)
            collectionfile.docs.update({doc.docID:doc})
    'Saving index to file'
    x.save(dictionary, ouput_filename)
    print("index created")
    #x.load(ouput_filename)
    #print ('Done')

if __name__ == '__main__':
    #test()
    
    collectionfile = Collection()
    Doclist=[]
    dictionary ={}
    indexingCranfield()
    
    
