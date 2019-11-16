
'''
query processing

'''
from doc import Collection
from cran import CranFile
from index import Posting,InvertedIndex
from cranqry import *

from util import isStopWord,stemming
from nltk.tokenize import RegexpTokenizer
import norvig_spell
import json
import numpy as np
import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial.distance import cosine
import math
from _operator import itemgetter
import random
import sys
class QueryProcessor:

    def __init__(self, query, index, collection):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index =index
        self.docs = collection

    def preprocessing(self):
        ''' apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector'''

        #ToDo: return a list of terms
        'tokenization of query along with removal of punctuation'
        tokenizer = RegexpTokenizer(r'\w+') 
        querytoken = tokenizer.tokenize(self.raw_query)
        '''checking for spell error in query tokens and making corrections and 
        storing the words in Query dictionary'''
        for token in querytoken:
            to_lower = ''.join(norvig_spell.words(token)) #converting list to string
            spellcorrection = norvig_spell.correction(to_lower)
            Query.append(spellcorrection)
            stopword = isStopWord(spellcorrection)
            if not stopword:
                stemqueryterm = stemming(spellcorrection)
                Queryterm.append(stemqueryterm)

    def booleanQuery(self):
        ''' boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C" 
        for retrieving posting lists and merge them'''
        #ToDo: return a list of docIDs
        PostingDict = {} #store key value pair of query term and postings by processing index file
        boolen = [] #stores list of docid for each queryterm key
        booleanResult = set()
        tempDic={}
        QueryDic={}
        
        for qterm in Queryterm:
            plist = InvertedIndex.getPostingsList(qterm)
            '''since every term in inverted index is unique below code adds the qterm:postings list 
                to Postings Dictionary'''        
            PostingDict.update({qterm:plist})
        for qterms in PostingDict.keys():
            tempDic[qterms]=len(PostingDict[qterms])
        for qterms,cf in tempDic.items():
            if cf >0:
             if cf<300:
                QueryDic[qterms]=cf
        '''checking for length of query term is it contains only single word it directly posts 
                the result read from inverted index file'''
        if len(QueryDic) == 1:
            for key in QueryDic.keys():
                booleanResult = PostingDict[key]
                if not booleanResult:
                    print("Given query has no matched Document" ,''.join(Query))
                else:
                    print("Result of the search query ",booleanResult)
        else:
            keylist = list(QueryDic.keys())
            '''iterating over query terms as keys and merging postings list over intersection 
                to find list of postings that contains all query terms'''
            for key in QueryDic.keys():
                   'adding postings list of each queryterm'
                   boolen.append(sorted(PostingDict[key],key=int)) 
            '''checking the intersection result boolean result set '''
            booleanResult = set.intersection(*map(set, boolen))
        'If first boolean result is null then we process pairwise intersection of query terms'
        if booleanResult == set():
            for i in range(len(QueryDic)-1):
                if not i==len(QueryDic)-1:
                    p1 = PostingDict[keylist[i]]
                    p2 = PostingDict[keylist[i+1]]
                    temp =InvertedIndex.mergeList(p1,p2)
                    '''checking for empty result post merge if result is not empty set adding 
                    the intersection result boolean result set '''
                    if not temp == set():
                            booleanResult.update(temp)
        return sorted(booleanResult,key=int)
        
        

    def vectorQuery(self, k):
        ''' vector query processing, using the cosine similarity. '''
        #ToDo: return top k pairs of (docID, similarity), ranked by their cosine similarity with the query in the descending order
        # You can use term frequency or TFIDF to construct the vectors 
        'Finding TF and IDF of Queryterms and saving the result to TF.json and IDF.json file'
        termfrequency,IDF =postingobj.term_freq(collectionfile,Queryterm)
        'Saving TF,IDF of document for given query'
        indexobj.save(termfrequency,"TF.json")
        indexobj.save(IDF,"IDF.json")
        TF_filename = open("TF.json")
        TF = json.load(TF_filename)
        IDF_filename = open("IDF.json")
        IDF=json.load(IDF_filename)  
        QueryDict = {}
        Qlen = len(Query)
        Querytf ={}
        Querytfidf ={}
        tempdic ={}
        DocSim = []
        '''processing each query term and calculating TF-IDF of query and passing document 
            and query vector to cosine function to calculate cosine similarity'''
        for term in Queryterm:
            plist = InvertedIndex.getPostingsList(term)
            QueryDict.update({term:plist})
            if term not in Querytf.keys():
                Querytf[term]= 1
            else :Querytf[term]=Querytf[term]+1
        for qterms,posting in QueryDict.items():
            for pos in posting:
                for IDFword in IDF:   
                    if qterms == IDFword:
                        if qterms not in Querytfidf.keys():
                            '''calculating tf of query using query token frequency in query to the total query tokens'''
                            tf=Querytf[qterms]
                            '''calculating td-idf of query where idf of word in query is 1+log(N/n) 
                                where N total documents and n is number of documents that contain the term '''
                            Querytfidf[qterms]={pos:tf*(1+IDF[IDFword])}
                        else:Querytfidf[qterms].update({pos:(tf)*(1+IDF[IDFword])})
                        TFwordValues = TF[qterms]
                        '''calculating TF*IDF of document and converting it to vector''' 
                        for TFdoc,TFvalues in TFwordValues.items():
                            for IDFword in IDF:
                                if qterms == IDFword and TFdoc == pos:
                                    if qterms not in tempdic.keys():
                                        tempdic[qterms]={TFdoc:(TFvalues)*IDF[IDFword]}
                                    else:tempdic[qterms].update({TFdoc:TFvalues*IDF[IDFword]})         
        
        'converting Query tf -idf dictionary to matrix/vector'
        Querymatrix = pd.DataFrame(Querytfidf)
        'converting document tf-idf dictionary to matrix/vector'
        DocTFIDFmatrix = pd.DataFrame(data=tempdic)
        'processing the matrix/vector to make feasible for cosine function '
        for Qpos , Dpos in zip(list(Querymatrix.index) , list(DocTFIDFmatrix.index)):
            if Qpos==Dpos:
                Q = np.array(Querymatrix.loc[Qpos])
                where_are_NaNs = np.isnan(Q)
                Q[where_are_NaNs] = 0
                D= np.array(DocTFIDFmatrix.loc[Dpos])
                where_are_NaNs = np.isnan(D)
                D[where_are_NaNs] = 0
                cosine = QueryProcessor.cosine_similaritys(Q,D)    
                DocSim.append((int(Qpos),cosine))
        VectorID = sorted(DocSim,key=lambda x:x[1],reverse = True)
        TopID=sorted(DocSim[:10],key=lambda x:x[1],reverse = True)
        #print(VectorID)
        VectorResult.append({qid:VectorID})
        return TopID,k
    'calculating cosine score of query and document'    
    def norm(vector):
            return math.sqrt(sum(x * x for x in vector))    

    def cosine_similaritys(vec_a, vec_b):
        norm_a = QueryProcessor.norm(vec_a)
        #print(norm_a)
        norm_b = QueryProcessor.norm(vec_b)
        #print(norm_b)
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        #print(dot,(norm_a * norm_b))
        return dot / (norm_a * norm_b)        

def test():
    ''' test your code thoroughly. put the testing cases here'''
    qid='001'
    qprocessorobj = QueryProcessor(qrys[qid].text,InvertedIndex.items,collectionfile.docs)
    query()
    print ('Pass')

def query():
    ''' the main query processing program, using QueryProcessor'''

    # ToDo: the commandline usage: "echo query_string | python query.py index_file processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents
    'from query function calling pre processing function and retriving results according to given Processing Algorithm '
    qprocessorobj.preprocessing()
    #print(Queryterm)
    'code for calling given processing Algorithm'
    if ProcessingAlgorithm == '0':
        print("Executing Boolean Processing Algorithm")
        Bresult = qprocessorobj.booleanQuery()
        print("Total number of retrieved document for search is",len(Bresult))
        print(Bresult)
        BoolenQueryResultDic.append({qid:Bresult})
    else: 
       print("Vector Query TF-IDF calculation in progress")   
       Topk,k= qprocessorobj.vectorQuery(3)
       #print("vector",qid,qrys[qid].text)
       print("Top", k ,"(DocID Similarity)",Topk[:k])
''' ************this below code is reused in batch_eval also*******************'''       
input_filename = "cran.all"
ouput_filename = sys.argv[1]#"index_file" #sys.argv[2]
Queryfile = "query.text"#sys.argv[3]#"query.text"
'''creating object for cranefile and collection file and inverted index class,postings class'''
cf=CranFile(input_filename)
collectionfile = Collection()
indexobj = InvertedIndex()
'iterating over cran file for document id'
for i,doc in enumerate(cf.docs):
    collectionfile.docs.update({doc.docID:doc})
postingobj= Posting(doc.docID)
'''reading index file which is stored while creating index'''
with open(ouput_filename,"r") as invertedindex:
    InvertedIndex.items=json.load(invertedindex)
'formatting the query id in qrel.text and finding common query id in qrery.text'
qidlist ={}
qrys =loadCranQry(Queryfile)
for position,q in enumerate(qrys):
    qidlist[q]=position+1
'Below Variables are used for batch_eval.py file'
BoolenQueryResultDic =[]
VectorResult=[]

def batch_eval(number):
        '''This method returns Boolean result and vector result to "batch_eval.py" file 
                                                        for obtaining results for "N" random Queries'''
        randomquerylist = set()
        global qid
        global Queryterm
        global Query
        qid=random.sample(list(qidlist.keys()),number)
        randomquerylist=qid
        #print(len(randomquerylist),number)
        print("Randomly selected",number,"query Id's are")
        print(randomquerylist)
        print("Query processing for random queries is in Progress")
        for qid in randomquerylist:
            Queryterm = []
            Query =[] #actual query terms
            global qprocessorobj
            qprocessorobj = QueryProcessor(qrys[qid].text,InvertedIndex.items,collectionfile.docs) 
            'processing query for iterative query retrival'
            qprocessorobj.preprocessing()
            Bresult = qprocessorobj.booleanQuery()
            BoolenQueryResultDic.append({qid:Bresult})   
            Topk,k= qprocessorobj.vectorQuery(10)
        return BoolenQueryResultDic,VectorResult
if __name__ == '__main__':
    #test() 
    Queryterm = []
    Query =[] #actual query terms
    ProcessingAlgorithm = sys.argv[2] 
    'retriving result for query id which is randomly selected from common qid list'
    qid=random.choice(list(qidlist.keys()))
    'displaying randomly selected query with its ID'
    print("Randomly selected query")
    print(qid,qrys[qid].text)
    #print("Query position",qidlist[qid])
    qprocessorobj = QueryProcessor(qrys[qid].text,InvertedIndex.items,collectionfile.docs)
    query()
       
    
