'''
a program for evaluating the quality of search algorithms using the vector model

it runs over all queries in query.text and get the top 5 results,
and then qrels.text is used to compute the NDCG metric

usage:
    python batch_eval.py index_file query.text qrels.text

    output is the average NDCG over all the queries

'''
from query import *
from metrics import *
from scipy import stats
import json
import pandas as pd
def eval():
        ''' eval() function reads two json files of booleanresult and vectorresults ,transform the values to dataframes for assiging
             binary relevance for actual and true ,calculate NDCG@5,Average NDCG and perform ttest to get p_value'''
        # ToDo  
        print("NDCG@10 calculation for Vector and Boolean results is in Progress")    
        '''Reading vector from file'''         
        Vectorfile = open("vectoresult")
        vectorlist = json.load(Vectorfile)
        '''iteration over vectorfile and qrel file and storing the data in dataframes for further assigning relevance '''
        vectorNDCGDic ={}
        RVector ={}
        IRVector ={}
        '''processing vector list (docid,Similarity) w.r.t cosine value under relevance and Irrelevance docid '''
        for vector in vectorlist:
            #print(vector)
            for vectorID in vector.keys():
                #print(vectorID)#queryId
                for cosinevalue in vector[vectorID]:
                         #print(cosinevalue)
                         if cosinevalue[1] < 1.0:
                             if vectorID not in IRVector:
                                 IRVector[vectorID]=[cosinevalue[0]]
                                 
                             else:
                                 IRVector[vectorID].append(cosinevalue[0])
                         else:
                             if vectorID not in RVector:
                                 RVector[vectorID]=[cosinevalue[0]]
                                 
                             else:
                                 RVector[vectorID].append(cosinevalue[0])
        'storing ideal values ,vector relevant and irrelevant values in dictionary and converting it into matrix form'   
        for vector in vectorlist:
            for vectorID in vector.keys():
                for QrelID in qrelDic.keys():
                 if qidlist[vectorID]==QrelID:
                    vectorNDCGDic[vectorID]=[{"IRV":sorted(IRVector[vectorID])},{"RV":sorted(RVector[vectorID])},{"I":sorted(qrelDic[qidlist[vectorID]])}]
        VectorQrelmatrix = pd.DataFrame(vectorNDCGDic)
        'Iterating over each query vectorQrelmatrix and assigning  relevance cosine value to the document Y_Scoreor vector relevance'
        #print("r",RVector)
        vectorAvgNDCG={}
        for vector in vectorlist:
            for vectorID in vector.keys():
                for QrelID in qrelDic.keys():
                 if qidlist[vectorID]==QrelID:
                    IRV=VectorQrelmatrix.loc[0][vectorID]
                    RV=VectorQrelmatrix.loc[1][vectorID]
                    I=VectorQrelmatrix.loc[2][vectorID]
                    vectorNDCGmatrix ={}
                    for IRVvalues in IRV.values():
                        for RVvalues in RV.values():
                            for Idealvalues in I.values():
                                match = set(RVvalues).intersection(set(Idealvalues))#(1,1)(cosinevalue,1),(1.0,1)
                                IdealOnly=set(Idealvalues)-match #(0,1)(cosinevalue,1)
                                #print(IdealOnly)
                                VectorRelonly=set(RVvalues)-match #(1,0)(cosinevalue,0)(1.0,0)
                                VectorIRonly=set(IRVvalues)-set(Idealvalues) #(0,0)(cosinevalue,0)
                                
                                for v in sorted(set(match),key=int):
                                    ''' 1strow document id,second row Vector  relevance,third row True relevance'''
                                    for cosinevalues in vector[vectorID]:#cosinevalues[0] is docid and cosinevalue[1] is cosine similairy
                                            if cosinevalues[0]==int(v):
                                                    #print("v",v,cosinevalues)
                                                    vectorNDCGmatrix[v]=[cosinevalues[1],1]
                                #print(vectorNDCGmatrix)
                                for v1 in sorted(IdealOnly,key=int):
                                    for cosinevalues in vector[vectorID]:#cosinevalues[0] is docid and cosinevalue[1] is cosine similairy
                                            if cosinevalues[0]==int(v1):
                                                    #print("v1",v1,cosinevalues)
                                                    vectorNDCGmatrix[v1]=[cosinevalues[1],1]
                                    #print("v1",v1)
                                #print(vectorNDCGmatrix)
                                for v2 in sorted(VectorRelonly,key=int):
                                    for cosinevalues in vector[vectorID]:#cosinevalues[0] is docid and cosinevalue[1] is cosine similairy
                                            if cosinevalues[0]==int(v2):
                                                    #print("v2",v2,cosinevalues)
                                                    vectorNDCGmatrix[v2]=[cosinevalues[1],0]
                                #print(vectorNDCGmatrix)
                                for v3 in sorted(VectorIRonly,key=int):
                                    for cosinevalues in vector[vectorID]:#cosinevalues[0] is docid and cosinevalue[1] is cosine similairy
                                            if cosinevalues[0]==int(v3):
                                                    #print("v3",v3,cosinevalues)
                                                    vectorNDCGmatrix[v3]=[cosinevalues[1],0]
                                #print(vectorNDCGmatrix)
                    'using Dataframes transforming binary values into matrix for given query'                
                    VectorNDCGmatrixDF = pd.DataFrame(vectorNDCGmatrix)
                    #print("VectorNDCGmatrixDF",VectorNDCGmatrixDF)
                    'passing matrix values as array to ndcg_score function from metrics.py'
                    Y_score=np.array(VectorNDCGmatrixDF.loc[0]) #y score
                    Ytrue=np.array(VectorNDCGmatrixDF.loc[1]) #true
                    vresult = ndcg_score(Ytrue,Y_score)
                    vectorAvgNDCG[vectorID]=vresult
        'calculating Average ndcg for queries in vector model'
        vndcg =0
        for vndcg in vectorAvgNDCG.values():
            vndcg=vndcg+1
        #print("vector NDCG for queries",vectorAvgNDCG) 
        
        
        '************Code for Boolean result NDCG and Average NDCG calculation***********************'
        ''' reading boolean results from Boolean file'''      
        BoolenResultfile = open("Boolean")
        Boollist = json.load(BoolenResultfile)
        '''iterating qrel dic and boolenresult and storing values in NDCG dictionary'''
        PreNCDGDic = {}
        for Blistvalues in Boollist:
            for BoolID in Blistvalues.keys():
                for QrelID in qrelDic.keys():  
                 if qidlist[BoolID]==QrelID:
                     PreNCDGDic[BoolID]=[{"B":sorted(Blistvalues[BoolID],key =int)},{"I":sorted(qrelDic[qidlist[BoolID]])}]
        'Converting dictionary to matrix'
        BoolQrelmatrix = pd.DataFrame(PreNCDGDic)
        'Iterating over Boolean result and assigning relevance 0 or 1 to the queryid'
        AvergaeNDCG={}
        for Blistvalues in Boollist:
            for BoolID in Blistvalues.keys():
                for QrelID in qrelDic.keys():
                 if qidlist[BoolID]==QrelID: 
                    B=BoolQrelmatrix.loc[0][BoolID]
                    I=BoolQrelmatrix.loc[1][BoolID]
                    NDCGmatrix ={}
                    for values in B.values():
                        for v in I.values():
                            match = set(values).intersection(set(v))
                            IdealOnly=set(v)-match
                            Boolenonly=set(values)-match
                            for v in sorted(set(match),key=int):
                                ''' 1strow document id,second row Boolean relevance,third row True relevance'''
                                NDCGmatrix[v]=[1,1]
                            for v1 in sorted(IdealOnly,key=int):
                                NDCGmatrix[v1]=[0,1] 
                            for v2 in sorted(Boolenonly,key=int):
                                NDCGmatrix[v2]=[1,0]  
                    'storing relevance into matrix form '
                    NDCGmatrixDF = pd.DataFrame(NDCGmatrix)
                    Y_score=np.array(NDCGmatrixDF.loc[0]) #y score
                    Ytrue=np.array(NDCGmatrixDF.loc[1]) #true
                    'calculating ndcg_score for each given query using ndcg_score function'     
                    result = ndcg_score(Ytrue,Y_score)
                    AvergaeNDCG[BoolID]=result
        'Calculating Average NDCG for boolean algorithm results'
        ndcg =0
        for ndcg in AvergaeNDCG.values():
            ndcg=ndcg+1
        #print("NDCG for Boolean processing",AvergaeNDCG)     
        print("Average NDCG for Boolean",ndcg/len(AvergaeNDCG))
        print("Average NDCG for Vector",vndcg/len(vectorAvgNDCG))  
        '''*********Below code for passing calculated NDCG values for each given query to test function and return p-value********'''  
        ttest_Y=[]#Vector NDCG values
        ttest_X=[]#Boolean NDCG values
        
        for vector in vectorAvgNDCG.values():
           ttest_Y.append(vector) 
        for bool in AvergaeNDCG.values():
           ttest_X.append(bool) 
        p_value = stats.wilcoxon(ttest_X,ttest_Y)
        'ttest function passed with boolean NDCG and Vector NDCG values "scipy.stats.ttest_ind"'
        v_value = stats.ttest_ind(ttest_X,ttest_Y)
        print(p_value)
        print(v_value)
if __name__ == '__main__':
        
    '''variables of Qrelfile'''
    qrelDic ={}
    qrelsfile =sys.argv[3]
    '''opens qrels.text file and reads all qid,docid relevances and stores in a Dictionary qrelDic'''
    with open(qrelsfile, "r") as relevancefile:
            QDlines = relevancefile.readlines()
            for QD in QDlines:
                for Q in QD.split()[0:1]:
                    for D in QD.split()[1:2]:
                        if int(Q) not in qrelDic:
                            qrelDic[int(Q)]=[D]
                        else:
                            qrelDic[int(Q)].append(D)
    '''calling batch_eval from query.py to get query results for vector and boolean model for NDCG,Average NDCG'''
    Boolenresult,vectorresult=batch_eval(int(sys.argv[4]))  
    #print("topk boolean",Boolenresult)
    'storing Boolean and Vector result returned by executing batch_eval function at Query.py'
    indexobj.save(Boolenresult,"Boolean")
    indexobj.save(vectorresult,"vectoresult")
    #print("topk vector",vectorresult)
    '''calling eval() function to assign binary values to actual and true relevance 
         calculating NDCG@20,Average NDCG for boolean and vector results and do calculate p-value using t-test'''
    eval()

        
    
