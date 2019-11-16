'''

processing the special format used by the Cranfield Dataset



'''
from doc import Document,Collection


class CranFile:
    def __init__(self, filename):
        self.docs = []

        cf = open(filename)
        docid = ''
        title = ''
        author = ''
        body = ''
        for line in cf:
            if '.I' in line:
                if docid != '':
                    body = buf
                    self.docs.append(Document(docid, title, author, body))
                # start a new document
                docid = line.strip().split()[1]
                buf = ''
            elif '.T' in line:
                None
            elif '.A' in line:
                title = buf # got title
                buf = ''
            elif '.B' in line:
                author = buf # got author
                buf = ''
            elif '.W' in line:
                buf = '' # skip affiliation
            else:
                buf += line
        self.docs.append(Document(docid, title, author, body)) # the last one

if __name__ == '__main__':
    ''' testing '''

    cf = CranFile ('/Applications/SimpleSearchEngine/CranfieldDataset/cran.all')
    print(cf)
    
    for i ,doc in enumerate(cf.docs):
        if i <2:
            print (doc.docID,doc.body)
            #print(len(doc.body),len(doc.title),len(doc.docID),len(doc.author))
           # cf.docs.append(Collection)
    print ("total number of documents",len(cf.docs))
    
