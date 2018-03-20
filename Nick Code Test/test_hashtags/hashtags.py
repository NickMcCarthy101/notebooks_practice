# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 00:13:27 2017

@author: Skrepetos
"""
import sys
import os
import csv
import nltk as nk
# nk.download()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Sentence:
    senCount = 0
    def __init__(self, num, text, document_num):
        self.num = num
        self.text = text
        self.document = document_num
        Sentence.senCount += 1
        
class Term:
    def __init__(self, name, document_num, sentence_text):
        termCount = 1
        documentList = []
        sentencesList = []
        self.name = name
        self.occur_count = termCount
        self.documentList = documentList
        self.documentList.append(document_num)
        self.sentencesList = sentencesList
        self.sentencesList.append(sentence_text)
        
    def increaseOccurrence(self):
        self.occur_count += 1
    
    def addDocument(self, document_num):
        if not self.documentList:
            self.documentList.append(document_num)
        else:
            if document_num not in self.documentList:
                self.documentList.append(document_num)
        
    def addSentence(self, sentence_text):
        self.sentencesList.append(sentence_text)
        
def occurrences_table(ps, stop_words, document_num, lines):
    terms = []
    document = []
    for line in lines:
        count = 0
        tokens = []
        if line:
            count += 1
            sen_obj = Sentence(count, line, document_num)
            document.append(sen_obj)
            tokens_raw = word_tokenize(line)
            tokens_raw = [token.lower() for token in tokens_raw if token.isalpha()]
            for token in tokens_raw:
                if token not in stop_words:
                    tokens.append(ps.stem(token))
            tokens = list(set(tokens))
            for token in tokens:
#                check if already exists in list of terms
                flag = False
                for term in terms:
                    if term.name == token:
                        flag = True
                        term.addDocument(document_num)
                        term.increaseOccurrence()
                        term.addSentence(line)
                if flag == False:
                    term = Term(token, document_num, line)
                    terms.append(term)
        terms = sorted(terms, key=lambda term: term.occur_count, reverse=True)
    return terms

def merge_obj(terms1, terms2):
    for element2 in terms2:
        found = False
        for element1 in terms1:
            if element1.name == element2.name:
                element1.occur_count += element2.occur_count
                element1.documentList += element2.documentList
                element1.sentencesList += element2.sentencesList
                found = True
        if found == False:
            terms1.append(element2)
    return terms1

#function to create csv easier for technical read/reuse
def writeCSV(terms, top_results):
    try:
        os.remove('hashtags_out.csv')
    except OSError:
        pass
    #    write the new file
    with open('hashtags_out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Word#', 'Occurrences', 'Documents', 'Sentences'))
        if int(top_results) > len(terms):
            top_results = len(terms)-1
            print('%d hashtags exist.'%(len(terms)-1))
        for term in terms[:int(top_results)]:
            line = (term.name, term.occur_count, '\n'.join('{}'.format(k) for k in term.documentList), '\n'.join('{}: {}'.format(*k) for k in enumerate(term.sentencesList)))
            writer.writerow(line)

#function to create txt file for better presentation
def writeTXT(terms, top_results):
    try:
        os.remove('hashtags_out.txt')
    except OSError:
        pass
#    write the new file
    with open('hashtags_out.txt', 'w') as file:
        if int(top_results) > len(terms):
            top_results = len(terms)-1
            print('%d hashtags exist.' %(len(terms)-1))
        for term in terms[:int(top_results)]:
            file.write(('Word# :\t' + term.name + '\tOccurred: ' + str(term.occur_count) +' times\n'))
            file.write(('Found in documents: ' + ', '.join('{}'.format(k) for k in term.documentList) +'\n'))
            file.write('Sentences occurred:\n\t')
            line = ('\n\t'.join('{}: {}'.format(*k) for k in enumerate(term.sentencesList)))
            file.write(line +'\n\n')

def hashtags(top_results):
    # set stop_words
    stop_words = set(stopwords.words('english'))
    stop_words.add('us')
    # create stem object
    ps = PorterStemmer()
    documents = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt', 'doc5.txt', 'doc6.txt']
    try:
        with open(documents[0], 'r') as file:
            data = file.read()
    except:
        with open(documents[0], 'r', encoding='utf8') as file:
            data = file.read()
    lines = data.splitlines()
    terms = occurrences_table(ps, stop_words, file.name, lines)
    for doc in documents[1:]:
        try:
            with open(doc, 'r') as file:
                data = file.read()
        except:
            with open(doc, 'r', encoding='utf8') as file:
                data = file.read()
        lines = data.splitlines()
        terms_temp = occurrences_table(ps, stop_words, file.name, lines)
        terms = merge_obj(terms, terms_temp)
    terms = sorted(terms, key=lambda term: term.occur_count, reverse=True)
    return terms

def main(argv):
#    arguments checks: if number and if only one number is provided
    if len(argv) > 3 or len(argv) < 1:
        print('Only the number of top results is required and \'csv\' option.')
        print('Try again.')
        sys.exit()
    elif len(argv) == 1:
        print('Default 20 top results will be printed')
        terms = hashtags(20)
        writeTXT(terms, 20)
    elif len(argv) == 2:
        if argv[1].isdigit():
            terms = hashtags(argv[1])
            writeTXT(terms, argv[1])
        else:
            print('Only a positive integer is accepted as an argument.')
            print('Please try again.')
            sys.exit()
    elif len(argv) == 3:
        if argv[1].isdigit():
            csvSTR = ['csv', 'CSV', '.csv', 'Csv', 'excel', 'Excel']
            if argv[2] in csvSTR:
                terms = hashtags(argv[1])
                writeCSV(terms, argv[1])
            else:
                print('Ignoring argument... Exporting txt file.')
                terms = hashtags(argv[1])
                writeTXT(terms, argv[1])
        else:
            print('Only a positive integer is accepted as an argument.')
            print('Please try again.')
            sys.exit()
    
if __name__ == "__main__":
    main(sys.argv)
