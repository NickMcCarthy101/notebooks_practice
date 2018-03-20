#TODO everything file-related is hardcoded, think of adding flexibility
#     - json file with raw data is fixed
#     - data_dir is fixed
#     - the separation between train and test docs is fixe 
#       (string 'trainingset' or 'evaluationset' in file name)
import sys
import os
import re
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
# substitute your path to eigen
exe_dir = '/home/petar.petrov/Desktop/Project_Beaker_PoC/Revolving_Loan_Analysis/'
maindir = '/home/petar.petrov/eigen/'
sys.path.append(maindir)
os.chdir(maindir)
os.environ["DJANGO_SETTINGS_MODULE"] = "eigenapp.settings"
import django
django.setup()
    
try:
    from common.lib.predictive_pipeline.feature_extractors.clause_filter_feature_extractor import \
        ClauseFilterFeatureExtractor
    from common.lib.predictive_pipeline.feature_extractors.clause_filter_params import \
        ClauseFilterParams
    from common.lib.predictive_pipeline.feature_extractors.clause_filter_algorithms import\
        BoWFilter
except ModuleNotFoundError:
    print('Make sure the eigen repo is on branch <feature-text_filter>')
    raise
from common.lib.predictive_pipeline.pipeline_document import PipelineDocument
from common.lib.predictive_pipeline.pipelines.default_params import default_replacers
from common.lib.predictive_pipeline.pipelines.pipeline_params import PipelineParams
from common.lib.predictive_pipeline.predictor import Predictor
from common.lib.sectioning.DataStructure import Range
from common.lib.sectioning.Document import Document

os.chdir(exe_dir)

#import en_core_web_sm
#spacy_model = en_core_web_sm.load()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from base_signal import BaseSignalCalculator

class SBoWSignalCalculator(BaseSignalCalculator):
    # TODO figure out necessary arguments
    def __init__(self, vocab_filename, ngram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The __init__ function should do whatever training is needed
        # for the current implementation to work, therefore it should
        # transform the data for training in the necessary format 
        # which is up to you to decide.
        
        self._vocab_filename = vocab_filename
        self._ngram = ngram
        # TODO self._data_dir assignment assumes a very specific filepath
        #      structure between <cheatsheet> and the training files 
        self._data_dir = args[-1].split('/')[:-1]
        if not self._data_dir:
            self._data_dir = '../ING_POC/'
        else:
            self._data_dir += '/../ING_POC/'
        
        # The rest is vocab training
        try:
            with open(self._vocab_filename,'rb') as fin:
                self.bow_vocab = pickle.load(fin)
        except FileNotFoundError: 
            data = self.transform_data(self.raw_data)
            train_text = self.prepare_docs(data['train'])
            docs_to_train = [doc['document'] for doc in train_text]
            answers = {doc['document'].id: doc['training_data'] 
                       for doc in train_text}
            # Link files to spacy model and answers
            predictor = self.make_predictor(ngram=self._ngram)
            predictor.preprocess_docs(docs_to_train, args[2]) # TODO There must be a better way to pass the spacy_model object than args[1], maybe even kwargs['spacy_model']
            predictor.populate_answer_locations_for_docs(docs_to_train, answers)
            # prepare and train filter
            params = {'pipeline_docs':docs_to_train}
            my_filter = BoWFilter(predictor.feature_params['filter_params'])
            print('start fitting the vocabulary...', end='\r')
            my_filter.fit(params)
            # At this point there is access to the trained predictor and all its nested elements
            print('', end='\r')
            print('vocabulary is ready!')
            # vocab info to use as a crf feature
            nvocab = 10
            ordered_words = {k:(i,v) for i,(v,k) in enumerate(sorted([(v,k) for k,v in my_filter.word_scores.items()],reverse=True))}
            self.bow_vocab = {k:v for v,k in sorted([(v,k) for k,v in my_filter.word_scores.items()],reverse=True)[:nvocab]}
            with open(self._vocab_filename,'wb') as outf:
                pickle.dump(self.bow_vocab,outf)

    def build_both_signals(self, window):
        # TODO is window a universal argument and does it make sense to define it in __init__
        self.window=window
        tmp_signal = self.build_signal(window=self.window, token_ids=self.tokenised_doc_indices)
        self.signal = [tmp_signal, self.get_raw_text_signal(tmp_signal)]
       
    def build_signal(self, window, token_ids): # TODO do I need token_ids if it is self.tokenised_doc_indices?
        tokenised_doc = [stemmer.stem(self.spacy_doc[i].lemma_) for i in token_ids]
        max_score = 0
        max_idx = -1
        assert window <= len(tokenised_doc)
        rtn = []
        can_print = False
        window_score = 0
        for i in range(len(tokenised_doc)-window + 1):
            if can_print and int(float(i)/(len(tokenised_doc)-window)*100)%10 == 0:
                can_print = False
                print('Progress: {}%'.format(int(float(i)/(len(tokenised_doc)-window)*100)), end='\r')
            elif not can_print and int(float(i)/(len(tokenised_doc)-window)*100)%10 == 1:
                can_print = True
            if not i:
                number_of_loops = window - self._ngram + 1
                window_score = sum([self.bow_vocab[' '.join([tokenised_doc[w+ii] for ii in range(self._ngram)])] 
                                    for w in range(number_of_loops)#window-self._ngram+1) 
                                    if ' '.join([tokenised_doc[w+ii] for ii in range(self._ngram)]) in self.bow_vocab])
    #            print([[w+ii for ii in range(self._ngram)] for w in range(window-self._ngram+1)])
            else:
                ngram_in = ' '.join([tokenised_doc[i+window-ii] for ii in range(self._ngram,0,-1)])
                ngram_out = ' '.join([tokenised_doc[i+ii-1] for ii in range(self._ngram)])
    #            print([i+window-ii for ii in range(self._ngram,0,-1)], ngram_in, 
    #                  [i+ii-1 for ii in range(self._ngram)], ngram_out)
                change = (self.bow_vocab[ngram_in] if ngram_in in self.bow_vocab else 0) - \
                         (self.bow_vocab[ngram_out] if ngram_out in self.bow_vocab else 0)
                window_score += change
            rtn.append(window_score)
            if rtn[-1] > max_score:
                max_idx = i
                max_score = rtn[-1]
        print('peak score: {}; and position: {} of top {} words'.format(max_score,max_idx, len(self.bow_vocab)))
        return rtn


    # Functions used for the training of the vocabulary
    def transform_data(self, raw_data):
        train = [{'doc':self._data_dir + the_doc['document_name'], 
                  'clause':[(ans['start'],ans['end'],ans['text']) for ans in the_doc['document_answers']],
                  'class':'Yes', 'id':idx} for idx,the_doc in enumerate(raw_data[0]['Data']) 
                                 if 'trainingset' in the_doc['document_name']]
        test = [{'doc':self._data_dir + the_doc['document_name'], 
                  'clause':[(ans['start'],ans['end'],ans['text']) for ans in the_doc['document_answers']],
                  'class':'Yes', 'id':idx} for idx,the_doc in enumerate(raw_data[0]['Data']) 
                                 if 'evaluationset' in the_doc['document_name']]

        return {'train': train,
                'test':  test,
               }


    def prepare_docs(self, raw_docs):
        """
        Similar to the way self.docs is constructed in Statistician.load_data
        with some minor difference

        rtn = {i:element for i in range(len(raw_docs))}
        element = {'document':pipelineDocument(i,sectioning_doc),
                   'training_data':{'StartPos':startpos, 'EndPos':endpos, 'text':'Yes'/'No'}
        
        for each file:
            make pipeline_doc,
            get range from text of clause
            add the class as the value to the 'text' key
            
        """
        rtn = []
        for datum in raw_docs:
            secdoc = Document.fromFile(datum['doc'])
            element = {}
            element['training_data'] = []
            element['document'] = PipelineDocument(datum['id'], secdoc)
            # element['training_data'] can have multiple ranges if the answer
            # spans more than one consecutive sections. Here we collect all. 
            if datum['clause']:
                for chunk in datum['clause']:
                    rng = secdoc.rangeFromText(chunk[2])
                    if not rng:
                        rng = Range(secdoc.posFromTextOffset(chunk[0]),
                                    secdoc.posFromTextOffset(chunk[1]))
                    element['training_data'] += [rng.toJson()]
                    element['training_data'][-1]['text'] = datum['class']
            else:
                element['training_data'] += [{'text': datum['class']}]
            rtn.append(element)
        return rtn

    def make_predictor(self, ngram=2):
        
        bm25_filter_params = ClauseFilterParams(word_scorer=self.ntfr_word_score,
    #                                            section_scorer=self.bm25_sec_score,
                                                section_scorer_params={
                                              'k1': 1.75, 'b': 0.75,
                                              'delta': 1},
                                                vocab_ngram_range=(ngram,ngram),
                                                no_filter_elements=1,
                                                skip_filter=False)
        simple_cv_params = None

        filtered_standard_classifier_pipeline_params = PipelineParams(
            replacers=default_replacers,
            classifier='MNB',
            feature_extractor=ClauseFilterFeatureExtractor,
            feature_params={
                'filter_params': bm25_filter_params,
                'vect_params': simple_cv_params},
            attrs_to_pickle =['classifier', 'feature_extractor'])

        return Predictor(**filtered_standard_classifier_pipeline_params._asdict())


    # Functions for vocabulary 
    def ntfr_word_score(self, term_freq_df, term_freq_in_doc_df, element_freq_df,
                        normalisation_df):
        """
        ntf - normalise the entries of filt.tf to the length of the
              corresponding relevant text
        ntfd - normalise the entries of filt.tfd to the lenght of the
               corresponding document text
        The final score is an element-wise ratio between the two
        and then summed over the rows (files)
        """
        n_best = -1
        ntf = term_freq_df.apply(lambda row: row/normalisation_df['sec_len'])
        ntfd = term_freq_in_doc_df.apply(lambda row: row/normalisation_df['doc_len'])
        ntfr = ntf/ntfd
        return (ntfr.fillna(0).apply(sum)/
               ntfr.notnull().aggregate(any, axis=1).sum())\
               .sort_values(ascending=False)[:n_best].to_dict()

    #def bm25_sec_score(tokenised_elements, word_scores, term_freq_df,
    #                   term_freq_in_doc_df, element_freq_df, normalisation_df,
    #                   k1=1.75, b=0.75, delta=1):
    #    """
    #        BM25
    #    """
    #    rtn = 0.
    #    average_section_len = normalisation_df['sec_len'].mean()
    #    norm_doc_length = len(tokenised_elements)/average_section_len
    #    new_term_freq_dict = defaultdict(int)
    #    for token in tokenised_elements:
    #        if token in word_scores:
    #            new_term_freq_dict[token] += 1
    #    for token, freq in new_term_freq_dict.items():
    #        den = element_freq_df.notnull().aggregate(any,axis=1).sum()
    #        num = element_freq_df.sum()[token]
    #        idf = den/num
    #        rtn += (freq*(k1+1)/(freq + k1*(1-b+b*norm_doc_length)) + delta)*idf
    #    return rtn
    #




