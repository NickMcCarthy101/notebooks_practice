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

import en_core_web_sm
spacy_model = en_core_web_sm.load()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def get_ordered_files(raw_data=None):
    if not raw_data:
        with open('Revolving_Loan_cheatsheet.json') as jsonfile:
            raw_data = json.load(jsonfile)
    docs_in_order = [raw_data[0]['Data'][i]['document_name'] for i in range(len(raw_data[0]['Data']))] 
    return docs_in_order

def make_data_with_multiple_answers(raw_data=None):
    if not raw_data:
        with open('Revolving_Loan_cheatsheet.json') as jsonfile:
            raw_data = json.load(jsonfile)
    
    data_dir = '../ING_POC/'
    
    train = [{'doc':data_dir + the_doc['document_name'], 
              'clause':[(ans['start'],ans['end'],ans['text']) for ans in the_doc['document_answers']],
              'class':'Yes', 'id':idx} for idx,the_doc in enumerate(raw_data[0]['Data']) 
                             if 'trainingset' in the_doc['document_name']]
    test = [{'doc':data_dir + the_doc['document_name'], 
              'clause':[(ans['start'],ans['end'],ans['text']) for ans in the_doc['document_answers']],
              'class':'Yes', 'id':idx} for idx,the_doc in enumerate(raw_data[0]['Data']) 
                             if 'evaluationset' in the_doc['document_name']]

    return {'train': train,
            'test':  test,
           }


def ntfr_word_score(term_freq_df, term_freq_in_doc_df, element_freq_df,
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

def bm25_sec_score(tokenised_elements, word_scores, term_freq_df,
                   term_freq_in_doc_df, element_freq_df, normalisation_df,
                   k1=1.75, b=0.75, delta=1):
    """
        BM25
    """
    rtn = 0.
    average_section_len = normalisation_df['sec_len'].mean()
    norm_doc_length = len(tokenised_elements)/average_section_len
    new_term_freq_dict = defaultdict(int)
    for token in tokenised_elements:
        if token in word_scores:
            new_term_freq_dict[token] += 1
    for token, freq in new_term_freq_dict.items():
        den = element_freq_df.notnull().aggregate(any,axis=1).sum()
        num = element_freq_df.sum()[token]
        idf = den/num
        rtn += (freq*(k1+1)/(freq + k1*(1-b+b*norm_doc_length)) + delta)*idf
    return rtn


def make_predictor(ngram=2):
    
    bm25_filter_params = ClauseFilterParams(word_scorer=ntfr_word_score,
                                            section_scorer=bm25_sec_score,
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


def prepare_docs_with_multiple_answers(raw_docs):
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

# TODO use better variable names and add description
def bow_similarity(vocab, window, compare, ngram = 1):
    max_score = 0
    max_idx = -1
    assert window <= len(compare)
    rtn = []
    can_print = False
    window_score = 0
    for i in range(len(compare)-window + 1):
        if can_print and int(float(i)/(len(compare)-window)*100)%10 == 0:
            can_print = False
            print('Progress: {}%'.format(int(float(i)/(len(compare)-window)*100)), end='\r')
        elif not can_print and int(float(i)/(len(compare)-window)*100)%10 == 1:
            can_print = True
        if not i:
            number_of_loops = window - ngram + 1
            window_score = sum([vocab[' '.join([compare[w+ii] for ii in range(ngram)])] 
                                for w in range(number_of_loops)#window-ngram+1) 
                                if ' '.join([compare[w+ii] for ii in range(ngram)]) in vocab])
#            print([[w+ii for ii in range(ngram)] for w in range(window-ngram+1)])
        else:
            ngram_in = ' '.join([compare[i+window-ii] for ii in range(ngram,0,-1)])
            ngram_out = ' '.join([compare[i+ii-1] for ii in range(ngram)])
#            print([i+window-ii for ii in range(ngram,0,-1)], ngram_in, 
#                  [i+ii-1 for ii in range(ngram)], ngram_out)
            change = (vocab[ngram_in] if ngram_in in vocab else 0) - \
                     (vocab[ngram_out] if ngram_out in vocab else 0)
            window_score += change
        rtn.append(window_score)
        if rtn[-1] > max_score:
            max_idx = i
            max_score = rtn[-1]
    print('peak score: {}; and position: {} of top {} words'.format(max_score,max_idx, len(vocab)))
    return rtn

# main function for analysis of a single file
def file_signal(filename, spacy_doc=None, no_stopwords=True, ax=None):
    if not spacy_doc:
        with open(filename) as fin:
            raw_text = fin.read()
        spacy_doc = spacy_model(raw_text) 

    tokenised_doc_indices = [tk.i for tk in spacy_doc if not (tk.is_stop or tk.is_punct or tk.is_space)]
    tokenised_doc = [stemmer.stem(spacy_doc[i].lemma_) for i in tokenised_doc_indices]

    # get signal
    window = 20
    ngram = 2
    file_signal_no_stopwords = bow_similarity(bow_vocab, window, tokenised_doc, ngram)

    # get answer without stopwords
    true_ans_limits_no_stopwords, true_ans_limits_raw_text = get_true_answer(spacy_doc, tokenised_doc_indices, raw_data)

    # get signal for raw text
    file_signal_raw_text = get_raw_text_signal(file_signal_no_stopwords, tokenised_doc_indices, spacy_doc)

    # get span(s) around best peak(s)
    sig_array, span_window = (file_signal_no_stopwords, window) if no_stopwords else \
                             (file_signal_raw_text, int(window*len(spacy_doc)/float(len(tokenised_doc_indices))))
    list_of_limits_of_peaks = get_peak_limits(sig_array, span_window, spacy_doc)

    if ax:
        if no_stopwords:
            ax = plot_single_file(file_signal_no_stopwords, 
                                  true_ans_limits_no_stopwords, 
                                  list_of_limits_of_peaks,
                                  filename, 
                                  ax)
        else:
            ax = plot_single_file(file_signal_raw_text, 
                                  true_ans_limits_raw_text,
                                  list_of_limits_of_peaks, 
                                  filename,
                                  ax)

    if no_stopwords:
        text_snippets = text_in_peaks(list_of_limits_of_peaks, spacy_doc, tokenised_doc_indices)
    else:
        text_snippets = text_in_peaks(list_of_limits_of_peaks, spacy_doc) 
    return text_snippets

def plot_single_file(signal, true_ans, peak_spans, filename, ax):
    x = list(range(len(signal)))
    y = signal
    ax.plot(x,y)

    ymax = ax.get_ylim()[1]
    start_tok, end_tok = true_ans
    ax.plot([start_tok,start_tok],[0,ymax], c='#ff7f0e', zorder=0)
    ax.plot([end_tok,end_tok],[0,ymax], c='#ff7f0e', zorder=0)

    for xi in range(peak_spans.shape[0]):
        start_tok = peak_spans[xi,0]
        end_tok = peak_spans[xi,1]
        ax.plot([start_tok,start_tok],[0,ymax], c='#7f7f7f', ls='--', zorder=0)
        ax.plot([end_tok,end_tok],[0,ymax], c='#7f7f7f', ls='--', zorder=0)
    
    text_color = 'red' if 'evaluationset' in filename else 'black'
    ax.text(0.6, 0.9, docs_in_order[doc_i], transform=ax.transAxes, fontsize=15, color=text_color) 
    return ax

def text_in_peaks(peak_spans, spacy_doc, tokenised_doc_indices=None):
    rtn = []
    for xi in range(peak_spans.shape[0]):
        start_tok = peak_spans[xi,0] if not tokenised_doc_indices else tokenised_doc_indices[peak_spans[xi,0]]
        end_tok = peak_spans[xi,1] if not tokenised_doc_indices else tokenised_doc_indices[peak_spans[xi,1]]
        curr_str = ''.join([spacy_doc[i].text_with_ws for i in range(start_tok, end_tok+1)])
        rtn.append((start_tok, end_tok, curr_str))
    return rtn

def get_peak_limits(signal, window, spacy_doc):
    # If two peaks are closer together than a window width - combine into one span
    # elif the distance is < 3*window - place a border at ratio 2:1
    # else - the right border of the left span is peak + 2*window 
    #        and the left border of the right span is peak - window
    top_hits = []
    top_val = 0
    y = signal
    x = list(range(len(signal)))
    for i,(yi,xi) in enumerate(sorted(zip(y,x), reverse=True)):
        if not i:
            top_val = yi
            top_hits.append(xi)
        elif yi == top_val:
            top_hits.append(xi)
        else:
            break
# NOTE the next two lines are a way to estimate sig eff. and bkg rej. 
#    peaks_in_ans = sum([start_tok < i < end_tok for i in top_hits])
#    peaks_total = len(top_hits)
    top_hits = sorted(top_hits)
    peak_limits = np.full((len(top_hits),2), -1, dtype=int)
    xi = 0
    peak_limits[xi,0] = max(top_hits[xi] - window, 0)
    while xi < len(top_hits) - 1: # for xi in range(len(top_hits) - 1):
        distance = top_hits[xi+1] - top_hits[xi]
        if distance <= window:
            top_hits.pop(xi)
            xi -= 1
        elif distance <= 3*window:
            squeezed_window = int(distance/3.)
            peak_limits[xi,1] = top_hits[xi] + 2*squeezed_window
            peak_limits[xi+1,0] = top_hits[xi+1] - squeezed_window
        else:
            peak_limits[xi,1] = top_hits[xi] + 2*window
            peak_limits[xi+1,0] = top_hits[xi+1] - window
        xi += 1
    peak_limits[xi,1] = min(top_hits[xi] + 2*window, len(spacy_doc)-1)
    peak_limits = peak_limits[:xi+1,:]        
    return peak_limits

def get_true_answer(spacy_doc, tokenised_doc_indices, raw_data=None):
    if not raw_data:
        with open('Revolving_Loan_cheatsheet.json') as jsonfile:
            raw_data = json.load(jsonfile)
    
    ans_start = raw_data[1]['Data'][doc_i]['document_answers'][0]['start']
    ans_end = raw_data[1]['Data'][doc_i]['document_answers'][-1]['end']
    get_start = True
    for i,tok in enumerate(tokenised_doc_indices):
        if get_start and spacy_doc[tok].idx > ans_start:
            start_tok = max(i-1,0)
            get_start = False
        if (not get_start) and spacy_doc[tok].idx + len(spacy_doc[tok].text_with_ws) >= ans_end:
            end_tok = i
            break
    return (start_tok,end_tok),(ans_start,ans_end)

def get_raw_text_signal(signal, tokenised_doc_indices, spacy_doc):
    signal_raw_text = np.interp(range(len(spacy_doc)), tokenised_doc_indices[:len(signal)], signal)
    return signal_raw_text

if __name__ == "__main__":
   
    with open('Revolving_Loan_cheatsheet.json') as jsonfile:
        raw_data = json.load(jsonfile)

    docs_in_order = get_ordered_files(raw_data)
    _ngram = 2
    try:
        with open('bow_vocab_ngram_{_ngram}.pkl','rb') as fin:
            bow_vocab = pickle.load(fin)
    except FileNotFoundError: 
        data = make_data_with_multiple_answers(raw_data)
        train_text = prepare_docs_with_multiple_answers(data['train'])
        docs_to_train = [doc['document'] for doc in train_text]
        answers = {doc['document'].id: doc['training_data'] for doc in train_text}
        # Link files to spacy model and answers
        predictor = make_predictor(ngram=_ngram)
        predictor.preprocess_docs(docs_to_train,spacy_model)
        predictor.populate_answer_locations_for_docs(docs_to_train,answers)
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
        bow_vocab = {k:v for v,k in sorted([(v,k) for k,v in my_filter.word_scores.items()],reverse=True)[:nvocab]}
        with open('bow_vocab_ngram_{_ngram}.pkl','wb') as outf:
            pickle.dump(bow_vocab,outf)
    
    
    print("BEGIN file analysis")

    rows, cols = 2,2 #9, 3 
    fig, axs = plt.subplots(rows,cols)
    fig.set_size_inches(10,20)
    doc_i = -1
    for row in range(rows):
        for col in range(cols):
            ax = axs[row][col]
            doc_i += 1
            #print(docs_in_order[doc_i])
            filename = exe_dir + '../ING_POC/{}'.format(docs_in_order[doc_i])
            with open(filename) as fin:
                raw_text = fin.read()
            spacy_doc = spacy_model(raw_text) 
            

            text_in_answer = file_signal(filename,spacy_doc=spacy_doc, ax=ax)
            feature_values = np.zeros(len(spacy_doc))

            file_str = 60*'-'+'\n'+docs_in_order[doc_i]+'\n'
            for s,e,txt in text_in_answer:
                span_str = '(start={}, end={})\n\n'.format(s,e) + txt + '\n...\n\n' 
                file_str += span_str
                # separate operation
                feature_values[s:e] = 1
            if text_in_answer:
                file_str = file_str[:-6] + '\n\n'

            print(file_str)
    fig.savefig('ThePlot.pdf')

