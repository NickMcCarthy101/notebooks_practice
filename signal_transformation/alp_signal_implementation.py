import sys
#sys.path.append("/home/petar.petrov/eigentech")
sys.path.append('/home/petar.petrov/prototypes')
import json
import pickle
import numpy as np
import pandas as pd

import re

from collections import Iterable
from itertools import product

from scipy.stats import hmean, genlogistic

#import spacy
#import en_core_web_sm
#spacy_model = en_core_web_sm.load()

from alp.AlpCalculations import alp_similarity as alp

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from base_signal import BaseSignalCalculator

###############################################################################
## Find all combinations of matches(probes) that cover enough of the examples
###############################################################################
class ProbeBuilder:
    # TODO: Figure out how to use references and pointers in numpy arrays instead of lists
    def __init__(self, unique_matches, original_probes, coverage_ratio_thres):
        self.all_matches = unique_matches
        self.all_strings = original_probes
        self.coverage_ratio_thres = coverage_ratio_thres
        # stack of potential list of probes that cover enough of the examples
        self.match_stack = list()
        # stack of remaining examples that need to be covered
        self.remaining_stack = list(range(len(self.all_strings)))
        # return list of suggested probe subsets and their coverage
        self.combos = list()
        self.recurring_function()
        
    
    
    def recurring_function(self,i_prev = -1):
        # TODO, calc len(self.all_matches) in init and save value
        for i in range(i_prev + 1, len(self.all_matches)):
            add = False
            if any([(self.is_subset(self.all_matches[j],self.all_matches[i]) or 
                    self.is_subset(self.all_matches[i],self.all_matches[j]))
                    for j in self.match_stack]):
                continue
            # strings covered by all_matches[i]
            local_stack = list()
            s = 0
            while s < len(self.remaining_stack):
                if self.is_subset(self.all_matches[i],self.all_strings[self.remaining_stack[s]]):
                    add = True
                    # how can this be efficient? Not really a stack operation
                    # maybe use position in all_strings instead of actual string
                    local_stack.append(self.remaining_stack.pop(s))
                    s -= 1
                s += 1
            if not add:
                continue
            try:
                current_coverage = 1. - float(len(self.remaining_stack))/len(self.all_strings)
            except ZeroDivisionError:
                print('Why are there no examples in self.all_strings?')
                raise
            self.match_stack.append(i)
            if current_coverage < self.coverage_ratio_thres:
                self.recurring_function(i)
            else:
                self.combos.append((current_coverage,[self.all_matches[ii] for ii in self.match_stack]))
            self.match_stack.pop()
            while local_stack:
                self.remaining_stack.append(local_stack.pop())

    def is_subset(self, small_set, big_set):
        # Assume the objects in the lists can be compared with ==
        if len(small_set) > len(big_set):
            return False
        assert len(small_set)
        assert type(small_set) == type(big_set)
        if isinstance(small_set,str):
            return small_set in big_set
        for i in range(len(big_set)-len(small_set) + 1):
            found_match = True
            for j in range(len(small_set)):
                if small_set[j] != big_set[i+j]:
                    found_match = False
                    break
            if found_match:
                return True
        return False

##########################################################################
##########################################################################
class ALPAutoProbeSignalCalculator(BaseSignalCalculator):
    # TODO figure out necessary arguments
    def __init__(self, probes_filename, param_point, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The __init__ function should do whatever training is needed
        # for the current implementation to work, therefore it should
        # transform the data for training in the necessary format 
        # which is up to you to decide.
        
        self._probes_filename = probes_filename
        # TODO self._data_dir assignment assumes a very specific filepath
        #      structure between <cheatsheet> and the training files 
        self._data_dir = args[-1].split('/')[:-1]
        if not self._data_dir:
            self._data_dir = '../ING_POC/'
        else:
            self._data_dir += '/../ING_POC/'
        
        # The rest is probe training
        try:
            with open(self._probes_filename, 'rb') as fin:
                self.auto_probes = pickle.load(fin)
        except FileNotFoundError:
            print('Training auto-probes...',end='\r')
            # TODO passing the spacy_model is an issue
            self.auto_probes = self.build_probes(param_point, self.raw_data, args[2]) 
            print('',end='\r')
            print('Auto-probes trained.')
            with open(self._probes_filename, 'wb') as outf:
                pickle.dump(self.auto_probes, outf)

##########################
## ALP score calculation
##########################
    def build_both_signals(self, window):
        # TODO is window a universal argument and does it make sense to define it in __init__
        self.window=window
        tmp_signal = self.build_signal(token_ids=self.tokenised_doc_indices)
        self.signal = [tmp_signal, self.get_raw_text_signal(tmp_signal)]
       
    def build_signal(self, token_ids): # TODO do I need token_ids if it is self.tokenised_doc_indices?
        tokenised_doc = [stemmer.stem(self.spacy_doc[i].lemma_) for i in token_ids]
        # TODO should use arrays here because we know both size and type
        individual_signals = []
        for probe in self.auto_probes:
            individual_signals.append(self.alp_similarity(probe,tokenised_doc))
        rtn = self.combine_signal(individual_signals)
        return rtn

    def alp_similarity(self, probe, tokenised_doc):
        max_score = 0
        max_idx = -1
        assert len(probe) <= len(tokenised_doc)
        rtn = []
        can_print = False
        for i in range(len(tokenised_doc)-len(probe)):
            if can_print and int(float(i)/(len(tokenised_doc)-len(probe))*100)%10 == 0:
                can_print = False
                print('Progress: {}%'.format(int(float(i)/(len(tokenised_doc)-len(probe))*100)), end='\r')
            elif not can_print and int(float(i)/(len(tokenised_doc)-len(probe))*100)%10 == 1:
                can_print = True
            rtn.append(alp(probe,tokenised_doc[i:i+len(probe)]))
            if rtn[-1] > max_score:
                max_idx = i
                max_score = rtn[-1]
        print('peak score: {}; and position: {} of probe: {}'.format(max_score,max_idx, ' '.join(probe)))
        return rtn

    def combine_signal(self, individual_signals):
        rtn = list(map(max,zip(*individual_signals)))
        return rtn




##########################################################################
### TRAINING
##########################################################################
    def build_probes(self, param_point, raw_data, spacy_model):
        # prepare probes
        all_probes = [self.probe_tokenisation(raw_data[0]['Data'][i]['document_answers'][0]['text'],spacy_model) 
                      for i in range(len(raw_data[0]['Data'])) 
                      if 'trainingset' in raw_data[0]['Data'][i]['document_name']]

        min_match_len = param_point[0]
        coverage_ratio_thres = param_point[1]
        lc_ratio = param_point[2]

        print('probe parameters: min_len={}; min_coverage={}; length_to_coverage_weight_ratio={}'
              .format(min_match_len,coverage_ratio_thres,lc_ratio))

        all_matches = self.get_all_matches_from_list(all_probes, min_match_len)
        probe_set = ProbeBuilder(all_matches, all_probes, coverage_ratio_thres)
        if not len(probe_set.combos):
            return None
        final = self.calc_scores(
            final_probes=probe_set.combos, 
            all_strings=all_probes, 
            l_c_ratio=lc_ratio,
        )
        ordered_indices = list(zip(*sorted([(v,i) for i,v in enumerate(final['final'])], reverse=True)))[1]
            
        tokenized_probes = final['match_list'][ordered_indices[0]]

        print('PROBE LIST:')
        for prb in tokenized_probes:
            print(prb)

        return tokenized_probes

    def probe_tokenisation(self, text, spacy_model):
        spacy_probe = spacy_model(text)
        # NOTE could also use a stemmer on the lemma but we need nltk for that
        rtn = [stemmer.stem(tk.lemma_) for tk in spacy_probe if not (tk.is_stop or tk.is_punct or tk.is_space)]
        return rtn

    #################################################
    ## Find all repeated elements within the probes
    #################################################
    def get_all_matches_from_list(self, list_strings, min_match_len=5):
        len_list = len(list_strings)
        all_matched_strings = []
        for str1 in range(len_list-1):
            for str2 in range(str1+1,len_list):
                all_matched_strings+= self.common_substring_list_conv(list_strings[str1], list_strings[str2], min_match_len)
        all_matched_strings = sorted(self.set_from_iterables(all_matched_strings), key=len, reverse=True)
        return all_matched_strings

    def common_substring_list_static(self, str1, str2, min_match_len):
        loop_len = min(len(str1), len(str2))
        curr_match_len = 0
        matched_strings = []
        for i in range(loop_len):
            if str1[i] == str2[i]:
                curr_match_len+=1
                if i == loop_len-1 and curr_match_len >= min_match_len:
                    curr_matched_string = str1[(i-curr_match_len+1):(i+1)]
                    matched_strings.append(curr_matched_string)
            else:
                if curr_match_len >= min_match_len:
                    curr_matched_string = str1[(i-curr_match_len):i]
                    matched_strings.append(curr_matched_string)
                curr_match_len = 0
        return matched_strings

    def common_substring_list_conv(self, str1, str2, min_match_len):
        len1 = len(str1)
        len2 = len(str2)
        strlen = max(len1, len2)

        all_matched_strings = []
        for t in range(strlen):
            x1 = str1[(strlen-t-1):strlen ]
            all_matched_strings = all_matched_strings+self.common_substring_list_static(x1, str2, min_match_len)
        for t in range(strlen):
            x2 = str2[(strlen - t - 1):strlen]
            all_matched_strings = all_matched_strings + self.common_substring_list_static(x2, str1, min_match_len)

        all_matched_strings = sorted(self.set_from_iterables(all_matched_strings), key=len, reverse=True)
        return all_matched_strings

    ###########################################
    ## Combine scores from different factors
    ###########################################
    def calc_scores(self, final_probes, all_strings, l_c_ratio=1.):
        coverage_score, success_match_list = zip(*final_probes)
        length_score = [] # harmonic mean of the lengths
        final_score = []

        for cr, sm in final_probes:
            l = hmean(list(map(len,sm)))
            length_score.append(l)
            fs = self.final_score_function(cr,0,l,len(sm), l_c_ratio)
            final_score.append(fs)
        
        ret = {
            'final': final_score,
            'coverage': coverage_score,
            'overlap': [0]*len(coverage_score),
            'length': length_score,
            'match_list': success_match_list
        }
        return ret

    def final_score_function(self, coverage, overlap, length, number, l_c_ratio,):
        coverage_score = self.transform_coverage(coverage)
        length_score = self.transform_length(length)
        overlap_score = overlap
        weight_c = 1/(1+l_c_ratio)
        weight_l = l_c_ratio/(1+l_c_ratio)
        weight_o = 0.
        base_n = 1.2 # not sure a power law is the best solution here
        return (coverage_score*weight_c + length_score*weight_l + overlap_score*weight_o)*pow(base_n,-number)

    def transform_coverage(self, coverage):
        p_c = 0.3
        scale_c = genlogistic.ppf(0.99,p_c)-genlogistic.ppf(0.01,p_c)
        loc_c = genlogistic.ppf(0.01,p_c)
        coverage_score = genlogistic.cdf(coverage*scale_c,p_c,-loc_c)
        return coverage_score

    def transform_length(self, length):
        p_l = 0.3
        scale_l = genlogistic.ppf(0.99,p_l)-genlogistic.ppf(0.01,p_l)
        loc_l = genlogistic.ppf(0.01,p_l)
        length_score = 1 - genlogistic.cdf((1-length/15)*scale_l,p_l,-loc_l)
        return length_score


    #########################
    ## Auxiliary functions
    #########################
    def alpha_only_string(self, list_strings):
        new_list = []
        for s in list_strings:
            s = re.sub(r'\W+', '', s)
            new_list.append(s.lower())
        return new_list

    # NOTE like set() but for list of lists (will hashes help here?)
    def set_from_iterables(self, iter_objects):
        rtn = iter_objects.copy()
        assert all(map(lambda x: isinstance(x,Iterable),iter_objects))
        # How do I check if objects can be compared with == operator?
        # Right now only assume this is the case
        i = 0
        while i < len(rtn)-1:
            for j in range(i+1,len(rtn)):
                if rtn[i] == rtn[j]:
                    rtn.pop(i)
                    i -= 1
                    break
            i += 1
        return rtn


