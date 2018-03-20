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

import spacy
import en_core_web_sm
spacy_model = en_core_web_sm.load()

from alp.AlpCalculations import alp_similarity as alp


def probe_tokenization(text,spacy_model):
    spacy_doc = spacy_model(text)
    # NOTE could also use a stemmer on the lemma but we need nltk for that
    rtn = [tk.lemma_ for tk in spacy_doc if not (tk.is_stop or tk.is_punct or tk.is_space)]
    return rtn

##########################
## ALP score calculation
##########################
def alp_similarity(probe, compare):
    max_score = 0
    max_idx = -1
    assert len(probe) <= len(compare)
    rtn = []
    can_print = False
    for i in range(len(compare)-len(probe)):
        if can_print and int(float(i)/(len(compare)-len(probe))*100)%10 == 0:
            can_print = False
            print('Progress: {}%'.format(int(float(i)/(len(compare)-len(probe))*100)), end='\r')
        elif not can_print and int(float(i)/(len(compare)-len(probe))*100)%10 == 1:
            can_print = True
        rtn.append(alp(probe,compare[i:i+len(probe)]))
        if rtn[-1] > max_score:
            max_idx = i
            max_score = rtn[-1]
    print('peak score: {}; and position: {} of probe: {}'.format(max_score,max_idx, ' '.join(probe)))
    return list(enumerate(rtn))


def max_probe_signal(probes, compare):
    # TODO should use arrays here because we know both size and type
    individual_signals = []
    for probe in probes:
        individual_signals.append(alp_similarity(probe,compare))
    rtn = combine_signal(individual_signals)
    return rtn

def combine_signal(individual_signals):
    # TODO this function can be augmented to do a more sophisticated signal combination
    rtn = list(map(max,zip(*individual_signals)))
    return rtn

#################################################
## Find all repeated elements within the probes
#################################################
def get_all_matches_from_list(list_strings, min_match_len=5):
    len_list = len(list_strings)
    all_matched_strings = []
    for str1 in range(len_list-1):
        for str2 in range(str1+1,len_list):
            all_matched_strings+= common_substring_list_conv(list_strings[str1], list_strings[str2], min_match_len)
    all_matched_strings = sorted(set_from_iterables(all_matched_strings), key=len, reverse=True)
    return all_matched_strings

def common_substring_list_static(str1, str2, min_match_len):
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

def common_substring_list_conv(str1, str2, min_match_len):
    len1 = len(str1)
    len2 = len(str2)
    strlen = max(len1, len2)

    all_matched_strings = []
    for t in range(strlen):
        x1 = str1[(strlen-t-1):strlen ]
        all_matched_strings = all_matched_strings+common_substring_list_static(x1, str2, min_match_len)
    for t in range(strlen):
        x2 = str2[(strlen - t - 1):strlen]
        all_matched_strings = all_matched_strings + common_substring_list_static(x2, str1, min_match_len)

    all_matched_strings = sorted(set_from_iterables(all_matched_strings), key=len, reverse=True)
    return all_matched_strings

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
            if any([(is_subset(self.all_matches[j],self.all_matches[i]) or 
                    is_subset(self.all_matches[i],self.all_matches[j]))
                    for j in self.match_stack]):
                continue
            # strings covered by all_matches[i]
            local_stack = list()
            s = 0
            while s < len(self.remaining_stack):
                if is_subset(self.all_matches[i],self.all_strings[self.remaining_stack[s]]):
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


###########################################
## Combine scores from different factors
###########################################
def calc_scores(final_probes, all_strings, l_c_ratio=1.):
    coverage_score, success_match_list = zip(*final_probes)
    length_score = [] # harmonic mean of the lengths
    final_score = []

    for cr, sm in final_probes:
        l = hmean(list(map(len,sm)))
        length_score.append(l)
        fs = final_score_function(cr,0,l,len(sm), l_c_ratio)
        final_score.append(fs)
    
    ret = {
        'final': final_score,
        'coverage': coverage_score,
        'overlap': [0]*len(coverage_score),
        'length': length_score,
        'match_list': success_match_list
    }
    return ret

def final_score_function(coverage, overlap, length, number, l_c_ratio):
    coverage_score = transform_coverage(coverage)
    length_score = transform_length(length)
    overlap_score = overlap
    weight_c = 1/(1+l_c_ratio)
    weight_l = l_c_ratio/(1+l_c_ratio)
    weight_o = 0.
    base_n = 1.2 # not sure a power law is the best solution here
    return (coverage_score*weight_c + length_score*weight_l + overlap_score*weight_o)*pow(base_n,-number)

def transform_coverage(coverage):
    p_c = 0.3
    scale_c = genlogistic.ppf(0.99,p_c)-genlogistic.ppf(0.01,p_c)
    loc_c = genlogistic.ppf(0.01,p_c)
    coverage_score = genlogistic.cdf(coverage*scale_c,p_c,-loc_c)
    return coverage_score

def transform_length(length):
    p_l = 0.3
    scale_l = genlogistic.ppf(0.99,p_l)-genlogistic.ppf(0.01,p_l)
    loc_l = genlogistic.ppf(0.01,p_l)
    length_score = 1 - genlogistic.cdf((1-length/15)*scale_l,p_l,-loc_l)
    return length_score


#########################
## Auxiliary functions
#########################
def alpha_only_string(list_strings):
    new_list = []
    for s in list_strings:
        s = re.sub(r'\W+', '', s)
        new_list.append(s.lower())
    return new_list

def set_from_iterables(iter_objects):
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

def is_subset(small_set, big_set):
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




#####################
## data preparation
#####################
def get_stats(the_list):
    # the_list = [ [file,[1/0,tot]], ...]
    sig_eff = 0
    false_pos = 0
    for _,(hit,shots) in the_list:
        sig_eff += hit
        if hit:
            false_pos += shots - 1
        else:
            false_pos += shots
    recall = sig_eff / float(len(the_list))
    precision = float(sig_eff) / (sig_eff + false_pos)
    return recall,precision

def split_results(res):
    res_copy = res.copy()
    end_res = []
    while res_copy:
        curr = res_copy.pop(0)
        this_res = [1,curr[1]]
        idx = 0
        while idx < len(res_copy):
            comp = res_copy.pop(0)
            if comp[1] == curr[1]:
                this_res[0] += 1
            else:
                res_copy.append(comp)
            idx += 1
        end_res.append(this_res)
    return end_res

def build_probes(param_point,raw_data):
    # prepare probes
    all_probes = [probe_tokenization(raw_data[0]['Data'][i]['document_answers'][0]['text'],spacy_model) 
                  for i in range(len(raw_data[0]['Data'])) 
                  if 'trainingset' in raw_data[0]['Data'][i]['document_name']]

    min_match_len = param_point[0]
    coverage_ratio_thres = param_point[1]
    lc_ratio = param_point[2]

    print('probe parameters: min_len={}; min_coverage={}; length_to_coverage_weight_ratio={}'
          .format(min_match_len,coverage_ratio_thres,lc_ratio))

    all_matches = get_all_matches_from_list(all_probes,min_match_len)
    probe_set = ProbeBuilder(all_matches, all_probes, coverage_ratio_thres)
    if not len(probe_set.combos):
        return None
    final = calc_scores(final_probes=probe_set.combos, all_strings=all_probes, l_c_ratio=lc_ratio)
    ordered_indices = list(zip(*sorted([(v,i) for i,v in enumerate(final['final'])], reverse=True)))[1]
        
    tokenized_probes = final['match_list'][ordered_indices[0]]

    print('PROBE LIST:')
    for prb in tokenized_probes:
        print(prb)

    return tokenized_probes


def main():
    with open('Revolving_Loan_cheatsheet.json') as jsonfile:
        raw_data = json.load(jsonfile)

    docs_in_order = [raw_data[0]['Data'][i]['document_name'] for i in range(len(raw_data[0]['Data']))]
    min_lens = [4,5,6,7]
    min_covs = [0.6,0.7,0.8]
    l_c_ratios = [1.5,2.3]
    
    results = []
    for min_l,min_c,r in product(min_lens,min_covs,l_c_ratios):
        param_point = (min_l,min_c,r)
        alp_results = []
        tokenised_probes = build_probes(param_point,raw_data)
        if tokenised_probes:
            for doc_i in range(len(docs_in_order)):
                with open('../ING_POC/{}'.format(docs_in_order[doc_i])) as fin:
                    raw_text = fin.read()
                spacied_doc = spacy_model(raw_text)
                tokenised_doc_indices = [tk.i for tk in spacied_doc if not (tk.is_stop or tk.is_punct or tk.is_space)]
                tokenised_doc = [spacied_doc[i].lemma_ for i in tokenised_doc_indices]

                x,y = list(zip(*max_probe_signal(tokenised_probes,tokenised_doc)))

                # Get true answer limits in the reduced token list (tokenised_doc)
                ans_start = raw_data[1]['Data'][doc_i]['document_answers'][0]['start']
                ans_end = raw_data[1]['Data'][doc_i]['document_answers'][-1]['end']
                get_start = True
                for i,tok in enumerate(tokenised_doc_indices):
                    if get_start and spacied_doc[tok].idx > ans_start:
                        start_tok = max(i-1,0)
                        get_start = False
                    if (not get_start) and spacied_doc[tok].idx + len(spacied_doc[tok].text_with_ws) >= ans_end:
                        end_tok = i
                        break

                # Calculate success by noting if there is a peak in the true answer range
                # and by counting how many peaks there are overall 
                # (peaks here mean only the peaks with maximum value; there are other peaks we ignore)
                top_hits = []
                top_val = 0
                for i,(yi,xi) in enumerate(sorted(zip(y,x), reverse=True)):
                    if not i:
                        top_val = yi
                        top_hits.append(xi)
                    elif yi == top_val:
                        top_hits.append(xi)
                    else:
                        break
                peaks_in_ans = sum([start_tok < i < end_tok for i in top_hits])
                peaks_total = len(top_hits)
                file_result = (doc_i,(int(bool(peaks_in_ans)),peaks_total))
                alp_results.append(file_result)
        results.append((param_point,alp_results))
        with open('results.pkl','wb') as f:
            pickle.dump(results,f)
    return results
