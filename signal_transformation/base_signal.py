import numpy as np
import matplotlib.pyplot as plt
import json

class BaseSignalCalculator:
    def __init__(self, filepath, doc_id, spacy_model, cheatsheet):
        """Assume a fixed format for the cheatsheet (WHAT IS IT)"""
        with open(cheatsheet) as jsonfile:
            # TODO Is it necessary to keep the raw data?
            #      Can we decide upon a processed data format?
            self.raw_data = json.load(jsonfile)
        self.filename = filepath.split('/')[-1]
#        self.doc_id = self.get_doc_id_from_name(self.filename, self.raw_data)
        with open(filepath) as fin:
            raw_text = fin.read()
        self.spacy_doc = spacy_model(raw_text)
        self.tokenised_doc_indices = [tk.i for tk in self.spacy_doc 
                                      if not (tk.is_stop 
                                              or tk.is_punct 
                                              or tk.is_space)]

        # Each container has two nested containers - one for the remaining 
        # tokens after normalisation [0] and one for the raw tokens [1].
        self.set_true_span(
            spacy_doc=self.spacy_doc,
            token_ids=self.tokenised_doc_indices,
            doc_i=doc_id,
            raw_data=self.raw_data,
        )
        # TODO how do I initialise these better? maybe overwrite in derived class
        self.signal = []
        # TODO Do we need two spans (normalised and raw) or just one would do?
        self.spans = []

#    def get_doc_id_from_name(self, filename, raw_data):
#        pass 
        


#   TODO This assumes there is one span (two limits) but actually there may be
#        a list of boundaries, so maybe this function as well as the functions that
#        determine the span should be more general
    def set_true_span(self, spacy_doc, token_ids, doc_i, raw_data):
        ans_start = raw_data[1]['Data'][doc_i]['document_answers'][0]['start']
        ans_end = raw_data[1]['Data'][doc_i]['document_answers'][-1]['end']
        get_start = True
        for i,tok in enumerate(token_ids):
            if get_start and spacy_doc[tok].idx > ans_start:
                start_tok = max(i-1,0)
                get_start = False
            curr_last_char = spacy_doc[tok].idx + len(spacy_doc[tok].text_with_ws)
            if (not get_start) and curr_last_char >= ans_end:
                end_tok = i
                break
        
        get_start = True
        start_raw = -1
        end_raw = -1
        for i,tok in enumerate(spacy_doc):
            if get_start and tok.idx <= ans_start < tok.idx + len(tok.text_with_ws):
                get_start = False
                start_raw = i
            if (not get_start) and tok.idx > ans_end:
                end_raw = i
                break
        if end_raw == -1:
            end_raw = len(spacy_doc)

        self.true_span = ((start_tok, end_tok), (start_raw, end_raw))
       
###################################################################
    def get_peak_limits(self, window, raw=False):
        # If two peaks are closer together than a window width - combine into one span
        # elif the distance is < 3*window - place a border at ratio 2:1
        # else - the right border of the left span is peak + 2*window 
        #        and the left border of the right span is peak - window
        if raw:
            window = int(window*len(self.spacy_doc)/float(len(self.tokenised_doc_indices)))
        # TODO CHeck that self.signal has been assigned
        signal = self.signal[int(raw)]
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
        peak_limits[xi,1] = min(top_hits[xi] + 2*window, len(self.spacy_doc)-1)
        peak_limits = peak_limits[:xi+1,:]        
        return peak_limits

    def get_both_span_lists(self):
        self.spans = [self.get_peak_limits(window=self.window, raw=False),
                      self.get_peak_limits(window=self.window, raw=True)]


    def get_raw_text_signal(self, signal):
        signal_raw_text = np.interp(range(len(self.spacy_doc)), self.tokenised_doc_indices[:len(signal)], signal)
        return signal_raw_text

###############################################################
    def plot_signal(self, ax, raw=False):
        y = self.signal[int(raw)]
        x = list(range(len(y)))
        ax.plot(x,y)

        ymax = ax.get_ylim()[1]
        start_tok, end_tok = self.true_span[int(raw)]
        ax.plot([start_tok,start_tok],[0,ymax], 
                c='#ff7f0e', zorder=0)
        ax.plot([end_tok,end_tok],[0,ymax], 
                c='#ff7f0e', zorder=0)

        for xi in range(self.spans[int(raw)].shape[0]):
            start_tok = self.spans[int(raw)][xi,0]
            end_tok = self.spans[int(raw)][xi,1]
            ax.plot([start_tok,start_tok],[0,ymax], 
                    c='#7f7f7f', ls='--', zorder=0)
            ax.plot([end_tok,end_tok],[0,ymax], 
                    c='#7f7f7f', ls='--', zorder=0)

        # TODO this line is too rigid
        text_color = 'red' if 'evaluationset' in self.filename else 'black'
        ax.text(0.6, 0.9, self.filename, 
            transform=ax.transAxes, 
            fontsize=15, 
            color=text_color,
        ) 
        return ax

    def feature_vec_from_spans(self):
        rtn = np.zeros(len(self.spacy_doc))
        for s,e,_ in self.span[1]: # [1] means raw text span        
            rtn[s:e] = 1
        return rtn 

    def text_in_spans(self, raw=True):
        rtn = 60*'-' + '\n' + self.filename + '\n'
        for s,e,txt in self.spans[int(raw)]:
            span_str = '(start={}, end={})\n\n'.format(s,e) \
                       + txt \
                       + '\n...\n\n' 
            rtn += span_str
        return rtn
        

    def build_signal(self):
        # TODO This function should be re-defined in each explicit implementation
        return np.zeros(len(self.spacy_doc))

    def build_both_signals(self):
        tmp_signal = self.build_signal()
        self.signal = [tmp_signal, self.get_raw_text_signal(tmp_signal)]
