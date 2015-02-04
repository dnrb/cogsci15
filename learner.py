import csv, re
import os, errno
import time, datetime
import random
from subprocess import call
from collections import defaultdict as dd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
#
import parameters
# parameters is the parameters.py file where you can specify the model's parameters
# such as which languages it uses to derive the semantic space and where the files
# for these languages are

class learner(object):

    def __init__(self):
        """
        Initializer function for a learner: sets the parameters
        and opens a folder for the experimental results, pastes the
        current parameter file in it as well.
        """
        #
        t = time.time()
        ts = datetime.datetime.fromtimestamp(t).strftime('%Y_%m_%d_%H_%M_%S')
        self.out_dir = 'experiment_%s' % ts
        try: os.makedirs(self.out_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self.out_dir):
                pass
            else: raise
        call(['cp', 'parameters.py', '%s/parameters.py' % self.out_dir])
        

    def initialize_word_frequencies(self):
        """
        Sets the word frequencies to 1.0 each if the 'use frequency'
        parameters is not set to 'corpus', otherwise reads
        the corpus frequencies
        """
        if parameters.use_frequency == 'corpus':
            self.word_frequencies = self.read_word_frequencies()
        else: self.word_frequencies = { w : 1.0 for w in
                                        self.words[parameters.target_language] }

    def read_word_frequencies(self):
        """
        Reads the word frequencies for the target language from a file
        called <TARGET LANGUAGE>_frequencies.csv
        The make-up of that file should be:
        <WORD>,<COUNT>
        <WORD>,<COUNT>
        ...
        <WORD>,<COUNT>
        (see dutch_frequencies.csv for an example)
        All words used in the <TARGET LANGUAGE>.csv file should be given
        in the frequencies file
        """
        frequencies = {}
        location = '%s/%s_frequencies.csv' % (parameters.file_path,
                                              parameters.target_language)
        handle = csv.reader(open(location), quotechar = '|')
        for line in handle:
            frequencies[line[0]] = int(line[1])
        return frequencies

    def read_source_data(self):
        """
        Reads source data for all languages from files called <LANGUAGE>.csv.
        This file should be structured as follows:
        <SITUATION>,<WORD>
        <SITUATION>,<WORD>
        ...
        <SITUATION>,<WORD>
        where the situation should be an integer, and all
        situations should range from 1 to n, not leaving out numbers,
        and the word should be a string
        """
        self.data = dd(list)
        self.words = dd(lambda : list())
        #
        for language in parameters.languages:
            location = "%s/%s.csv" % (parameters.file_path, language)
            handle = csv.reader(open(location), quotechar = '|')
            words_per_situation = dd(lambda : dd(int))
            words_l = set()
            for row in handle:
                situation = int(row[0])-1
                word = re.sub('[^a-zA-Z\-]', '', row[1])
                if word not in self.words[language]:
                    self.words[language].append(word)
                words_per_situation[situation][word] += 1
            self.data[language] = [ [ words_per_situation[situation][marker]
                                      for marker in self.words[language] ]
                                    for situation in
                                    range(parameters.n_situations) ]
        #
        return

    def set_modal_markers(self):
        tl = parameters.target_language
        tl_data = self.data[tl]
        xrs = xrange(parameters.n_situations)
        self.actual_Ps= [ [ float(count) / sum(tl_data[situation])
                                        for count in tl_data[situation] ]
                                      for situation in xrs ]
        _modal_markers = [ [ marker for marker in range(len(self.words[tl]))
                             if (self.actual_Ps[situation][marker] ==
                                 max(self.actual_Ps[situation])) ]
                           for situation in xrs ]
        self.observed_modals = [ _modal_markers[situation][0]
                                 if len(_modal_markers[situation]) == 1 
                                 else None for situation in xrs ]
        return

    def initialize_semantic_space(self):
        """
        Does dimensionality reduction with PCA and initializes the resulting
        semantic space
        """
        #
        data = np.array([[d for l in parameters.languages
                          for d in self.data[l][s]]
                         for s in range(parameters.n_situations)])
        pca = PCA()
        self.semantic_space = pca.fit_transform(data)
        return

    """
    GENERAL
    """
    def run_experiment(self):
        #
        self.start_statistic_dump()
        #
        experimental_space = [ x[:parameters.n_components]
                               for x in self.semantic_space ]
        #
        n_batch, n_iter = parameters.batch_size, parameters.length_simulation
        #
        # all space in self.space upto the specified number of components
        for simulation in xrange(parameters.n_simulations):
            sampled_space = [ self.sample(experimental_space)
                             for i in xrange(parameters.length_simulation) ]
            word_train, sem_train, sit_train = zip(*sampled_space)
            for batch in range(n_batch, n_iter + n_batch, n_batch):
                self.exp_parameters = [parameters.n_components,simulation,batch]
                self.test( sem_train[:batch], word_train[:batch],
                           sit_train[:batch], experimental_space )
        #
        self.fout.close()

        

    def test(self, sem_train, word_train, sit_train, sit_space):
        """
        Runs a leave-one-situation-out test procedure, as described in
        Beekhuizen, Fazly & Stevenson (2014)
        """
        cls = GaussianNB()
        xrt = xrange(self.exp_parameters[2])
        xrw = xrange(len(self.words[parameters.target_language]))
        xrs = xrange(parameters.n_situations)
        accuracy = 0.0
        for situation in xrs:
            sem_train_s = [sem_train[x] for x in xrt
                           if sit_train[x] != situation]
            word_train_s = [word_train[x] for x in xrt
                            if sit_train[x] != situation]
            counts = [len([d for d in word_train_s if d == word])
                      for word in xrw ]
            #
            cls.fit(sem_train_s, word_train_s)
            #
            _predicted_Ps = list(cls.predict_proba([sit_space[situation]])[0])
            predicted_Ps = [ _predicted_Ps.pop(0) if word in word_train_s 
                             else 0.0 for word in xrw ]
            predicted_modals = [ word for word in xrw
                                 if predicted_Ps[word] == max(predicted_Ps)]
            predicted_modal = (predicted_modals[0]
                               if len(predicted_modals) == 1 else None)
            self.write_statistic_dump(counts, self.actual_Ps[situation], 
                                      predicted_Ps,
                                      self.observed_modals[situation],
                                      predicted_modal, situation)
            #
            accuracy += (1 if predicted_modal == self.observed_modals[situation]
                         else 0)
            # optional - for screen output
        print self.exp_parameters, accuracy/parameters.n_situations
        return

    """
    SAMPLING
    """
    
    def sample(self, data):
        """
        Samples on the basis of the probability of the situation given the word
        times the probability of the word. Returns a word (integer), a datapoint
        (array of values) and a situation (integer)
        """
        words = self.words[parameters.target_language]
        total_frequency = float(sum(self.word_frequencies.values()))
        word_Ps = np.array([self.word_frequencies[word]/total_frequency
                            for word in words])
        word_index = len(words)
        while word_index >= len(words):
            word_index = word_Ps.cumsum().searchsorted(np.random.sample(1))[0]
        sit_Ps = [self.data[parameters.target_language][s][word_index]
                  for s in range(parameters.n_situations)]
        sit_Ps = np.array([d/float(sum(sit_Ps)) for d in sit_Ps])
        situation = sit_Ps.cumsum().searchsorted(np.random.sample(1))[0]
        return [word_index, data[situation], situation]

    """
    PRINTING FUNCTIONS
    """
    def write_component_space(self):
        out = open('%s/components.txt' % self.out_dir, 'wb')
        for sit, modal, vals in zip(range(parameters.n_situations),
                                    self.observed_modals, self.semantic_space):
            word = self.words[parameters.target_language][modal]
            valString = ','.join(['%.5f' % v for v in vals])
            out.write('%d,%s,%s\n' % (sit+1, word, valString))
        out.close()
                                    
    def start_statistic_dump(self):
        """
        The result files contain on the columns the number of components used,
        the simulation, the iteration, the situation, one column for the
        observed probability given the situation of each marker in the target
        language, one column for the predicted probability given the situation
        of each marker in the target language, and one column for each marker
        giving the number of times that marker has been seen in the training
        data.
        """
        name_out = ('%s/results_%s_%s.csv' %
                    (self.out_dir, parameters.use_frequency, parameters.loocv))
        lg = parameters.target_language
        self.fout = open(name_out, 'wb')
        self.fout.write("'ncomp','simulation','iteration','situation',")
        self.fout.write("%s," % (','.join(['%sAct' % word
                                           for word in self.words[lg]])))
        self.fout.write("%s,"%(','.join(["'%sPred'" % word
                                         for word in self.words[lg]])))
        self.fout.write("%s,"%(','.join(["'%sCount'" % word
                                         for word in self.words[lg]])))
        self.fout.write("'actModal','predModal','modalCorrect'\n")
        return
    
    def write_statistic_dump(self,count,act_p,pred_p,act_m,pred_m,situation):
        act_p_string = ','.join(['%.3f' % d for d in act_p])
        pred_p_string = ','.join(['%.3f' % d for d in pred_p])
        count_string = ','.join(['%d' % d for d in count])
        exp_string = ','.join(['%d' % d for d in self.exp_parameters])
        words = self.words[parameters.target_language]
        self.fout.write('%s,%d,%s,%s,%s,%s,%s,%d\n' %
                        (exp_string, situation, act_p_string,
                         pred_p_string, count_string,
                         words[act_m], words[pred_m],
                         1 if act_m == pred_m else 0))
        return
            
    
def main():
    l = learner()
    l.read_source_data()
    l.initialize_word_frequencies()
    l.set_modal_markers()
    l.initialize_semantic_space()
    l.run_experiment()
    l.write_component_space()
    return
    
if __name__ == "__main__":
    main()
