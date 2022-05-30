import os
import sys
import json
from collections import Counter, defaultdict


class HmmModel:
    # initializer
    def __init__(self, model_file_name = "hmmmodel.txt", training_root = "/path/to/input"):
        self.root = root
        self.training_data = []
        self.low_freq_words = set()
        self.model_file_name = model_file_name
        self.transition = defaultdict(lambda: defaultdict(int))
        self.emission = defaultdict(lambda: defaultdict(int))


    # load the training data
    def read_training_data(self):
        with open(self.root, 'r') as rf:
            for line in rf:
                line = line.rstrip()
                self.training_data.append(line.split(' '))


    # find low requency word in training set
    def find_low_freq_word(self, threshold = 1):
        word_counter = Counter()
        for token_sentence in self.training_data: # fetching each tokenizes sentence and processing tag/word
            for tagged_word in token_sentence:
                word, tag = tagged_word.rsplit('/',1)
                word_counter.update([word])
        
        for word, count in word_counter.most_common()[::-1]:
            if count > threshold:
                break
            self.low_freq_words.add(word)


    # create model: learn transition(tag|prev_tag) and emission(word|tag) probability 
    def create_model(self):
        for token_sentence in self.training_data:
            prev_tag = 'start_sentence'
            for tagged_word in token_sentence:
                word, tag = tagged_word.rsplit('/',1)

                if word.isdigit():
                    self.emission[tag]['<nmbr>'] += 1
                if word in self.low_freq_words:
                    self.emission[tag]['<unk>'] +=1

                self.emission[tag][word] += 1
                self.transition[prev_tag][tag] += 1
                prev_tag = tag

            tag = 'end_sentence'
            self.transition[prev_tag][tag] += 1

        # laplace smoothing for unseen transitions(tag|prev_tag) in train data
        self.transition['end_sentence'] = {}
        for prev_tag in self.transition:
            for tag in self.transition:
                self.transition[prev_tag][tag] = self.transition[prev_tag].get(tag,0) + 1

    # save the model
    def save_model(self):
        data = []
        data.append(self.emission)
        data.append(self.transition)
        with open(self.model_file_name, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    # locate training directory
    argument = sys.argv
    if len(argument) >= 2:
        root = sys.argv[1]
    else:
        root = "/path/to/input"
    root = "/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_02/hmm-training-data/it_isdt_train_tagged.txt"
    
    #build the model
    hmmModel = HmmModel("hmmmodel.txt", root)
    hmmModel.read_training_data()
    hmmModel.find_low_freq_word()
    hmmModel.create_model()
    hmmModel.save_model()