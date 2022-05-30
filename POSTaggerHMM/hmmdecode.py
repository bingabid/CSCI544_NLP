import sys
import json
from numpy import log
from random import choice
from queue import PriorityQueue

class Viterbi:
    # initializer
    def __init__(self, model_file_name = 'hmmmodel.txt', output_file = 'hmmoutput.txt', test_root = "/path/to/input"):
        self.emission = {}
        self.test_data = []
        self.transition = {}
        self.tag_output = []
        self.root = test_root
        self.emission_probability = {}
        self.transition_probability = {}
        self.output_file = output_file
        self.model_file_name = model_file_name


    # load the test data
    def read_test_data(self):
        with open(self.root, 'r', encoding='utf8') as rf:
            for line in rf:
                line = line.rstrip()
                self.test_data.append(line.split(' '))

        
    # retrieve hmm model: transition and emission probability
    def load_model(self):
        with open(self.model_file_name, 'r') as rf:
            data = json.load(rf)
            self.emission = data[0]
            self.transition = data[1]

    # computer transition probabilities: P(tag|prev_tag)
    def transition_probabilities(self):
        for prev_tag in self.transition:
            self.transition_probability[prev_tag] = {}
            total_count = sum(self.transition[prev_tag].values())
            for tag in self.transition[prev_tag]:
                self.transition_probability[prev_tag][tag] = self.transition[prev_tag][tag]/total_count


    # computer emission probabilities: P(word|tag)
    def emission_probabilities(self):
        for tag in self.emission:
            self.emission_probability[tag] = {}
            total_count = sum(self.emission[tag].values())
            for word in self.emission[tag]:
                self.emission_probability[tag][word] = self.emission[tag][word]/total_count


    # Viterbi Algorithms: decoding using viterbi decoder
    def viterbi_decoder(self):
        for idx, test_token_sentence in enumerate(self.test_data):

            tag_prob = [] # computing prob of each tag: tag_prob[index-1][tag] * P(tag|prev_tag) * P(word|tag)
            tag_prob_back_ptr = [] # computing most likely sequence: storing maximum probability back pointer

            for index, word in enumerate(test_token_sentence):
                tag_prob.append({})
                tag_prob_back_ptr.append({})
                
                # if current word is digit, process it as number(similar to model creation)
                if word.isdigit() and index != 0:
                    for tag in self.emission:
                        tag_prob[index][tag] = -1000000.00
                        emission_probability = log(self.emission_probability[tag].get('<nmbr>',1e-9))
                        for prev_tag in tag_prob_back_ptr[index-1]:
                            transition_probability = log(self.transition_probability[prev_tag].get(tag))
                            total_joint_probability = emission_probability + transition_probability + tag_prob[index-1][prev_tag]
                            if tag_prob[index][tag] < total_joint_probability:
                                tag_prob[index][tag] = total_joint_probability
                                tag_prob_back_ptr[index][tag] = prev_tag

                # first word of sentence
                if index == 0:
                    for tag in self.emission:
                        emission_probability = log(self.emission_probability[tag].get(word,1e-9))
                        transition_probability = log(self.transition_probability['start_sentence'].get(tag))
                        tag_prob[index][tag] = emission_probability + transition_probability
                        tag_prob_back_ptr[index][tag] = 'start_sentence'
                else:
                    flag_unseen_word = 1 # flag for unseen word in training data
                    # compute probabilities for all possible tags
                    for tag in tag_prob[index-1]:
                        tag_prob[index][tag] = -1000000.00
                        emission_probability = self.emission_probability[tag].get(word,0)

                        # given tag is never seen with given word in training: P(word|tag) = 0
                        if emission_probability == 0:
                            continue
                        emission_probability = log(emission_probability)
                        flag_unseen_word = 0 # P(word|tag) != 0, hence not a unseen word with given tag

                        # compute most likely tag:  tag_prob[index-1][tag] * P(tag|prev_tag) * P(word|tag)
                        for prev_tag in tag_prob_back_ptr[index-1]:
                            transition_probability = log(self.transition_probability[prev_tag].get(tag))
                            total_joint_probability = emission_probability + transition_probability + tag_prob[index-1][prev_tag]
                            if tag_prob[index][tag] < total_joint_probability:
                                tag_prob[index][tag] = total_joint_probability
                                tag_prob_back_ptr[index][tag] = prev_tag
                    
                    # unseen word: using emission probability of unknown word
                    if flag_unseen_word:
                        for tag in self.emission:
                            emission_probability = log(self.emission_probability[tag].get('<unk>',1e-9))
                            for prev_tag in tag_prob_back_ptr[index-1]:
                                transition_probability = log(self.transition_probability[prev_tag].get(tag))
                                total_joint_probability = emission_probability + transition_probability + tag_prob[index-1][prev_tag]
                                if tag_prob[index][tag] < total_joint_probability:
                                    tag_prob[index][tag] = total_joint_probability
                                    tag_prob_back_ptr[index][tag] = prev_tag

            # end of sentence tag 
            for tag in tag_prob_back_ptr[index]:
                end_prob = log(self.transition_probability[tag].get('end_sentence'))
                tag_prob[index][tag] += end_prob

            # get the most likely tag sequence ending
            high_probability_tag = max(tag_prob[-1], key=tag_prob[-1].get)

            # assign each word the most likely tag by following the most likely tag path
            tagged_sentence = []
            for i in range(len(tag_prob)-1,-1,-1):
                word_tag = test_token_sentence[i] + '/' + high_probability_tag
                tagged_sentence.insert(0,word_tag)
                high_probability_tag = tag_prob_back_ptr[i][high_probability_tag]
            self.tag_output.append(tagged_sentence)

    # dump the tagged output
    def save_output(self):
        with open(self.output_file, 'w', encoding='utf8') as f:
            for sentence in self.tag_output:
                for word in sentence:
                    f.write('%s ' % word)
                f.write('\n')


if __name__ == '__main__':

    # locate test directory
    argument = sys.argv
    if len(argument) >= 2:
        root = sys.argv[1]
    else:
        root = "/path/to/input"
    root = "/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_02/hmm-training-data/it_isdt_dev_raw.txt"

    viterbi = Viterbi('hmmmodel.txt', 'hmmoutput.txt', root)
    viterbi.read_test_data()
    viterbi.load_model()
    viterbi.transition_probabilities()
    viterbi.emission_probabilities()
    viterbi.viterbi_decoder()
    viterbi.save_output()