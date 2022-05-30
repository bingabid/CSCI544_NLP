import re
import numpy as np
from pickle import dump
from collections import Counter

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", 
                "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", 
                "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
                "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", 
                "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", 
                "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", 
                "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
                "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
            ]

pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`] '''

class Naive_Bayes():
    def __init__(self, classes, train_positive = [], train_negative = [], train_truthful = [], train_deceptive = []) -> None:
        self.classes = classes
        self.train_positive = train_positive
        self.train_negative = train_negative
        self.train_truthful = train_truthful
        self.train_deceptive = train_deceptive
        self.logprior = [None] * len(self.classes)
        self.loglikelihood = [None] * len(self.classes)

    def test_data_preprocessing(self, reviews):
        self.reviews = []
        for review in reviews:
            text = review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[b","").replace("[","").replace("]","").replace("*", "").replace("(", "").replace("\"", "").replace(")", "").replace("/", "").replace("\\","")
            text = text.replace("'","")
            words = text.split()
            clean_words = [word for word in words if word not in stop_words]
            self.reviews.append(' '.join(clean_words))
        
        return self.reviews

    def data_preprocessing(self):

        #remove punctuation and apply stemming
        train_positive = []
        for review in self.train_positive:
            # text = review.replace('\\n','').replace('\"','').replace('\'','').replace('(',' ').replace(')','')
            text = review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[b","").replace("[","").replace("]","").replace("*", "").replace("(", "").replace("\"", "").replace(")", "").replace("/", "").replace("\\","")
            text = text.replace("'","")
            # text = re.sub(r"[,.;@#?!&$'\[\]]+\ *", " ", text, flags=re.VERBOSE)
            # stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in text.split()]
            # text = ' '.join(stemmed)
            train_positive.append(text)
        self.train_positive = train_positive

        train_negative = []
        for review in self.train_negative:
            text = review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[b","").replace("[","").replace("]","").replace("*", "").replace("(", "").replace("\"", "").replace(")", "").replace("/", "").replace("\\","")
            text = text.replace("'","")
            # text = re.sub(r"[,.;@#?!&$'\[\]]+\ *", " ",review.replace('\\n',''), flags=re.VERBOSE)
            # stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in text.split()]
            # text =  ' '.join(stemmed)
            train_negative.append(text.lower())
        self.train_negative = train_negative
        
        train_truthful = []
        for review in self.train_truthful:
            text = review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[b","").replace("[","").replace("]","").replace("*", "").replace("(", "").replace("\"", "").replace(")", "").replace("/", "").replace("\\","")
            text = text.replace("'","")
            # text = re.sub(r"[,.;@#?!&$'\[\]]+\ *", " ",review.replace('\\n',''), flags=re.VERBOSE)
            # stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in text.split()]
            # text = ' '.join(stemmed)
            train_truthful.append(text.lower())
        self.train_truthful = train_truthful
        
        train_deceptive = []
        for review in self.train_deceptive:
            text = review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[b","").replace("[","").replace("]","").replace("*", "").replace("(", "").replace("\"", "").replace(")", "").replace("/", "").replace("\\","")
            text = text.replace("'","")
            # text = re.sub(r"[,.;@#?!&$'\[\]]+\ *", " ",review.replace('\\n',''), flags=re.VERBOSE)
            # stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in text.split()]
            # text = ' '.join(stemmed)
            train_deceptive.append(text.lower())
        self.train_deceptive = train_deceptive


        # remove stop words
        train_positive = []
        for review in self.train_positive:
            words = review.split()
            clean_words = [word for word in words if word not in stop_words]
            train_positive.append(' '.join(clean_words))
        self.train_positive = train_positive

        train_negative = []
        for review in self.train_negative:
            words = review.split()
            clean_words = [word for word in words if word not in stop_words]
            train_negative.append(' '.join(clean_words))
        self.train_negative = train_negative

        train_truthful = []
        for review in self.train_truthful:
            words = review.split()
            clean_words = [word for word in words if word not in stop_words]
            train_truthful.append(' '.join(clean_words))
        self.train_truthful = train_truthful

        train_deceptive = []
        for review in self.train_deceptive:
            words = review.split()
            clean_words = [word for word in words if word not in stop_words]
            train_deceptive.append(' '.join(clean_words))
        self.train_deceptive = train_deceptive

        # print(len(self.train_positive))
        # print(len(self.train_negative))
        # print(len(self.train_truthful))
        # print(len(self.train_deceptive))


    def train_model(self):
        #Initialize n_c[ci]: number of documents of class i
        positive_doc = len(self.train_positive)
        negative_doc = len(self.train_negative)
        truthful_doc = len(self.train_truthful)
        deceptive_doc = len(self.train_deceptive)
        
        total_docs_pos_neg = positive_doc + negative_doc
        total_docs_truthful_deceptive = truthful_doc + deceptive_doc

        #compute log prior
        self.logprior[0] = np.log(positive_doc / total_docs_pos_neg)
        self.logprior[1] = np.log(negative_doc / total_docs_pos_neg)
        self.logprior[2] = np.log(truthful_doc / total_docs_truthful_deceptive)
        self.logprior[3] = np.log(deceptive_doc / total_docs_truthful_deceptive)

        #Creates a vocabulary list. For large datasets, this code becomes slow.
        self.vocab = set()
        for review in self.train_positive:
            words = review.split()
            for word in review.split():
                if word not in self.vocab:
                    self.vocab.add(word)

        for review in self.train_negative:
            words = review.split()
            for word in review.split():
                if word not in self.vocab:
                    self.vocab.add(word)

        for review in self.train_truthful:
            words = review.split()
            for word in review.split():
                if word not in self.vocab:
                    self.vocab.add(word)
        
        for review in self.train_deceptive:
            words = review.split()
            for word in review.split():
                if word not in self.vocab:
                    self.vocab.add(word)
    
        self.vocab_size = len(self.vocab)
        # print(self.vocab_size)

        #computer conditional prob
        def computeConditionalProb(idx, reviews):
            counter = Counter()
            word_count = 0
            for review in reviews:
                words = review.split()
                word_count += len(words)
                counter += Counter(words)
            denom = word_count + self.vocab_size
            dic = {}
            for word in self.vocab:
                numer = counter[word] + 1
                dic[word] = np.log(numer/denom)
            self.loglikelihood[idx] = dic

        computeConditionalProb(0, self.train_positive)
        computeConditionalProb(1, self.train_negative)
        computeConditionalProb(2, self.train_truthful)
        computeConditionalProb(3, self.train_deceptive)
        # print(self.loglikelihood[0])


    #save the model
    def create_model(self):
        with open('nbmodel.txt', 'w') as f:
            for item in self.logprior:
                f.write("%s\n" % item)
            idx = 0
            for item in self.loglikelihood:
                f.write("%s\n" % self.classes[idx])
                idx += 1
                # print(len(item))
                for key, value in item.items(): 
                    f.write('%s:%s\n' % (key, value))
        
        with open('vocabulary.pkl','wb') as f:
            dump(self.vocab,f)