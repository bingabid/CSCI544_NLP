import sys
import numpy as np
from pickle import load
from itertools import count
from readWrite import ReadWrite
from naive_bayes import Naive_Bayes

if __name__=='__main__':
    argument = sys.argv
    if len(argument) >= 2:
        root = sys.argv[1]
    else:
        root = "/path/to/input"

    rw_test= ReadWrite(root)
    # rw_test = ReadWrite("/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/test/")
    reviews, paths = rw_test.load_test_data()
    # print(train_positive[0])
    # print(len(train_positive))

    classes = ['positive','negative','truthful','deceptive']

    naive_bayes = Naive_Bayes(classes)
    processed_reviews = naive_bayes.test_data_preprocessing(reviews)
    # print(len(processed_reviews))

    logprior = [None] * len(classes)
    loglikelihood = [None] * len(classes)

    with open('nbmodel.txt','r') as file:
        model = file.read().splitlines()

    idx = 0 
    for idx in range(len(classes)):
        logprior[idx] = float(model[idx])

    idx += 2
    class_type = 0
    dic = {}
    while idx < len(model):
        words = model[idx].split(':')
        if len(words) == 1:
            # print(class_type)
            # print(words)
            # print(len(dic))
            loglikelihood[class_type] = dic
            dic = {}
            class_type += 1
        else:
            dic[words[0]] = float(words[1])
        idx += 1
    loglikelihood[class_type] = dic

    with open('vocabulary.pkl','rb') as f:
        vocab = load(f)


    #classification into positive/negative:
    def classification_positive_negative(review):
        logpost = [0] * 2
        sumloglikelihoods = [0] * 2
        for word in review.split():
            if word in vocab:
                sumloglikelihoods[0] += loglikelihood[0][word]
                sumloglikelihoods[1] += loglikelihood[1][word]

        logpost[0] = logprior[0] + sumloglikelihoods[0]
        logpost[1] = logprior[1] + sumloglikelihoods[1]
        if logpost[0] > logpost[1]:
            return classes[0]
        else:
            return classes[1]

    
    #classification into trutful/deceptive:
    def classification_truthful_deceptive(review):
        logpost = [0] * 2
        sumloglikelihoods = [0] * 2
        for word in review.split():
            if word in vocab:
                sumloglikelihoods[0] += loglikelihood[2][word]
                sumloglikelihoods[1] += loglikelihood[3][word]

        logpost[0] = logprior[2] + sumloglikelihoods[0]
        logpost[1] = logprior[3] + sumloglikelihoods[1]
        if logpost[0] > logpost[1]:
            return classes[2]
        else:
            return classes[3]



    with open("nboutput.txt", "w") as wf:
        for i in range(len(reviews)):
            review = processed_reviews[i]
            path = paths[i]
            pos_neg = classification_positive_negative(review)
            truth_decept = classification_truthful_deceptive(review)
            res = truth_decept + " " + pos_neg + " " + path + "\n"
            wf.write(res)