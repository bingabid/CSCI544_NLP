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

    rw = ReadWrite("/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/test/")
    train_positive, train_negative, train_truthful, train_deceptive = rw.load_train_data()
    # print(train_positive[0])
    # print(len(train_positive))

    classes = ['positive','negative','deceptive','truthful']

    naive_bayes = Naive_Bayes(classes,train_positive, train_negative, train_truthful, train_deceptive)
    naive_bayes.data_preprocessing()

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
    
    
    classification_pos_neg = [] * 2
    def classification_positive_negative(reviews):
        result = []
        logpost = [0] * 2
        sumloglikelihoods = [0] * 2
        for review in reviews:
            for word in review.split():
                if word in vocab:
                    sumloglikelihoods[0] += loglikelihood[0][word]
                    sumloglikelihoods[1] += loglikelihood[1][word]

            logpost[0] = logprior[0] + sumloglikelihoods[0]
            logpost[1] = logprior[1] + sumloglikelihoods[1]
            # print(logpost)
            result.append(logpost.index(max(logpost)))
        classification_pos_neg.append(result)

    # print(len(naive_bayes.train_positive))
    # print(naive_bayes.train_positive[0])
    # print(len(naive_bayes.train_negative))
    classification_positive_negative(naive_bayes.train_positive)
    classification_positive_negative(naive_bayes.train_negative)

    correct = 0
    for i in range(2):
        # print(len(classification[i]))
        correct += classification_pos_neg[i].count(i)
        print(classification_pos_neg[i].count(i))

    accuracy = correct/320
    print(accuracy)


    classification_truth_decept = [] * 2
    def classification_truthful_deceptive(reviews):
        result = []
        logpost = [0] * 2
        sumloglikelihoods = [0] * 2
        for review in reviews:
            for word in review.split():
                if word in vocab:
                    sumloglikelihoods[0] += loglikelihood[2][word]
                    sumloglikelihoods[1] += loglikelihood[3][word]

            logpost[0] = logprior[2] + sumloglikelihoods[0]
            logpost[1] = logprior[3] + sumloglikelihoods[1]
            # print(logpost)
            result.append(logpost.index(max(logpost)))
        classification_truth_decept.append(result)

    # print(len(naive_bayes.train_truthful))
    # print(naive_bayes.train_truthful[0])
    # print(len(naive_bayes.train_deceptive))
    classification_truthful_deceptive(naive_bayes.train_truthful)
    classification_truthful_deceptive(naive_bayes.train_deceptive)
    correct = 0
    for i in range(2):
        # print(len(classification[i]))
        correct += classification_truth_decept[i].count(i)
        print(classification_truth_decept[i].count(i))

    accuracy = correct/320
    print(accuracy)