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
    # rw = ReadWrite(root)
    rw = ReadWrite("/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/train")
    train_positive, train_negative, train_truthful, train_deceptive = rw.load_train_data()
    # print(train_positive[0])
    # print(len(train_positive))

    classes = ["positive","negative","truthful","deceptive"]

    naive_bayes = Naive_Bayes(classes, train_positive, train_negative, train_truthful, train_deceptive)
    naive_bayes.data_preprocessing()
    naive_bayes.train_model()
    naive_bayes.create_model()