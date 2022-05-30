import os
import sys


class ReadWrite:

    def __init__(self, root):
        self.root = root
        self.train_positive = []
        self.train_negative = []
        self.train_truthful = []
        self.train_deceptive = []
        self.reviews = []
        self.paths = []

    def load_train_data(self):
        # print(self.root)
        for (root,dirs,files) in os.walk(self.root, topdown=True):
            if len(dirs)==0 and any(fname.endswith('.txt') for fname in files):
                path = root.split("/")
                c1 = path[-3].split("_")[0]
                c2 = path[-2].split("_")[0]
                for file in files:
                    with open(f"{root}/{file}",'r') as f:
                        lines = f.readlines()
                        document = str(lines)
                        if(c1 == "positive"):
                            self.train_positive.append(document)
                        elif(c1 == "negative"):
                            self.train_negative.append(document)
                        
                        if(c2 == "truthful"):
                            self.train_truthful.append(document)
                        elif(c2 == "deceptive"):
                            self.train_deceptive.append(document)
        
        # print(len(self.train_positive))
        # print(len(self.train_negative))
        # print(len(self.train_truthful))
        # print(len(self.train_deceptive))
        return self.train_positive, self.train_negative, self.train_truthful, self.train_deceptive

    def load_test_data(self):
        # print(self.root)
        for (root,dirs,files) in os.walk(self.root, topdown=True):
            if len(dirs)==0 and any(fname.endswith('.txt') for fname in files):
                for file in files:
                    path = ""
                    document = ""
                    with open(f"{root}/{file}",'r') as f:
                        path = f"{root}/{file}"
                        lines = f.readlines()
                        document = str(lines)
                    self.reviews.append(document)
                    self.paths.append(path)
        
        # print(len(self.reviews))
        # print(len(self.paths))
        return self.reviews, self.paths
if __name__ == "__main__":
    argument = sys.argv
    if len(argument) >= 2:
        root = sys.argv[1]
    else:
        root = "/path/to/input"
    # rw_train = ReadWrite(root)
    rw_train = ReadWrite("/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/train")
    train_positive, train_negative, train_truthful, train_deceptive = rw_train.load_train_data()
    # print(len(train_positive))

    # rw_test = ReadWrite(root)
    rw_test = ReadWrite("/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/test")
    reviews, paths = rw_test.load_test_data()
    print(len(reviews))
    print(reviews[0])
    print(len(paths))
    print(paths[0])