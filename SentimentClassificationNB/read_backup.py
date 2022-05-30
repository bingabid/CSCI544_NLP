


def load_train_data(root_directory):
    print(root_directory)
    positive, negative, truthful, deceptive = [], [], [], []
    for (root,dirs,files) in os.walk(root_directory, topdown=True):
        if len(dirs)==0 and any(fname.endswith('.txt') for fname in files):
            path = root.split("/")
            c1 = path[-3].split("_")[0]
            c2 = path[-2].split("_")[0]
            for file in files:
                with open(f"{root}/{file}",'r') as f:
                    lines = f.readlines()
                    document = str(lines)
                    if(c1 == 'positive'):
                        positive.append(document)
                    else:
                        negative.append(document)
                    
                    if(c2 == "truthful"):
                        truthful.append(document)
                    else:
                        deceptive.append(document)
    return positive, negative, truthful, deceptive


# if __name__ == "__main__":
#     test_directory = '/Users/abid/Desktop/2ndSem/CSCI544_NLP/coding_01/'
#     positive, negative, truthful, deceptive = load_train_data(test_directory)
#     print(len(positive))
#     print(len(negative))
#     print(len(truthful))
#     print(len(deceptive))
    
