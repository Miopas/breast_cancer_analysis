from sklearn.metrics import cohen_kappa_score
import pandas as pd
import sys

def load_data(infile):
    df = pd.read_csv(infile)
    text2label = {}
    for text, label in zip(df['text'], df['class']):
        text2label[text] = label
    return text2label

file_1 = load_data(sys.argv[1])
file_2 = load_data(sys.argv[2])

assert len(file_1) == len(file_2)

print('data size:', len(file_1))
file_1 = sorted(file_1.items())
file_2 = sorted(file_2.items())
f1_vec = []
f2_vec = []
for (k1, v1), (k2, v2) in zip(file_1, file_2):
    #assert k1 == k2
    f1_vec.append(v1)
    f2_vec.append(v2)

print(cohen_kappa_score(f1_vec, f2_vec))
