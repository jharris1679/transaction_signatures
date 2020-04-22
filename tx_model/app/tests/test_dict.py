import pickle
import numpy

with open('datasets/merchant_seqs_by_tx_32/dictionary', 'rb') as f:
    dict = pickle.load(f)

print(dict.items())

print(len(dict.idx2token))
print(len(dict.idx2user))
print(len(dict.idx2cat))
