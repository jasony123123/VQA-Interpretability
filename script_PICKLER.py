import sys
import pickle

fnamein = sys.argv[1]
fnameout = sys.argv[2]

with open(fnamein, 'rb') as handle:
    data = pickle.load(handle)

with open(fnameout, 'wb') as handle:
    pickle.dump(data, handle)
