import io
import sys
import pickle

with open(sys.argv[1], 'rb') as f:
    b_array = f.read()

f = io.BytesIO(b_array.replace(b'.label', b'', 1))
__, lb = pickle.load(f)

print('\n'.join(l for l in lb.classes_))
