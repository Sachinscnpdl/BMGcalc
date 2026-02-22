# save as find_nonascii.py
import sys, unicodedata
fname = sys.argv[1]
with open(fname, 'rb') as f:
    data = f.read()
text = data.decode('utf-8')
for i, ch in enumerate(text):
    if ord(ch) > 127:
        print(i, hex(ord(ch)), repr(ch), unicodedata.name(ch, 'UNKNOWN'))
