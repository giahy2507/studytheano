import cPickle
import os

if os.path.isfile("hynguyen"):
    print "co file"
else:
    print "ko co file"
with open("hynguyen",mode="rb") as f:
    data = cPickle.load(f)
    print data