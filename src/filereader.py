import sys
import os
#modified the code further for reference
rootpath = "/test data path/"
for dir in os.listdir(rootpath):
    if not dir.startswith('.'):
        print(dir)
        os.system(" ano_test.py " + rootpath + " " + dir)
