#!/usr/bin/env python

fid = open('selected_attribute_idx.txt')
idx = fid.readlines()
fid.close()
fid = open('rap_annotation_attribute-english.txt')
name = fid.readlines() 
fid.close()
fid = open('selected_attribute_name.txt', 'w+')
for i in idx:
    fid.write(name[int(i.strip())-1])
fid.close() 
