#!/usr/bin/env python
fid = open('../static/selected_attribute_name.txt')
att_names = []
for line in fid.readlines():
    att_names.append(line.strip())
fid.close()

fid = open('./multiatt_query_index.txt')
fid_att = open('./multiatt_query_name.txt', 'w+')
cnt = 0
for line in fid.readlines():
    line = map(int, line.strip().split())
    fid_att.write('%d '%(cnt+1))
    cnt = cnt+1
    for k in line:
        if k == 0:
            fid_att.write('male ')
        elif k == 1:
            fid_att.write('female ')
        else:
            fid_att.write('%s '%(att_names[k-1]))
    fid_att.write('\n')

fid.close()

fid_att.close()
