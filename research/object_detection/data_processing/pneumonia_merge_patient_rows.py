import random


content = [l.strip().split(',') for l in open(r'Z:\Research\pneumonia\stage_1_train_labels.csv').readlines()]

data = {}

for l in content:
    pid = l[0]
    if pid not in data:
        data[pid] = []
    if l[-1] == '1':
        data[pid] += [[l[1], l[2], l[3], l[4]]]

pids = list(data.keys())
train_ids = set(pids[1000:])
val_ids = set(pids[:1000])

with open(r'Z:\Research\pneumonia\stage_1_train_labels_train.txt', 'w') as f_train,\
    open(r'Z:\Research\pneumonia\stage_1_train_labels_val.txt', 'w') as f_val:

    for pid, bboxes in data.items():

        f = f_val if pid in val_ids else f_train

        f.write(pid)

        for box in bboxes:
            f.write(' {}'.format(','.join(box)))

        f.write('\n')