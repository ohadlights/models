

all_descs = [l.strip().split(',') for l in open(r'X:\OpenImages\InclusiveChallenge\docs\class-descriptions.csv', encoding='utf8').readlines()[1:]]
all_descs = {l[0]: l[1] for l in all_descs}

trainable_classes = [l.strip() for l in open(r'X:\OpenImages\InclusiveChallenge\docs\classes-trainable.csv', encoding='utf8').readlines()[1:]]

with open(r'X:\OpenImages\InclusiveChallenge\docs\class-descriptions_trainable.csv', 'w', encoding='utf8') as f:
    for c in trainable_classes:
        f.write('{},{}\n'.format(c, all_descs[c]))
