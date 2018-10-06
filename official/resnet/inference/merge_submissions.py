import os
from collections import defaultdict


def main(output_path, submissions_dir, merge_files):
    all_classifications = defaultdict(list)

    for file in merge_files:
        content = [l.strip().split(',') for l in open(os.path.join(submissions_dir, file))][1:]
        for l in content:
            image_id = l[0]
            all_classifications[image_id] += l[1].split()

    with open(output_path, 'w') as f:
        f.write('image_id,labels')
        for image_id, classes in all_classifications.items():
            f.write('{},{}\n'.format(image_id, ' '.join(classes)))


if __name__ == '__main__':
    main(output_path=r'X:\OpenImages\InclusiveChallenge\submissions\merged\submission1.txt',
         submissions_dir=r'X:\OpenImages\InclusiveChallenge\submissions',
         merge_files=['',
                      ''])
