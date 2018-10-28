import os
from collections import defaultdict


def main(output_path, submissions_dir, merge_files):
    all_classifications = defaultdict(list)

    for file in merge_files:
        content = [l.strip().split(',') for l in open(os.path.join(submissions_dir, file+'.txt'))][1:]
        for l in content:
            image_id = l[0]
            all_classifications[image_id] += l[1].split()

    with open(output_path, 'w') as f:
        f.write('image_id,labels\n')
        for image_id, classes in all_classifications.items():
            f.write('{},{}\n'.format(image_id, ' '.join(set(classes))))


if __name__ == '__main__':
    main(output_path=r'X:\OpenImages\InclusiveChallenge\submissions\merged\submission4.txt',
         submissions_dir=r'X:\OpenImages\InclusiveChallenge\submissions',
         merge_files=['full_list_resnet100_1-1100721',
                      'full_list_resnet100_0-522626',
                      'full_list_resnet100_1-1311969'])
