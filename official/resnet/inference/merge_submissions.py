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
            f.write('{},{}\n'.format(image_id, ' '.join(classes)))


if __name__ == '__main__':
    main(output_path=r'X:\OpenImages\InclusiveChallenge\submissions\merged\submission2.txt',
         submissions_dir=r'X:\OpenImages\InclusiveChallenge\submissions',
         merge_files=['range_more_100k_val_multi-2866309',
                      'range_in_10k_100K_multi-2743824',
                      'range_in_1k_10K_multi-2760289',
                      'range_in_100_1k_multi_label-908655'])
