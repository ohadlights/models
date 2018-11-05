import os
from collections import defaultdict


def main(output_path, submissions_dir, merge_files):
    all_classifications = defaultdict(set)
    added_classes = defaultdict(int)

    for file in merge_files:
        content = [l.strip().split(',') for l in open(os.path.join(submissions_dir, file+'.txt'))][1:]
        for l in content:
            image_id = l[0]
            num_classes_before = len(all_classifications[image_id])
            for add_label in l[1].split():
                all_classifications[image_id].add(add_label)
            num_classes_after = len(all_classifications[image_id])
            added_classes[file] += (num_classes_after - num_classes_before)

    with open(output_path, 'w') as f:
        f.write('image_id,labels\n')
        for image_id, classes in all_classifications.items():
            f.write('{},{}\n'.format(image_id, ' '.join(classes)))

    for file, count in added_classes.items():
        print('{} added {} classes'.format(file, count))


if __name__ == '__main__':
    val = False
    if val:
        main(output_path=r'X:\OpenImages\InclusiveChallenge\submissions_val\merged1.txt',
             submissions_dir=r'X:\OpenImages\InclusiveChallenge\submissions_val',
             merge_files=['model.ckpt-1319557',
                          'finetune_range_in_10K_100K_0-186505'])
    else:
        main(output_path=r'X:\OpenImages\InclusiveChallenge\submissions\merged_1.txt',
             submissions_dir=r'X:\OpenImages\InclusiveChallenge\submissions',
             merge_files=['full_list_resnet100_1-1319557_0.295',
                          '50_100-23450',
                          '100_1K-15984',
                          '1K_10K-12488',
                          '10K_100K-166487',
                          '100K_1M-284367',
                          ])
