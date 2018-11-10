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
    output_path = r'X:\OpenImages\InclusiveChallenge\submissions\merged_all-1319557_50_enhanced_95_pr75.txt'
    main(output_path=output_path,
         submissions_dir=os.path.dirname(output_path),
         merge_files=[
             'all-1319557_50_enhanced_95',
             # 'all-1319557_50',
             '100K_1M-284367_pick',
             '10K_100K-166487_pick',
             '1K_10K-12488_pick',
             '100_1K-15984_pick',
             '50_100-23450_pick',
             '0_50-7515_pick'
         ])
