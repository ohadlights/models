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
    output_path = r'D:\temp\inclusive\analysis\merged_balanced40_all90_100K70_10K65_1K65_100S55_50S65.txt'
    main(output_path=output_path,
         submissions_dir=os.path.dirname(output_path),
         merge_files=[
             'merged_balanced40_all90_100K70_10K65_1K65_100S55',
             '0_50-7515_pick'
         ])
