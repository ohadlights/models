import argparse

from tqdm import tqdm


def main(args):

    challenge_labels = {l.split(',')[0] for l in open(args.challenge_label_map)}

    content = open(args.input_path).readlines()

    take_indexes = [0, 2, 4, 5, 6, 7, 10]

    with open(args.output_path, 'w') as f:
        info = content[0].strip().split(',')
        f.write('{}\n'.format(','.join([info[i] for i in take_indexes])))
        for l in tqdm(content[1:]):
            info = l.strip().split(',')
            label = info[2]
            if label in challenge_labels:
                f.write('{}\n'.format(','.join([info[i] for i in take_indexes])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--challenge_label_map', default=r'X:\open_images_v5\docs\challenge-2019-classes-description-500.csv')
    p.add_argument('--input_path', required=True)
    p.add_argument('--output_path', required=True)
    main(p.parse_args())
