import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import fbeta_score, recall_score, precision_score

from official.resnet.inference.common import get_classes_desc


def get_predictions_as_array(image_ids, predictions, num_classes):
    output = []
    for image_id in image_ids:
        p = predictions[image_id]
        r = np.zeros(num_classes)
        for i in p:
            r[i] = 1
        output += [r]
    return np.array(output)


def main(args):

    # Class indexes

    classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=True)
    classes_desc = {i[1]: i[0] for i in classes_desc.items()}

    # calculate F1 score

    ground_truth_content = [l.strip().split() for l in open(args.ground_truth_file).readlines()]
    ground_truth = {l[0].replace('.jpg', ''): [classes_desc[a] for a in l[1:]] for l in ground_truth_content}

    predictions_content = [l.strip().split(',') for l in open(args.prediction_file).readlines()[1:]]
    predictions = {l[0]: [classes_desc[a] for a in l[1].split()] for l in predictions_content}

    image_ids = predictions.keys()

    ground_truth_array = get_predictions_as_array(image_ids, ground_truth, num_classes)
    predictions_array = get_predictions_as_array(image_ids, predictions, num_classes)

    print('sklearn Micro-Re-Score:', recall_score(ground_truth_array, predictions_array, average='micro'))
    print('sklearn Micro-Pr-Score:', precision_score(ground_truth_array, predictions_array, average='micro'))
    print('sklearn Micro-F2-Score:', fbeta_score(ground_truth_array, predictions_array, average='micro', beta=2))

    # find most missed labels and most false-predicted labels

    fp = defaultdict(float)
    fn = defaultdict(float)
    num_appearences = defaultdict(int)

    for image_id in image_ids:
        gt = set(ground_truth[image_id])
        pred = set(predictions[image_id])
        for p in pred:
            if p not in gt:
                fp[p] += 1
        for g in gt:
            num_appearences[g] += 1
            if g not in pred:
                fn[g] += 1

    for i in range(num_classes):
        fp[i] = fp[i] / num_appearences[i] if num_appearences[i] != 0 else 0
        fn[i] = fn[i] / num_appearences[i] if num_appearences[i] != 0 else 0

    print()
    print('fp: {}'.format(sorted(fp.items(), key=lambda a: a[1], reverse=True)))
    print('fn: {}'.format(sorted(fn.items(), key=lambda a: a[1], reverse=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_file', default=r'X:\OpenImages\InclusiveChallenge\lists\all_val_multi.txt')
    parser.add_argument('--prediction_file', required=True)
    parser.add_argument('--classes_path', required=True)
    main(parser.parse_args())
