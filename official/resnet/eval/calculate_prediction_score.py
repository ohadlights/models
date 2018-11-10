import os
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import fbeta_score, recall_score, precision_score

from official.resnet.inference.common import get_classes_desc


def get_ground_truth_as_array(image_ids, predictions, num_classes):
    output = []
    for image_id in image_ids:
        p = predictions[image_id]
        r = np.zeros(num_classes)
        for i in p:
            r[i] = 1
        output += [r]
    return np.array(output)


def get_predictions_as_array(image_ids, predictions, num_classes, threshold):
    output = []
    predictions_list = defaultdict(list)
    for image_id in image_ids:
        p = predictions[image_id]
        r = np.zeros(num_classes)
        for index, score in p:
            if score >= threshold:
                r[index] = 1
                predictions_list[image_id] += [index]
        output += [r]
    return np.array(output), predictions_list


def output_acuracy(image_ids, ground_truth, predictions, num_classes, prediction_file, threshold, classes_desc):
    tp = defaultdict(float)
    tn = defaultdict(float)
    fp = defaultdict(float)
    fn = defaultdict(float)
    num_appearences = defaultdict(int)

    for image_id in image_ids:
        gt = set(ground_truth[image_id])
        pred = set(predictions[image_id])
        for p in pred:
            if p not in gt:
                fp[p] += 1
            else:
                tp[p] += 1
        for g in gt:
            num_appearences[g] += 1
            if g not in pred:
                fn[g] += 1
            else:
                tn[g] += 1

    precision = defaultdict(float)
    recall = defaultdict(float)
    f1 = defaultdict(float)

    for i in range(num_classes):
        precision[i] = tp[i] / (tp[i] + fp[i] + 1e-5)
        recall[i] = tp[i] / (tp[i] + fn[i] + 1e-5)
        f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-5)

    # print('F1: {}'.format(sorted(f1.items(), key=lambda a: a[1], reverse=True)))
    # print('Pr: {}'.format(sorted(precision.items(), key=lambda a: a[1], reverse=True)))
    # print('Re: {}'.format(sorted(recall.items(), key=lambda a: a[1], reverse=True)))

    with open(prediction_file.replace('.txt', '_accuracy_{}.txt'.format(int(threshold * 100))), 'w') as f:
        for i in range(num_classes):
            if precision[i] > 0.75:
                f.write('{}\n'.format(classes_desc[i]))


def main(args):
    prediction_files = args.prediction_files.split(',')

    for prediction_file in prediction_files:

        args.prediction_file = os.path.join(args.prediction_dir, prediction_file)

        # Class indexes

        classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=True)
        index_to_id = classes_desc
        classes_desc = {i[1]: i[0] for i in classes_desc.items()}

        # calculate F1 score

        ground_truth_content = [l.strip().split() for l in open(args.ground_truth_file).readlines()]
        ground_truth = {l[0].replace('.jpg', ''): [classes_desc[a] for a in l[1:]] for l in ground_truth_content}

        image_ids = ground_truth.keys()
        ground_truth_array = get_ground_truth_as_array(image_ids, ground_truth, num_classes)

        predictions_content = [l.strip().split(',') for l in open(args.prediction_file).readlines()[1:]]
        if ':' in predictions_content[0][1] or ':' in predictions_content[1][1]:

            predictions = {l[0]: [(classes_desc[a.split(':')[0]], float(a.split(':')[1])) for a in l[1].split()] for l in predictions_content}

            for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
                predictions_array, predictions_list = get_predictions_as_array(image_ids, predictions, num_classes, threshold)

                print('threshold: {}'.format(threshold))
                print('sklearn Micro-Re-Score:', recall_score(ground_truth_array, predictions_array, average='micro'))
                print('sklearn Micro-Pr-Score:', precision_score(ground_truth_array, predictions_array, average='micro'))
                print('sklearn Micro-F2-Score:',
                      fbeta_score(ground_truth_array, predictions_array, average='micro', beta=2))
                print('')
                output_acuracy(image_ids, ground_truth, predictions_list, num_classes, args.prediction_file, threshold, index_to_id)
        else:

            predictions = {l[0]: [classes_desc[a] for a in l[1].split()] for l in predictions_content}

            predictions_array = get_ground_truth_as_array(image_ids, predictions, num_classes)

            print('sklearn Micro-Re-Score:', recall_score(ground_truth_array, predictions_array, average='micro'))
            print('sklearn Micro-Pr-Score:', precision_score(ground_truth_array, predictions_array, average='micro'))
            print('sklearn Micro-F2-Score:', fbeta_score(ground_truth_array, predictions_array, average='micro', beta=2))

            # output_acuracy(image_ids, ground_truth, predictions, num_classes, args.prediction_file, 0.0, index_to_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_file', default=r'X:\OpenImages\InclusiveChallenge\lists\all_val_multi.txt')
    parser.add_argument('--prediction_files', required=True)
    parser.add_argument('--prediction_dir', required=True)
    parser.add_argument('--classes_path', required=True)
    main(parser.parse_args())
