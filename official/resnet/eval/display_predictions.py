import os
import argparse
import cv2
from official.resnet.inference.common import get_classes_desc


def main():
    images_root_dir = r'D:\temp\stage_1_test_images'
    submission_file = r'X:\OpenImages\InclusiveChallenge\submissions\merged\submission6.txt'
    classes_path = r'X:\OpenImages\InclusiveChallenge\docs\class-descriptions_trainable.csv'

    content = [l.strip().split(',') for l in open(classes_path, encoding='utf8').readlines()]
    classes_desc = {content[i][0]: content[i][1] for i in range(len(content))}

    content = [l.strip().split(',') for l in open(submission_file).readlines()[1:]]
    for line in content:
        labels = [classes_desc[index] for index in line[1].split()]

        if len(labels) > 0:
            image_path = os.path.join(images_root_dir, line[0] + '.jpg')
            image = cv2.imread(image_path)
            image = cv2.resize(image, (480, 480))

            print(', '.join(labels))

            cv2.imshow('View', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
