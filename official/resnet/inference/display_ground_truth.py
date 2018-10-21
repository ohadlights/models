import os
import cv2


def main():
    images_dir = r'X:\OpenImages\images\train_320'

    input_file = r"X:\OpenImages\InclusiveChallenge\lists\all_train_balanced.txt"
    content = [l.strip().split() for l in open(input_file).readlines()]

    class_descs = {l[0]: l[1] for l in [l.strip().split(',') for l in open(r"X:\OpenImages\InclusiveChallenge\docs\class-descriptions_trainable.csv", encoding='utf-8').readlines()]}

    viewed_images = set()

    for l in content:
        image_path = os.path.join(images_dir, l[0]) + '.jpg'
        if image_path in viewed_images:
            continue

        image = cv2.imread(image_path)
        image = cv2.resize(image, (480, 480))

        print(' '.join([class_descs[a] for a in l[1:]]))

        cv2.imshow('', image)
        cv2.waitKey(0)

        viewed_images.add(image_path)


if __name__ == '__main__':
    main()
