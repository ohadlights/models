import os

import cv2

from official.resnet.inference.common import get_parser, get_classes_desc
from official.resnet.inference.model_inference import ModelInference


def main(args):
    classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=False)

    with ModelInference(num_classes=num_classes, resnet_size=args.resnet_size, model_path=args.model_path) as model:

        for file in os.listdir(args.images_dir):

            image_path = os.path.join(args.images_dir, file)
            image = cv2.imread(image_path)

            found = model.infer(image=image)

            found = [classes_desc[index] for index in found]
            print(', '.join(found))

            image = cv2.resize(image, (480, 480))
            cv2.imshow('view', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
