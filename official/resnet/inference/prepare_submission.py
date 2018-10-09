"""
image_id,labels
2b2b327132556c767a736b3d,/m/0sgh53y /m/0g4cd0
2b2b394755692f303963553d,/m/0sgh70d /m/0g44ag
etc
"""

import os

import cv2
from tqdm import tqdm

from official.resnet.inference.common import get_parser, get_classes_desc
from official.resnet.inference.model_inference import ModelInference


def main(args):
    classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=True)

    all_classifications = []

    with ModelInference(num_classes=num_classes, resnet_size=args.resnet_size, model_path=args.model_path) as model:
        for file in tqdm(os.listdir(args.images_dir)):
            image_path = os.path.join(args.images_dir, file)
            image = cv2.imread(image_path)

            found = model.infer(image=image)

            found = [classes_desc[index] for index in found]
            all_classifications += [(file, found)]

    model_path = args.model_path.split('\\')
    output_path = os.path.join(args.output_dir,
                               '{}{}.txt'.format(model_path[-2], model_path[-1].replace('model.ckpt', '')))
    with open(output_path, 'w') as f:
        f.write('image_id,labels')
        for file, found in all_classifications:
            f.write('{},{}\n'.format(file.replace('.jpg', ''), ' '.join(found)))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--output_dir', required=True)
    main(parser.parse_args())
