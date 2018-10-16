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


def get_files(args):
    if args.list_path:
        return [l.split()[0] for l in open(args.list_path).readlines()]
    else:
        return os.listdir(args.images_dir)


def main(args):
    num_classes = 28

    all_classifications = []

    with ModelInference(num_classes=num_classes, resnet_size=args.resnet_size, model_path=args.model_path) as model:
        files = get_files(args)
        for file in tqdm(files):
            image_path = os.path.join(args.images_dir, file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            found = model.infer(image=image)

            all_classifications += [(file, found)]

    model_path = args.model_path.split('\\')
    output_path = os.path.join(args.output_dir,
                               '{}-{}{}.txt'.format(args.list_type, model_path[-2], model_path[-1].replace('model.ckpt', '')))
    with open(output_path, 'w') as f:
        f.write('Id,Predicted\n')
        for file, found in tqdm(all_classifications):
            f.write('{},{}\n'.format(file.replace('_rgba.png', ''), ' '.join([str(a) for a in found])))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--list_path')
    parser.add_argument('--list_type', default='test')
    main(parser.parse_args())
