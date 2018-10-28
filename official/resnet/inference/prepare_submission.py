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
        return [l.split()[0] + '.jpg' for l in open(args.list_path).readlines()]
    else:
        return os.listdir(args.images_dir)


def main(args):
    batch_size = 32

    classes_desc, num_classes = get_classes_desc(args.classes_path, map_to_id=True)

    all_classifications = []

    with ModelInference(num_classes=num_classes, resnet_size=args.resnet_size, model_path=args.model_path) as model:
        files = get_files(args)
        for start_index in tqdm(range(0, len(files), batch_size)):
            batch_files = files[start_index:start_index+batch_size]
            batch_paths = [os.path.join(args.images_dir, file) for file in batch_files]
            batch_images = [cv2.imread(image_path) for image_path in batch_paths]

            founds = model.infer(images=batch_images)

            for i in range(0, len(batch_files)):
                file = batch_files[i]
                found = founds[i]
                found = [classes_desc[index] for index in found]
                all_classifications += [(file, found)]

    model_path = args.model_path.split('\\')
    output_path = os.path.join(args.output_dir,
                               '{}{}.txt'.format(model_path[-2], model_path[-1].replace('model.ckpt', '')))
    with open(output_path, 'w') as f:
        f.write('image_id,labels\n')
        for file, found in all_classifications:
            image_id = file.replace('.jpg', '')
            f.write('{},{}\n'.format(image_id, ' '.join(found)))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--list_path')
    main(parser.parse_args())
