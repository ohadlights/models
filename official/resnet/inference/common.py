import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--images_dir', default=r'D:\temp\stage_1_test_images')
    parser.add_argument('--resnet_size', type=int, default=50)
    parser.add_argument('--classes_path', required=True)
    parser.add_argument('--threshold', type=float, default=0.75)
    return parser


def get_classes_desc(classes_path, map_to_id):
    content = [l.strip().split(',') for l in open(classes_path, encoding='utf8').readlines()]
    index = 0 if map_to_id else 1
    classes_desc = {i: content[i][index] for i in range(len(content))}
    return classes_desc, len(content)
