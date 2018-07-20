import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
from nets import nets_factory


def main(args):
    ####################
    # Read images list #
    ####################
    images_list = [i.strip() for i in open(args.images_list).readlines()]

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        args.model_name,
        num_classes=None,
        is_training=False)

    ####################
    # Define the model #
    ####################
    images = tf.placeholder(tf.float32, shape=(None, args.image_height, args.image_width, 3), name='input')
    logits, _ = network_fn(images)

    ###########
    # Extract #
    ###########

    with tf.Session() as sess:

        tf.train.Saver().restore(sess=sess, save_path=args.model_path)

        for i in tqdm(range(0, len(images_list), args.batch_size)):

            batch_image_paths = images_list[i:i+args.batch_size]
            batch_images = []
            for path in batch_image_paths:
                image = cv2.imread(path).astype(np.float32) / 255 - 0.5
                image = cv2.resize(image, (args.image_width, args.image_height))
                batch_images += [image]
            batch_images = np.array(batch_images)

            embeddings = sess.run(logits, feed_dict={images: batch_images})

            for index in range(len(batch_image_paths)):
                image_name = os.path.basename(batch_image_paths[index])
                path = os.path.join(args.output_dir, image_name + '.npy')
                np.save(path, embeddings[index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='pnasnet_large')
    parser.add_argument('--model_path', type=str, default=r'.\checkpoints\pnasnet-5_large_2017_12_13.ckpt')
    parser.add_argument('--images_list', type=str, default=r'data\test_images.txt')
    parser.add_argument('--output_dir', type=str, default=r'X:\OpenImages\embeddings\pnasnet_large\test')
    parser.add_argument('--image_width', type=int, default=331)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    main(parser.parse_args())