import os


train_images_dir = r'X:\OpenImages\images\train'
test_images_dir = r'X:\OpenImages\images\test'

train_images = os.listdir(train_images_dir)
test_images = os.listdir(test_images_dir)

with open(r'..\data\train_images.txt', 'w') as f:
    for image in train_images:
        f.write('{}\n'.format(os.path.join(train_images_dir, image)))

with open(r'..\data\test_images.txt', 'w') as f:
    for image in test_images:
        f.write('{}\n'.format(os.path.join(test_images_dir, image)))
