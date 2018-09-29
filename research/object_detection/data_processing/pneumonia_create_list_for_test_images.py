import os


images_dir = r'Z:\Research\pneumonia\stage_1_test_images'
files = [f.replace('.dcm', '') for f in os.listdir(images_dir)]

with open(r'Z:\Research\pneumonia\stage_1_test_list.txt', 'w') as f:
    for file in files:
        f.write('{}\n'.format(file))
