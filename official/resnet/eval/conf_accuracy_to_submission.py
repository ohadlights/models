import os


def read_accuracy_files(conf_file):
    search_dir = os.path.dirname(conf_file)
    prefix = os.path.basename(conf_file).replace('.txt', '_accuracy_')
    files = filter(lambda f: f.startswith(prefix), os.listdir(search_dir))
    label_to_threshold = {}
    for file in files:
        threshold = 0.01 * float(file[-6:-4])
        labels = [l.strip() for l in open(os.path.join(search_dir, file))]
        for label in labels:
            if label not in label_to_threshold:
                label_to_threshold[label] = threshold
            elif threshold < label_to_threshold[label]:
                label_to_threshold[label] = threshold
    print('Number of labels: {}'.format(len(label_to_threshold)))
    return label_to_threshold


def main(conf_files_dir, conf_files):

    for conf_file in conf_files.split(','):

        print('Starting: {}'.format(conf_file))

        conf_file = os.path.join(conf_files_dir, conf_file)

        predictions_content = [l.strip().split(',') for l in open(conf_file).readlines()[1:]]
        predictions = {l[0]: [(a.split(':')[0], float(a.split(':')[1])) for a in l[1].split()] for l in predictions_content}

        label_to_threshold = read_accuracy_files(conf_file)

        output_path = conf_file.replace('_conf.txt', '_pick.txt')

        with open(output_path, 'w') as f:
            f.write('image_id,labels\n')
            for image_id, p in predictions.items():
                predict_labels = []
                for class_id, score in p:
                    if class_id in label_to_threshold and score >= label_to_threshold[class_id]:
                        predict_labels += [class_id]
                f.write('{},{}\n'.format(image_id, ' '.join(predict_labels)))


if __name__ == '__main__':
    main(conf_files_dir=r'X:\OpenImages\InclusiveChallenge\submissions',
         conf_files='100K_1M-284367_conf.txt,10K_100K-166487_conf.txt,1K_10K-12488_conf.txt,100_1K-15984_conf.txt,50_100-23450_conf.txt,0_50-7515_conf.txt')
