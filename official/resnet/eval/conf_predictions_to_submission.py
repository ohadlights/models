def main(conf_file, threshold):
    predictions_content = [l.strip().split(',') for l in open(conf_file).readlines()[1:]]
    predictions = {l[0]: [(a.split(':')[0], float(a.split(':')[1])) for a in l[1].split()] for l in predictions_content}

    output_path = conf_file.replace('_conf.txt', '_{}.txt'.format(int(threshold * 100)))

    with open(output_path, 'w') as f:
        f.write('image_id,labels\n')
        for image_id, p in predictions:
            predict_labels = []
            for class_id, score in p:
                if score >= threshold:
                    predict_labels += [class_id]
            f.write('{},{}\n'.format(image_id, ' '.join(predict_labels)))


if __name__ == '__main__':
    main(conf_file=r'',
         threshold=0.5)
