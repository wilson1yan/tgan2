#!/usr/bin/env python3

import argparse
import warnings

import chainer
import numpy
import yaml

import tgan2
import tgan2.evaluations.inception_score
from tgan2.utils import make_config
from tgan2.utils import make_instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-o', '--out', default='result.yml')
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--samples', type=str, required=True)
    args = parser.parse_args()

    conf_dicts = [yaml.load('conf/dset/ucf101_192x256.yml')]
    config = make_config(conf_dicts, args.attrs)
    return config, args


def main(config, args):
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    conf_classifier = config['inception_score']['classifier']
    classifier = make_instance(tgan2, conf_classifier)
    if 'model_path' in conf_classifier:
        chainer.serializers.load_npz(
            conf_classifier['model_path'],
            classifier, path=conf_classifier['npz_path'])
    if args.gpu >= 0:
        classifier.to_gpu()

    xs = numpy.load(args.samples)
    xs = xs.astype('float32') / 255
    xs = 2 * xs - 1
    mean, std = tgan2.evaluations.inception_score.inception_score(
        classifier, xs, args.batchsize, splits=1
    )

    print(f'{mean} +- {std}')


if __name__ == '__main__':
    config, args = parse_args()
    # Ignore warnings
    warnings.simplefilter('ignore')
    main(config, args)
