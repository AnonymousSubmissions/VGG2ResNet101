import os
import torch
import argparse
import numpy as np
import time
from models.vgg import VGG16Extractor
from models.resnet import ResnetExtractor
import yaml
import cv2


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to perform detection upon",
        default="images",
        type=str)
    parser.add_argument(
        '--num_workers',
        default=2,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')
    args = parser.parse_args()
    return args


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize_wh, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize_wh = resize_wh
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, target=None):

        interp_methods = [
            cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
            cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
        ]
        interp_method = interp_methods[0]
        img = cv2.resize(
            np.array(img), (self.resize_wh[0], self.resize_wh[1]),
            interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img), target


def main():
    global args
    args = arg_parse()
    bgr_means = [104, 117, 123]
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # VGG16 = VGG16Extractor()
    ResNet101 = ResnetExtractor('101')

    input_folder = args.images#'/home/leo/Documents/datasets/coco/images/val2014'  # args.images


    img_whs = [[224,224],
              [512,512],
              [1024,1024]]

    for i in range(0,3):
        img_wh = img_whs[i]
        #
        # # run vgg
        # tt, tt1, tt2, tt3, tt4, tt5, tt6, count = 0, 0, 0, 0, 0, 0, 0, 0
        # testTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
        # for iter in range(0,20):
        #     for item in os.listdir(input_folder):
        #         img_path = os.path.join(input_folder, item)
        #         img = cv2.imread(img_path)
        #         with torch.no_grad():
        #             x = testTransform(img)[0].unsqueeze(0)
        #             x = x.cuda()
        #
        #             output, t1, t2, t3, t4, t5, t6 = VGG16(x)
        #             tt1 += t1
        #             tt2 += t2
        #             tt3 += t3
        #             tt4 += t4
        #             tt5 += t5
        #             tt6 += t6
        #             count += 1
        #
        # tt = (tt1 + tt2 + tt3 + tt4 + tt5 + tt6) / count
        # tt1 = tt1 / count
        # tt2 = tt2 / count
        # tt3 = tt3 / count
        # tt4 = tt4 / count
        # tt5 = tt5 / count
        # tt6 = tt6 / count
        # print('[{:d}x{:d}] Time of     VGG16: {:.1f}ms, (block1: {:.1f}ms, block2:{:.1f}ms, block3:{:.1f}ms, block4:{:.1f}ms, block5:{:.1f}ms, dilation:{:.1f}ms)'.format(
        #         img_wh[0], img_wh[1], tt*1000, tt1*1000, tt2*1000, tt3*1000, tt4*1000, tt5*1000, tt6*1000))

    for i in range(0, 3):
        img_wh = img_whs[i]
        # run resnet101
        tt, tt1, tt2, tt3, tt4, tt5, count = 0, 0, 0, 0, 0, 0, 0
        testTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
        for iter in range(0, 20):
            for item in os.listdir(input_folder):
                img_path = os.path.join(input_folder, item)
                img = cv2.imread(img_path)
                with torch.no_grad():
                    x = testTransform(img)[0].unsqueeze(0)
                    x = x.cuda()

                    output, t1, t2, t3, t4, t5 = ResNet101(x)
                    tt1 += t1
                    tt2 += t2
                    tt3 += t3
                    tt4 += t4
                    tt5 += t5
                    count += 1

        tt = (tt1 + tt2 + tt3 + tt4 + tt5) / count
        tt1 = tt1 / count
        tt2 = tt2 / count
        tt3 = tt3 / count
        tt4 = tt4 / count
        tt5 = tt5 / count
        print('[{:d}x{:d}] Time of ResNet101: {:.1f}ms, (block1: {:.1f}ms, block2:{:.1f}ms, block3:{:.1f}ms, block4:{:.1f}ms, block5:{:.1f}ms)'.format(
                img_wh[0], img_wh[1], tt*1000, tt1*1000, tt2*1000, tt3*1000, tt4*1000, tt5*1000))


if __name__ == '__main__':
    main()
