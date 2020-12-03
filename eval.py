import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas
import os
import argparse
import cv2
import pandas as pd
from activations import *
import glob
from dataloader import Dataset
from model import CNNEncoder, RNNDecoder
import config

def load_imgs_from_video(video_fd)->list:
    """Extract images from video.

    Args:
        path(str): The path of video.

    Returns:
        A list of PIL Image.
    """

    #video_fd.set(16, True)
    # flag 16: 'CV_CAP_PROP_CONVERT_RGB'
    # indicating the images should be converted to RGB.

    if not video_fd.isOpened():
        raise ValueError('Invalid path!')

    images = [] # type: list[Image]

    success, frame = video_fd.read()
    images.append(Image.fromarray(frame))
    while success and len(images)<30:
        images.append(Image.fromarray(frame))
        success, frame = video_fd.read()

    return images

def _eval_fromVideo(model, device, video_path: str, labels=[])->list:
    """Inference the model and return the labels.

    Args:
        checkpoint(str): The checkpoint where the model restore from.
        path(str): The path of videos.
        labels(list): Labels of videos.

    Returns:
        A list of labels of the videos.
    """
    if not os.path.exists(video_path):
        raise ValueError('Invalid path! which is: {}'.format(video_path))

    # Do inference
    pred_labels = []
    video_names = glob.glob(os.path.join(video_path, '20180802-094306_912.mp4'))
    with torch.no_grad():
        for video in tqdm(video_names, desc='Inferencing'):
            # read images from video
            video_fd = cv2.VideoCapture(os.path.join(video_path, video))
            total = int(video_fd.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(total//30):
                images = load_imgs_from_video(video_fd)
                # apply transform
                images = [Dataset.transform(None, img) for img in images]
                # stack to tensor, batch size = 1
                images = torch.stack(images, dim=0).unsqueeze(0)
                # do inference
                images = images.to(device)
                pred_y = model(images) # type: torch.Tensor
                y_ordinalSoftmax = ordinal_softmax(pred_y)
                probs_df = pd.DataFrame(y_ordinalSoftmax.cpu().data.numpy())

                probs_df.head()
                labelstemp = probs_df.idxmax(axis=1)

                pred_labels.append([video+"-"+str(i), labelstemp.values])
                print(pred_labels[-1])

    if len(labels) > 0:
        acc = accuracy_score(pred_labels, labels)
        print('Accuracy: %0.2f' % acc)

    # Save results
    pandas.DataFrame(pred_labels).to_csv('result.csv', index=False)
    print('Results has been saved to {}'.format('result.csv'))

    return pred_labels

def _eval_fromCSV(model, device, csv_path: str):
    raw_data = pandas.read_csv(csv_path)
    dataloaders = DataLoader(Dataset(raw_data.to_numpy(), dataDir=os.path.abspath(os.path.dirname(csv_path))),batch_size=1, shuffle=False)
    model.eval()

    print('Size of Test Set: ', len(dataloaders.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    y_gd = []
    y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        for X, y in tqdm(dataloaders, desc='Validating'):
            # 对测试集中的数据进行预测
            X, y = X.to(device), y.to(device)
            y_ = model(X)

            # 收集prediction和ground truth
            y_ordinalSoftmax = ordinal_softmax(y_)
            probs_df = pd.DataFrame(y_ordinalSoftmax.cpu().data.numpy())

            probs_df.head()
            labels = probs_df.idxmax(axis=1)
            y_gd += y.cpu().numpy().tolist()
            # y_pred += y_.cpu().numpy().tolist()
            y_pred += labels.to_numpy().tolist()
    plt.plot(y_pred, label='current modeled')
    plt.plot(y_gd, label='accessed')
    plt.show()
    np.save('predicted'+os.path.splitext(os.path.basename(csv_path))[0]+'.npy', y_pred)
    np.save('y_gd'+os.path.splitext(os.path.basename(csv_path))[0]+'.npy', y_gd)
    return y_pred, y_gd

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/videos -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to data')
    parser.add_argument('-r', '--checkpoint', help='path to the checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    modeled = np.load('predicted'+os.path.splitext(os.path.basename(args.data_path))[0]+'.npy')
    accessed = np.load('y_gd'+os.path.splitext(os.path.basename(args.data_path))[0]+'.npy')
    plt.plot(modeled, label = "modeled")
    plt.plot(accessed, label = 'accessed')
    plt.legend()
    plt.show()
    print('Loading model from {}'.format(args.checkpoint))
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Build model
    model = nn.Sequential(
        CNNEncoder(**config.cnn_encoder_params),
        RNNDecoder(**config.rnn_decoder_params)
    )
    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    # Load model
    ckpt = torch.load(args.checkpoint, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Model has been loaded from {}'.format(args.checkpoint))

    label_map = [-1] * config.rnn_decoder_params['num_classes']
    # load label map
    if 'label_map' in ckpt:
        label_map = ckpt['label_map']
    directlyFromVideo = False
    if directlyFromVideo:
        _eval_fromVideo(model, device, args.data_path)
    else:
        _eval_fromCSV(model, device, args.data_path)
