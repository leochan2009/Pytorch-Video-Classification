import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import config, os

class Dataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, time_step=30, dataDir=''):
        '''
        定义一个数据集，从UCF101中读取数据
        '''
        # 用来将类别转换为one-hot数据
        self.labels = []
        # 用来缓存图片数据，直接加载到内存中
        self.images = []
        # 是否直接加载至内存中，可以加快训练速
        self.use_mem = False

        self.skip_frame = skip_frame
        self.time_step = time_step
        self.data_list = data_list
        self._build_data_list(data_list)
        self.dataDir = dataDir

    def __len__(self):
        #return len(self.data_list) // self.time_step
        return len(self.data_list)

    def __getitem__(self, index):
        # 每次读取time_step帧图片
        #index = index * self.time_step

        imgs = self.get_frames_for_sample(self.data_list[index], self.dataDir)
        #imgs = self.data_list[index:index + self.time_step]
        # 图片读取来源，如果设置了内存加速，则从内存中读取
        if self.use_mem:
            X = [self.images[x[3]] for x in imgs]
        else:
            X = [self._read_img_and_transform(x) for x in imgs]

        # 转换成tensor
        X = torch.stack(X, dim=0)

        # 为这些图片指定类别标签
        y = torch.tensor(self.labels.index(self.data_list[index][1]))
        return X, y

    def transform(self, img):
        img_channelfirst = transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor()
        ])(img)
        return img_channelfirst #.permute(1,2,0)

    def _read_img_and_transform(self, img:str):
        return self.transform(Image.open(img).convert('RGB'))

    def _build_data_list(self, data_list=[]):
        '''
        构建数据集
        '''
        if len(data_list) == 0:
            return []

        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            # 将视频分别按照classname和videoname分组
            [classname, videoname] = x[1:3]
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname][videoname] = []

            # 将图片数据加载到内存
            if self.use_mem:
                self.images.append(self._read_img_and_transform(x[2]))

            data_group[classname][videoname].append(list(x) + [len(self.images) - 1])

        # 处理类别变量
        self.labels = sorted(list(data_group.keys()))

        # ret_list = []
        # n = 0
        #
        # # 填充数据
        # for classname in data_group:
        #     video_group = data_group[classname]
        #     for videoname in video_group:
        #         # 如果某个视频的帧总数没法被time_step整除，那么需要按照最后一帧进行填充
        #         video_pad_count = len(video_group[videoname]) % self.time_step
        #         video_group[videoname] += [video_group[videoname][-1]] * (self.time_step - video_pad_count)
        #         ret_list += video_group[videoname]
        #         n += len(video_group[videoname])
        #
        # return ret_list

    @staticmethod
    def get_frames_for_sample(sample, dataDir):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        # path = os.path.join('data', sample[0], sample[1])
        # filename = sample[2]
        # images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        path = os.path.join(dataDir, sample[0])
        if sample[2].rfind('-MatchedToMP4') >= 0:
            filename = sample[2][0:sample[2].rfind('-MatchedToMP4')]
        else:
            filename = sample[2][0:sample[2].rfind('-')]
        seg = int(sample[2][sample[2].rfind('-') + 1:])
        images = [os.path.join(path, filename + '-' + str(i).zfill(5) + '.png') for i in
                  range(30 * seg, 30 * (seg + 1))]
        return images
