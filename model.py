import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.nn.parameter import Parameter

class CoralOrdinal(nn.Module):

    # We skip input_dim/input_shape here and put in the build() method as recommended in the tutorial,
    # in case the user doesn't know the input dimensions when defining the model.
    def __init__(self, num_classes, activation=None, input_shape=[16], **kwargs):
        """ Ordinal output layer, which produces ordinal logits by default.

        Args:
          num_classes: how many ranks (aka labels or values) are in the ordinal variable.
          activation: (Optional) Activation function to use. The default of None produces
            ordinal logits, but passing "ordinal_softmax" will cause the layer to output
            a probability prediction for each label.
        """
        # Pass any additional keyword arguments to Layer() (i.e. name, dtype)
        super(CoralOrdinal, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.activation = activation
        self.build(input_shape)

    # Following https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
    def build(self, input_shape):

        # Single fully-connected neuron - this is the latent variable.
        num_units = 1

        # I believe glorot_uniform (aka Xavier uniform) is pytorch's default initializer, per
        # https://pytorch.org/docs/master/generated/torch.nn.Linear.html
        # and https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
        # self.fc = self.add_weight(shape=(input_shape[-1], num_units),
        #                           # Need a unique name if there are multiple coral_ordinal layers.
        #                           name=self.name + "_latent",
        #                           initializer='glorot_uniform',
        #                           # Not sure if this is necessary:
        #                           dtype=tf.float32,
        #                           trainable=True)

        self.fc = Parameter(torch.FloatTensor(input_shape[-1], num_units))
        self.fc.requires_grad = True
        # num_classes - 1 bias terms, defaulting to 0.
        # self.linear_1_bias = self.add_weight(shape=(self.num_classes - 1,),
        #                                      # Need a unique name if there are multiple coral_ordinal layers.
        #                                      name=self.name + "_bias",
        #                                      initializer='zeros',
        #                                      # Not sure if this is necessary:
        #                                      dtype=tf.float32,
        #                                      trainable=True)
        self.linear_1_bias = Parameter(torch.randn(self.num_classes - 1))
        self.linear_1_bias.requires_grad = True

    # This defines the forward pass.
    def forward(self, inputs):
        fc_inputs = torch.matmul(inputs, self.fc)

        logits = fc_inputs + self.linear_1_bias
        if self.activation is None:
            outputs = logits
        else:
            # Not yet tested:
            outputs = self.activation(logits)
        #print(self.fc, self.linear_1_bias)
        return outputs

class CNNEncoder(nn.Module):
    def __init__(self, drop_prob=0.3, bn_momentum=0.01):
        '''
        使用pytorch提供的预训练模型作为encoder
        '''
        super(CNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=(2,2),padding=1) # to do, find equivelent of padding, kernel_initializer, kernel_regularizer in pytorch

        self.batNorm = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)  # to do, find equivelent of padding, kernel_initializer, kernel_regularizer in pytorch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,2)),

            # 2nd-5th (default) blocks
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), #end of 2nd block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), #end of 3nd block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # end of 4nd block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # end of 5nd block
        )

    def forward(self, x_3d):
        '''
        输入的是T帧图像，shape = (batch_size, t, h, w, 3)
        '''
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            # 为什么要用到no_grad()？
            # -- 因为我们使用的预训练模型，防止后续的层训练时反向传播而影响前面的层
            with torch.no_grad():
                x = self.cnn(x_3d[:, t, :, :, :])
                x = torch.flatten(x, start_dim=1)

            # 处理fc层
            #x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self, use_gru=False, cnn_out_dim=4608, rnn_hidden_layers=1, rnn_hidden_nodes=64,
            num_classes=4, drop_prob=0.0):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes

        self.drop_prob = drop_prob # for input
        self.num_classes = num_classes # 这里调整分类数目

        # rnn配置参数
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'dropout': self.drop_prob
        }

        # 使用lstm或者gru作为rnn层
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn层输出到线性分类器
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden_nodes, 16),
            nn.ReLU(True),
            CoralOrdinal(4)
        )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # 注意，前面定义rnn模块时，batch_first=True保证了以下结构：
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) # 只抽取最后一层做输出

        return x
