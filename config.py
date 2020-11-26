img_w = 250
img_h = 250

dataset_params = {
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 1,
    'pin_memory': True
}

cnn_encoder_params = {
    'drop_prob': 0.3,
    'bn_momentum': 0.01
}

rnn_decoder_params = {
    'use_gru': False,
    'cnn_out_dim': 4608,
    'rnn_hidden_layers': 1,
    'rnn_hidden_nodes': 64,
    'num_classes': 4,
    'drop_prob': 0.0
}

learning_rate = 1e-5
epoches = 1000
log_interval = 20 # 打印间隔，默认每20个batch_size打印一次
save_interval = 1 # 模型保存间隔，默认每个epoch保存一次
validation_interval = 5