import os 
from tqdm import tqdm

hidden_size_bar = tqdm(['128', '512', '1024'])
learning_rate_bar = tqdm(['2e-4', '3E-5'])
batch_size_bar = tqdm(['128', '512'])
layer_size_bar = tqdm(['1', '2'])
k_fold_bar = tqdm(['0', '1', '2', '3', '4'])

for h in hidden_size_bar:
    for l in layer_size_bar:
        for b in batch_size_bar:
            for lr in learning_rate_bar:
                for k in k_fold_bar:
                    hidden_size_bar.set_description('hidden size: {}'.format(h))
                    learning_rate_bar.set_description('learning rate: {}'.format(lr))
                    batch_size_bar.set_description('batch size: {}'.format(b))
                    layer_size_bar.set_description('layer: {}'.format(l))
                    k_fold_bar.set_description('{}-th fold'.format(k))
                    os.system("python main.py --data-dir {} --model-dir experiments/k_{}/lr_{}_hidden_{}_layer_{}_batchsize_{} --gpu {} --k {}".format('./data', k, lr, h, l, b, '0', k))