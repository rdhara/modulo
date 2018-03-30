"""
Modulo
Neural Modular Networks in PyTorch

train.py
"""

import argparse
import pickle
import os
import torch
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from torch.optim import Adam, SGD, Adadelta, RMSprop
import time
from random import shuffle

from seq2seq import AttentionSeq2Seq
from modules import *
from preprocessing import read_datasets

parser = argparse.ArgumentParser()

# General training arguments
parser.add_argument('--pickle_path', help='Path to pickle file generated '
                                          'during the preprocessing step. '
                                          'This file must be created prior '
                                          'to training.',
                    type=str, default='modulo.pkl')
parser.add_argument('--epochs', help='Number of epochs to train the model',
                    type=int, default=1000)
parser.add_argument('--visdom',
                    help='Boolean flag for using Visdom visualization',
                    type=bool, default=True)
parser.add_argument('--checkpoint_path', help='Path to store checkpoint TAR '
                                              'file during training',
                    type=str, default='checkpoint.tar')
parser.add_argument('--checkpoint_freq', help='Save a checkpoint every '
                                              'checkpoint_freq epochs',
                    type=int, default=100)
parser.add_argument('--optimizer', help='Training optimizer', type=str,
                    default='adam',
                    choices={'adam', 'sgd', 'adadelta', 'rmsprop'})
parser.add_argument('--learning_rate', help='Optimizer learning rate',
                    type=float, default=0.001)
parser.add_argument('-use_weight_decay',
                    help='Boolean flag for using weight decay',
                    type=bool, default=True)
parser.add_argument('--weight_decay',
                    help='Optimizer weight decay (L2 regularization)',
                    type=float, default=0)
parser.add_argument('--use_gradient_clipping',
                    help='Boolean flag for using gradient clipping',
                    type=bool, default=True)
parser.add_argument('--max_grad_norm', help='Max norm for gradient clipping',
                    type=float, default=10)

# Seq2Seq RNN arguments
parser.add_argument('--word_dim', help='Word embedding dimension',
                    type=int, default=256)
parser.add_argument('--hidden_dim', help='LSTM hidden dimension',
                    type=int, default=256)
parser.add_argument('--num_layers', help='Number of LSTM layers for encoder '
                                         'and decoder', type=int, default=2)
parser.add_argument('--use_dropout',
                    help='Boolean flag for using dropout in LSTM layers '
                         '(except the final layer as usual)',
                    type=bool, default=True)
parser.add_argument('--dropout',
                    help='Dropout ratio in encoder/decoder LSTM',
                    type=float, default=0.5)

args = parser.parse_args()

assert(os.path.exists(args.pickle_path)), 'Provided pickle path is invalid'

print('Loading preprocessed pickle...')
with open(args.pickle_path, 'rb') as f:
    state_dict = pickle.load(f)
print('...done.')

if state_dict['GPU_SUPPORT']:
    from torch.cuda import FloatTensor as FloatTensor
else:
    from torch import FloatTensor

training_task = SHAPESModuloTask()
param_list = []

for mod in training_task.module_dict.values():
    param_list.extend(list(mod.parameters()))
    if state_dict['GPU_SUPPORT']:
        mod.cuda()

attn_seq2seq = AttentionSeq2Seq(
    vocab_size_1=len(state_dict['VOCAB']),
    vocab_size_2=len(state_dict['TOKENS']),
    word_dim=args.word_dim,
    hidden_dim=args.hidden_dim,
    batch_size=state_dict['BATCH_SIZE'],
    num_layers=args.num_layers,
    use_dropout=args.use_dropout,
    dropout=args.dropout,
    use_cuda=state_dict['GPU_SUPPORT']
)

param_list.extend(list(attn_seq2seq.parameters()))

print('Number of trainable paramters: {}'.format(
    sum(param.numel() for param in param_list if param.requires_grad)))


if state_dict['GPU_SUPPORT']:
    attn_seq2seq.cuda()

loss = BCEWithLogitsLoss()

if args.optimizer == 'adam':
    optimizer = Adam(
        params=param_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'sgd':
    optimizer = SGD(
        params=param_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'adadelta':
    optimizer = Adadelta(
        params=param_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop(
        params=param_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
else:
    raise ValueError('{} is not a supported optimizer'.format(args.optimizer))


train_query_list, train_layout_list, train_answer_list = \
    read_datasets(state_dict['QUERY_TRAIN'], state_dict['LAYOUT_TRAIN'],
                  state_dict['ANSWER_TRAIN'], return_unique=False)
valid_query_list, valid_layout_list, valid_answer_list = \
    read_datasets(state_dict['QUERY_VALID'], state_dict['LAYOUT_VALID'],
                  state_dict['ANSWER_VALID'], return_unique=False)
test_query_list, test_layout_list, test_answer_list = \
    read_datasets(state_dict['QUERY_TEST'], state_dict['LAYOUT_TEST'],
                  state_dict['ANSWER_TEST'], return_unique=False)

train_qbatches = state_dict['TRAIN_QBATCHES']
train_lbatches = state_dict['TRAIN_LBATCHES']
train_obatches = state_dict['TRAIN_OBATCHES']
valid_qbatches = state_dict['VALID_QBATCHES']
valid_lbatches = state_dict['VALID_LBATCHES']
valid_obatches = state_dict['VALID_OBATCHES']
test_qbatches = state_dict['TEST_QBATCHES']
test_lbatches = state_dict['TEST_LBATCHES']
test_obatches = state_dict['TEST_OBATCHES']

if isinstance(training_task, VQAModuloTask):
    train_images = np.load(state_dict['IMG_TRAIN']).astype(np.float32)
    valid_images = np.load(state_dict['IMG_VALID']).astype(np.float32)
    test_images = np.load(state_dict['IMG_TEST']).astype(np.float32)


def save_checkpoint(state, filename=args.checkpoint_path):
    torch.save(state, filename)


def sigmoid_map(preds):
    return list(map(lambda x: 1.0 if x >= 0.5 else 0.0, preds))


if args.visdom:
    from visdom import Visdom

    print('Establishing Visdom connection')
    viz = Visdom()

    startup_sec = 10
    while not viz.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    assert viz.check_connection(), 'Vizdom server not found!'

    training_loss = viz.line(X=np.array([0]), Y=np.array([0]),
                             opts=dict(title='TRAINING LOSS', xlabel='EPOCH'))
    training_accuracy = viz.line(X=np.array([0]), Y=np.array([0]),
                                 opts=dict(title='TRAINING ACCURACY',
                                           xlabel='EPOCH'))
    validation_loss = viz.line(X=np.array([0]), Y=np.array([0]),
                               opts=dict(title='VALIDATION_LOSS',
                                         xlabel='EPOCH'))
    validation_accuracy = viz.line(X=np.array([0]), Y=np.array([0]),
                                   opts=dict(title='VALIDATION_ACCURACY',
                                             xlabel='EPOCH'))

    colnames = train_query_list[0].split()
    rownames = train_layout_list[0].split()
    colnames = ['{}{}'.format(chr(24)*i, colnames[i])
                for i in range(len(colnames))]
    rownames = ['{}{}'.format(chr(24)*i, rownames[i][1:])
                for i in range(len(rownames))]

    attention_heatmap = viz.heatmap(X=np.zeros((5, 9)),
                                    opts=dict(
                                        title='ATTENTION HEATMAP',
                                        columnnames=colnames,
                                        rownames=rownames,
                                        colormap='Jet'
                                    ))

    # Change to use named_parameters() and autoinference, this is hacky
    w1w = viz.line(X=np.array([0]), Y=np.array([0]),
            opts=dict(title='seq2seq: W1.weight', xlabel='EPOCH'))
    w1g = viz.line(X=np.array([0]), Y=np.array([0]),
                   opts=dict(title='seq2seq: W1.grad', xlabel='EPOCH'))

    w2w = viz.line(X=np.array([0]), Y=np.array([0]),
            opts=dict(title='seq2seq: W2.weight', xlabel='EPOCH'))
    w2g = viz.line(X=np.array([0]), Y=np.array([0]),
                   opts=dict(title='seq2seq: W2.grad', xlabel='EPOCH'))

    w3w = viz.line(X=np.array([0]), Y=np.array([0]),
            opts=dict(title='seq2seq: W3.weight', xlabel='EPOCH'))
    w3g = viz.line(X=np.array([0]), Y=np.array([0]),
                   opts=dict(title='seq2seq: W3.grad', xlabel='EPOCH'))

    w4w = viz.line(X=np.array([0]), Y=np.array([0]),
            opts=dict(title='seq2seq: W4.weight', xlabel='EPOCH'))
    w4g = viz.line(X=np.array([0]), Y=np.array([0]),
                   opts=dict(title='seq2seq: W4.grad', xlabel='EPOCH'))

    vw = viz.line(X=np.array([0]), Y=np.array([0]),
            opts=dict(title='seq2seq: v.weight', xlabel='EPOCH'))
    vg = viz.line(X=np.array([0]), Y=np.array([0]),
                   opts=dict(title='seq2seq: v.grad', xlabel='EPOCH'))

    def update_visdom(epoch, update_dict, attention):
        viz.line(X=np.array([epoch]),
                 Y=np.array([update_dict['training_loss']]),
                 win=training_loss, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([update_dict['training_accuracy']]),
                 win=training_accuracy, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([update_dict['validation_loss']]),
                 win=validation_loss, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([update_dict['validation_accuracy']]),
                 win=validation_accuracy, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W1.weight.data, 2))]),
                 win=w1w, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W1.weight.grad.data, 2))]),
                 win=w1g, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W2.weight.data, 2))]),
                 win=w2w, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W2.weight.grad.data, 2))]),
                 win=w2g, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W3.weight.data, 2))]),
                 win=w3w, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W3.weight.grad.data, 2))]),
                 win=w3g, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W4.weight.data, 2))]),
                 win=w4w, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.W4.weight.grad.data, 2))]),
                 win=w4g, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.v.weight.data, 2))]),
                 win=vw, update='append')
        viz.line(X=np.array([epoch]),
                 Y=np.array([torch.sum(torch.pow(attn_seq2seq.v.weight.grad.data, 2))]),
                 win=vg, update='append')
        viz.heatmap(X=attention, win=attention_heatmap,
                    opts=dict(
                        title='ATTENTION HEATMAP',
                        columnnames=colnames,
                        rownames=rownames,
                        colormap='Jet'
                    ))


def main():
    print('Beginning to train!\n')
    epoch_loss_lst = []
    valid_loss_lst = []

    num_epochs = args.epochs
    batch_size = state_dict['BATCH_SIZE']

    training_acc_lst = []
    validation_acc_lst = []

    attn_lst = []
    tst_preds = []
    tst_true = []

    for e in range(num_epochs):
        batch_loss = []
        vloss_lst = []
        trn_preds = []
        val_preds = []
        trn_true = []
        val_true = []
        attns = []
        update_dict = {}

        for batch in tqdm(range(len(train_answer_list)//batch_size)):
            batch_range = list(range(batch*batch_size, (batch+1)*batch_size))
            optimizer.zero_grad()
            train_seq = train_layout_list[batch*batch_size]
            label = Variable(FloatTensor(
                np.matrix([train_answer_list[br] for br in batch_range])))
            xtxt, attn, txtloss = attn_seq2seq.forward(train_qbatches[batch],
                                                       train_lbatches[batch],
                                                       train_obatches[batch])
            attns.append(attn)
            if isinstance(training_task, VQAModuloTask):
                img = Variable(FloatTensor(
                    train_images[batch_range, :, :, :]))
                xvis = training_task.module_dict['_Img'].forward(img)
                network = training_task.assemble(train_seq, xvis, xtxt)
            else:
                network = training_task.assemble(train_seq, None, xtxt)

            output = loss(network.squeeze(2), label.permute(1, 0)) + txtloss
            output.backward()

            if args.use_gradient_clipping:
                clip_grad_norm(param_list, max_norm=args.max_grad_norm)

            optimizer.step()

            if state_dict['GPU_SUPPORT']:
                label = label.cpu()
                output = output.cpu()
                network = network.cpu()

            trn_true.extend(list(label.permute(1, 0).data.numpy().flatten()))
            trn_preds.extend(
                list(sigmoid(network.squeeze(2)).data.numpy().flatten()))
            batch_loss.append(output.data.numpy()[0])

        epoch_loss_lst.append(np.mean(batch_loss))
        print('EPOCH {}/{} \n\tTRAINING LOSS = {}'.format(e + 1, num_epochs,
                                                          epoch_loss_lst[-1]))
        update_dict['training_loss'] = epoch_loss_lst[-1]

        for vbatch in range(len(valid_answer_list)//batch_size):
            vbatch_range = list(range(
                vbatch * batch_size, (vbatch + 1) * batch_size))
            valid_seq = valid_layout_list[vbatch*batch_size]
            valid_label = Variable(FloatTensor(
                    np.matrix([valid_answer_list[br] for br in vbatch_range])))
            vxtxt, _, vtxtloss = attn_seq2seq.forward(valid_qbatches[vbatch],
                                               valid_lbatches[vbatch],
                                               valid_obatches[vbatch])
            if isinstance(training_task, VQAModuloTask):
                vimg = Variable(FloatTensor(
                    valid_images[vbatch_range, :, :, :]))
                vx_vis = training_task.module_dict['_Img'].forward(vimg)
                network = training_task.assemble(valid_seq, vx_vis, vxtxt)
            else:
                network = training_task.assemble(valid_seq, None, vxtxt)

            output = loss(network.squeeze(2), valid_label.permute(1, 0)) + vtxtloss

            if state_dict['GPU_SUPPORT']:
                valid_label = valid_label.cpu()
                output = output.cpu()
                network = network.cpu()
            val_true.extend(
                list(valid_label.permute(1, 0).data.numpy().flatten()))
            val_preds.extend(
                list(sigmoid(network.squeeze(2)).data.numpy().flatten()))
            vloss_lst.append(output.data.numpy()[0])
        valid_loss_lst.append(np.mean(vloss_lst))

        print('\tVALIDATION LOSS = {}'.format(valid_loss_lst[-1]))
        update_dict['validation_loss'] = valid_loss_lst[-1]

        trn_preds = sigmoid_map(trn_preds)
        val_preds = sigmoid_map(val_preds)
        training_acc = np.mean(np.array(trn_preds) == np.array(trn_true))*100
        validation_acc = np.mean(np.array(val_preds) == np.array(val_true))*100
        print('\tTRAINING ACCURACY: {}\n\tVALIDATION ACCURACY: {}'.format(
            training_acc, validation_acc))

        training_acc_lst.append(training_acc)
        validation_acc_lst.append(validation_acc)
        attn_lst.append(attns)
        update_dict['training_accuracy'] = training_acc
        update_dict['validation_accuracy'] = validation_acc

        if state_dict['GPU_SUPPORT']:
            first_attn = torch.stack(attn_lst[-1][0]).permute(1, 0, 2)[0].cpu().data.numpy()
        else:
            first_attn = torch.stack(attn_lst[-1][0]).permute(1, 0, 2)[0].data.numpy()

        if args.visdom:
            update_visdom(e, update_dict, first_attn)

        if e % args.checkpoint_freq == 0:
            checkpoint_dict = {
                'seq2seq': attn_seq2seq.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            for mod_name, mod in training_task.module_dict.items():
                checkpoint_dict[mod_name] = mod.state_dict()

            save_checkpoint(checkpoint_dict)
            print('\nSAVED CHECKPOINT\n')
    print('DONE TRAINING, EVALUATING TEST PERFORMANCE...')
    tloss_lst = []

    for tbatch in range(len(test_answer_list) // batch_size):
        tbatch_range = list(range(
            tbatch * batch_size, (tbatch + 1) * batch_size))
        test_seq = test_layout_list[tbatch * batch_size]
        test_label = Variable(FloatTensor(
            np.matrix([test_answer_list[br] for br in tbatch_range])))
        txtxt, _, _ = attn_seq2seq.forward(test_qbatches[tbatch],
                                           test_lbatches[tbatch],
                                           test_obatches[tbatch])
        if isinstance(training_task, VQAModuloTask):
            timg = torch.autograd.Variable(FloatTensor(
                valid_images[tbatch_range, :, :, :]))
            tx_vis = training_task.module_dict['_Img'].forward(timg)
            network = training_task.assemble(test_seq, tx_vis, txtxt)
        else:
            network = training_task.assemble(test_seq, None, txtxt)

        output = loss(network.squeeze(2), test_label.permute(1, 0))

        if state_dict['GPU_SUPPORT']:
            test_label = test_label.cpu()
            output = output.cpu()
            network = network.cpu()
        tst_true.extend(
            list(test_label.permute(1, 0).data.numpy().flatten()))
        tst_preds.extend(
            list(sigmoid(network.squeeze(2)).data.numpy().flatten()))
        tloss_lst.append(output.data.numpy()[0])
    tst_preds = sigmoid_map(tst_preds)
    tst_acc = np.mean(np.array(tst_preds) == np.array(tst_true))*100

    print('TESTING LOSS: {}\nTESTING ACCURACY: {}'.format(
        np.mean(tloss_lst), tst_acc))


if __name__ == '__main__':
    main()
