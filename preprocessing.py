"""
Modulo
Neural Modular Networks in PyTorch

preprocessing.py
"""

import argparse
import os
import pickle
import numpy as np

from itertools import chain
from torch.autograd import Variable
import torch
from torch import LongTensor, FloatTensor
from torch.cuda import LongTensor as CudaLongTensor, \
    FloatTensor as CudaFloatTensor


parser = argparse.ArgumentParser()

parser.add_argument('--outpath', help='Preprocessed pickle path',
                    type=str, default='modulo.pkl')
parser.add_argument('-task_type', required=True, help='Type of training task',
                    type=str, default='vqa', choices={'vqa'})
parser.add_argument('--gpu_support', help='Enable GPU support', type=bool,
                    default=False)
parser.add_argument('-query_train', required=True,
                    help='Source file for training queries.Query words are to '
                         'be separated by spaces with one query per line',
                    type=str)
parser.add_argument('-query_valid', required=True,
                    help='Source file for validation queries', type=str)
parser.add_argument('-query_test', required=True,
                    help='Source file for testing queries', type=str)
parser.add_argument('-layout_train', required=True,
                    help='Source file for training layouts in Reverse-Polish '
                         'Notation. Layout tokens are to be separated by '
                         'spaces with one layout per line', type=str)
parser.add_argument('-layout_valid', required=True,
                    help='Source file for validation queries.', type=str)
parser.add_argument('-layout_test', required=True,
                    help='Source file for testing queries.', type=str)
parser.add_argument('-answer_train', required=True,
                    help='Source file for training answers, one per line',
                    type=str)
parser.add_argument('-answer_valid', required=True,
                    help='Source file for validation answers', type=str)
parser.add_argument('-answer_test', required=True,
                    help='Source file for testing answers', type=str)
parser.add_argument('--img_train',
                    help='Source npy file for training images. '
                         'Expects [N, C, H, W]', type=str)
parser.add_argument('--img_valid', help='Source npy file for validation images',
                    type=str)

parser.add_argument('--img_test', help='Source npy file for testing images',
                    type=str)
parser.add_argument('--batch_size',
                    help='To enable batching, the same network must be '
                         'constructed each time, so the layout tokens must be '
                         'the same for every batch_size chunk of each of the '
                         'training, validation and test sets. Otherwise, use '
                         'the default batch size of 1 if the data are shuffled '
                         'or inconsistent', type=int, default=1)


def unique_words(lines):
    return set(chain(*(line.split() for line in lines)))


def read_datasets(query_path, layout_path, answer_path, return_unique=True):

    query_list, layout_list, answer_list = [], [], []
    with open(query_path) as f:
        for line in f.readlines():
            query_list.append(line.strip())
    with open(layout_path) as f:
        for line in f.readlines():
            layout_list.append(line.strip())
    with open(answer_path) as f:
        for line in f.readlines():
            assert(line.strip() in ['0', '1']),\
                'Only binary classification is supported at this time'
            answer_list.append(int(line.strip()))

    if return_unique:
        word_list = unique_words(query_list)
        token_list = unique_words(layout_list)
        return query_list, layout_list, answer_list, word_list, token_list
    else:
        return query_list, layout_list, answer_list


def make_emb_idxs(query_list, layout_list, answer_list, batch_size,
                  stoi_words, stoi_tokens,  use_cuda):

    count = len(answer_list)
    q_batches, l_batches, o_batches = [], [], []
    for i in range(count // batch_size):
        s, e = i * batch_size, (i + 1) * batch_size
        qbatch, lbatch, onehot = [], [], []

        for query in query_list[s:e]:
            qs = query.split()
            q_ = [stoi_words[w] for w in qs]
            qbatch.append(q_)
        for layout in layout_list[s:e]:
            ls = layout.split()
            l_ = [stoi_tokens[w] for w in ls]
            ans_batch = np.array(l_)
            zmat = np.zeros((ans_batch.size, len(stoi_tokens)))
            zmat[np.arange(ans_batch.size), ans_batch] = 1
            if use_cuda:
                onehot.append(Variable(CudaFloatTensor(zmat)))
            else:
                onehot.append(Variable(FloatTensor(zmat)))
            lbatch.append(l_)
        if use_cuda:
            q_batches.append(Variable(CudaLongTensor(qbatch)))
            l_batches.append(Variable(CudaLongTensor(lbatch)))
        else:
            q_batches.append(Variable(LongTensor(qbatch)))
            l_batches.append(Variable(LongTensor(lbatch)))
        o_batches.append(torch.stack(onehot, dim=2).permute(2, 0, 1))
    return q_batches, l_batches, o_batches


def main():

    args = parser.parse_args()
    state_dict = dict()
    state_dict['GPU_SUPPORT'] = args.gpu_support
    if args.gpu_support:
        print('Enabling GPU Support')

    for file_type in ['query', 'layout']:
        for dataset_type in ['train', 'valid', 'test']:
            arg = '{}_{}'.format(file_type, dataset_type)
            assert(os.path.exists(getattr(args, arg))),\
                '{} path is invalid'.format(arg)

    state_dict['QUERY_TRAIN'] = args.query_train
    state_dict['QUERY_VALID'] = args.query_valid
    state_dict['QUERY_TEST'] = args.query_test
    state_dict['LAYOUT_TRAIN'] = args.layout_train
    state_dict['LAYOUT_VALID'] = args.layout_valid
    state_dict['LAYOUT_TEST'] = args.layout_test
    state_dict['ANSWER_TRAIN'] = args.answer_train
    state_dict['ANSWER_VALID'] = args.answer_valid
    state_dict['ANSWER_TEST'] = args.answer_test
    state_dict['BATCH_SIZE'] = args.batch_size

    if args.task_type == 'vqa':
        assert args.img_train, 'Must provide training images for VQA'
        assert(os.path.exists(args.img_train)), 'img_train path is invalid'
        assert args.img_valid, 'Must provide validation image for VQA'
        assert os.path.exists(args.img_valid), 'img_valid path is invalid'
        assert args.img_test, 'Must provide test images for VQA'
        assert(os.path.exists(args.img_test)), 'img_test path is invalid'

        state_dict['IMG_TRAIN'] = args.img_train
        state_dict['IMG_VALID'] = args.img_valid
        state_dict['IMG_TEST'] = args.img_test

    train_queries, train_layouts, train_answers, train_words, train_tokens = \
        read_datasets(args.query_train, args.layout_train, args.answer_train)

    print('Detected {} vocabulary words'.format(len(train_words)))
    print('Detected {} layout tokens'.format(len(train_tokens)))

    valid_queries, valid_layouts, valid_answers, valid_words, valid_tokens = \
        read_datasets(args.query_valid, args.layout_valid, args.answer_valid)

    test_queries, test_layouts, test_answers, test_words, test_tokens = \
        read_datasets(args.query_test, args.layout_test, args.answer_test)

    extra_valid_words = valid_words - train_words
    assert(len(extra_valid_words) == 0), \
        'Validation set has the following words not seen ' \
        'in the training set: {}'.format(extra_valid_words)
    extra_valid_tokens = valid_tokens - train_tokens
    assert(len(extra_valid_tokens) == 0), \
        'Validation set has the following tokens not seen ' \
        'in the training set: {}'.format(extra_valid_tokens)

    extra_test_words = test_words - train_words
    assert(len(extra_test_words) == 0), \
        'Test set has the following words not seen ' \
        'in the training set: {}'.format(extra_test_words)
    extra_test_tokens = test_tokens - train_tokens
    assert(len(extra_test_tokens) == 0), \
        'Test set has the following tokens not seen ' \
        'in the training set: {}'.format(extra_test_tokens)

    stoi_words = dict(zip(train_words, range(len(train_words))))
    stoi_tokens = dict(zip(train_tokens, range(len(train_tokens))))

    print('Creating query and layout batches...')
    train_qbatches, train_lbatches, train_obatches = \
        make_emb_idxs(train_queries, train_layouts, train_answers,
                      args.batch_size, stoi_words, stoi_tokens,
                      args.gpu_support)
    valid_qbatches, valid_lbatches, valid_obatches = \
        make_emb_idxs(valid_queries, valid_layouts, valid_answers,
                      args.batch_size, stoi_words, stoi_tokens,
                      args.gpu_support)
    test_qbatches, test_lbatches, test_obatches = \
        make_emb_idxs(test_queries, test_layouts, test_answers,
                      args.batch_size, stoi_words, stoi_tokens,
                      args.gpu_support)
    print('...done')

    state_dict['VOCAB'] = list(train_words)
    state_dict['TOKENS'] = list(train_tokens)
    state_dict['STOI_WORDS'] = stoi_words
    state_dict['STOI_TOKENS'] = stoi_tokens
    state_dict['TRAIN_QBATCHES'] = train_qbatches
    state_dict['VALID_QBATCHES'] = valid_qbatches
    state_dict['TEST_QBATCHES'] = test_qbatches
    state_dict['TRAIN_LBATCHES'] = train_lbatches
    state_dict['VALID_LBATCHES'] = valid_lbatches
    state_dict['TEST_LBATCHES'] = test_lbatches
    state_dict['TRAIN_OBATCHES'] = train_obatches
    state_dict['VALID_OBATCHES'] = valid_obatches
    state_dict['TEST_OBATCHES'] = test_obatches

    print('Writing to pickle file...')
    with open(args.outpath, 'wb') as outpath:
        pickle.dump(state_dict, outpath, pickle.HIGHEST_PROTOCOL)
    print('...done!')


if __name__ == '__main__':
    main()
