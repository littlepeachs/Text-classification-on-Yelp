import os
import copy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import sys
from dictionary import Dictionary
from dataset import CLSDataset, BOWDataset, FastTextDataset, collate_fn
import numpy as np
import collections
from matplotlib import pyplot as plt 
import argparse
import logging
from LSTM import LSTM
from CNN import CNN
from FastText import FastText
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--embedding-size", default=512, type=int)
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--model-type", default="bow", choices=["bow", "fast_text", "cnn", "lstm"])
    args = parser.parse_args()
    return args


def evaluate(args, model, valid_loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_num = 0
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) - 1
            outputs = model(inputs)
            nll_loss = criterion(outputs, labels)
            loss = nll_loss.mean()
            _, predict = torch.max(outputs, dim=1)
            correct = (predict == labels).sum().item()

            total_loss += nll_loss.sum().item()
            total_correct += correct
            total_num += len(labels)

        loss = total_loss / total_num
        acc = total_correct / total_num * 100

        return loss, acc


def train(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    vocab_file = "./yelp_small/vocab.txt"

    batch_size = args.batch_size

    # this can get the vocabulary
    # we provide the class ``CLSDataset'' to show how to use ``Dictionary''
    dictionary = Dictionary()
    dictionary.add_from_file(vocab_file)


    if args.model_type == 'bow':
        train_dataset = BOWDataset(dictionary=dictionary, split='train', n=2);
        valid_dataset = BOWDataset(dictionary=dictionary, split='valid', n=2);
        test_dataset = BOWDataset(dictionary=dictionary, split='test', n=2);

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    elif args.model_type == 'fast_text':
        n_gram_size = len(dictionary)
        train_dataset = FastTextDataset(dictionary=dictionary, split='train', n=2, n_gram_size=len(dictionary));
        valid_dataset = FastTextDataset(dictionary=dictionary, split='valid', n=2, n_gram_size=len(dictionary));
        test_dataset = FastTextDataset(dictionary=dictionary, split='test', n=2, n_gram_size=len(dictionary));

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    elif args.model_type == 'lstm' or args.model_type == 'cnn':
        train_dataset = CLSDataset(dictionary=dictionary, split='train', block_size=8*batch_size);
        valid_dataset = CLSDataset(dictionary=dictionary, split='valid');
        test_dataset = CLSDataset(dictionary=dictionary, split='test');

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("data preparation finished")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'bow':
        model = nn.Linear(train_dataset.vocab_size, 5)
    elif args.model_type == 'fast_text':
        model = FastText(train_dataset.total_size, args.embedding_size, dictionary.pad())
    elif args.model_type == 'lstm':
        model = LSTM(len(dictionary), args.embedding_size, args.hidden_size, args.num_layers, dictionary.pad())
    elif args.model_type == 'cnn':
        model = CNN(len(dictionary), args.embedding_size, args.hidden_size, dictionary.pad())

    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    save_dir = args.save_dir + '_' + args.model_type
    
    os.makedirs(save_dir, exist_ok=True) 

    log_file_path = args.model_type + ".log"
    logging.basicConfig(level=logging.DEBUG,  # 设置日志显示级别
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',  # 指定日期时间格式
                    filename=log_file_path,  # 指定日志存储的文件及位置
                    filemode='w',  # 文件打开方式
                    )  # 指定handler使用的日志显示格式
    

    best_acc = 0
    print("ready to train")
    for epoch in range(args.num_epoch):
        print("epoch: {}".format(epoch))
        total_loss = 0
        total_correct = 0
        total_num = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) - 1  # 因为最后是要0-4
            outputs = model(inputs)
            nll_loss = criterion(outputs, labels)
            loss = nll_loss.mean()
            #print(loss)
            _, predict = torch.max(outputs, dim=1)
            correct = (predict == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += nll_loss.sum().item()
            total_correct += correct
            total_num += len(labels)


            if (i+1) % 50 == 0:
                loss = total_loss / total_num
                acc = total_correct / total_num * 100
                logging.info("Epoch: %d, \t Iter: %d/%d, \t Loss: %0.8f, \t Acc: %0.3f , \t lr: %0.6f" % (epoch + 1, i+1, len(train_loader), loss, acc, optimizer.param_groups[0]['lr']))
        
        loss = total_loss / total_num
        acc = total_correct / total_num * 100
        train_loss.append(loss)
        train_acc.append(acc)

        loss, acc = evaluate(args, model, valid_loader, criterion)
        print("valid_loss:{},valid_acc:{}".format(loss, acc))
        valid_loss.append(loss)
        valid_acc.append(acc)

        if acc > best_acc:
            torch.save(model, save_dir + "/best_checkpoint.pt")
            best_acc = acc

        logging.info(("Epoch: %d, \t Valid, Loss: %0.8f, \t Acc: %0.3f , \t Best Acc: %.3f" % (epoch + 1, loss, acc, best_acc)))


        loss_and_acc = {'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss':valid_loss, 'valid_acc': valid_acc}
        torch.save(loss_and_acc, save_dir + "/loss_acc.pt")

        x = range(epoch + 1)
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(x, train_loss, label='training loss')
        ax1.plot(x, valid_loss, label='valid loss')
        ax1.set_ylabel("loss")
        ax1.set_xlabel("epoch")
        plt.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(x, train_acc, label='training accuracy')
        ax2.plot(x, valid_acc, label='valid accuracy')
        ax2.set_ylabel("accuracy")
        ax2.set_xlabel("epoch")

        plt.legend()
        plt.savefig(save_dir + "/loss_and_acc.pdf")
        plt.savefig(save_dir + "/loss_and_acc.png")
        plt.close()
    
    model = torch.load(save_dir + "/best_checkpoint.pt")
    valid_loss, valid_acc = evaluate(args, model, valid_loader, criterion)
    test_loss, test_acc = evaluate(args, model, test_loader, criterion)
    
    logging.info(("valid loss: %0.3f, \t valid acc: %0.3f, \t test loss: %.3f, \t test acc: %.3f " % (valid_loss, valid_acc, test_loss, test_acc)))
    

if __name__ == "__main__":
    args = get_args()
    train(args)
        



