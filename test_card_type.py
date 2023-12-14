from __future__ import print_function
import datetime
import os
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from PIL import ImageFile
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score



#import utils

from sklearn.metrics import confusion_matrix


def test(model, criterion, data_loader, classes):
    n = len(data_loader.dataset)
    #print(n)
    y_pred = np.zeros((n))
    y_true = np.zeros((n))

    index = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            batch_start = time.time()
            output = model(image)
            batch_end = time.time()
            loss = criterion(output, target)

            scope = image.size(0)
            _, preds = torch.max(output, 1)
            correct = torch.sum(preds == target.data)
            loss_ = loss.item() * image.size(0)

            print('[TEST] Epoch: Acc: {:.4f}, Loss: {:.4f}, Inference time:{:.4f}'.format(
                correct.double() / image.size(0),
                loss_ / image.size(0), (batch_end-batch_start)/image.size(0)))

            #print(index, scope)
            y_pred[index:index+scope] = preds.view(-1).cpu().numpy()
            y_true[index:index+scope] = target.data.cpu().numpy()

            #print(y_pred[index:index+scope])
            #print(y_true[index:index+scope])

            index += scope
    return y_pred, y_true

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, accuracy, f1


def main(args):

    testdir = os.path.join(args.data_dir, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Loading test data")
    dataset_test = torchvision.datasets.ImageFolder(
                testdir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,]))

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    classes = torch.load(args.checkpoint)['classes']
    print(classes)
    model = torchvision.models.__dict__[args.model](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    model = nn.DataParallel(model, device_ids=args.device)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    y_pred, y_true = test(model, criterion, test_dataloader, classes)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

    plt.savefig('image\Image')
    plt.show()

    precision, recall, accuracy, f1 = calculate_metrics(y_true, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("F1 score:", f1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-dir', default='E:/codes_py/Larkimas/Data', help='dataset')
    parser.add_argument('--model', default='resnet34', help='model')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=1000, type=int)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--checkpoint', default='checkpoints/resnet34_10_0.9500.pth', help='checkpoint')

    args = parser.parse_args()
    print(args)
    main(args)

