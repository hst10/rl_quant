import time
import torch
import torch.nn as nn
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from pytorch_mvm_class import *
# from resnet18_imnet_mvm import *
from resnet18_mvm import *
from vgg16_mvm import *

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names = default_model_names

print('support models: ', default_model_names)


def accuracy(output, target, training, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if training:
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def test_mvm(mvm_model):
    global best_acc
    flag = True
    training = False
    mvm_model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(testloader):
        
        curr_time = time.time()

        print("batch_idx = ", batch_idx)
        target = target.cuda()
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        if batch_idx>=1 and batch_idx<2:                            
            output = mvm_model(data_var)
            loss= criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            if flag == True:
                if batch_idx % 1 == 0:
                    print('[{0}/{1}({2:.0f}%)]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                           loss=losses, top1=top1, top5=top5))
            else:
                if batch_idx % 1 == 0:
                   print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                           loss=losses, top1=top1, top5=top5))  

        print(time.time() - curr_time)

        if batch_idx == 1:
            break

    acc = top1.avg
 
    # if acc > best_acc:
    #     best_acc = acc
    #     save_state(model, best_acc)
    

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg

class QuantEvaluator:
    def __init__(self, batch_size=1, workers=4, dnn_model=vgg16):
        self.data_path = '/home/shuang91/data/imagenet/'
        traindir = os.path.join(self.data_path, 'train')
        print(traindir)
        valdir = os.path.join(self.data_path, 'val')
        print(valdir)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()

        self.test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,drop_last=True)

        self.iter_test = iter(self.testloader)

    def explore(self):
        (data, target) = next(self.iter_test)
        print("data = ", data)
        print("target = ", target)

    def evaluate(self, quant_cfg):

        model_mvm = None
        if dnn_model is vgg16:
            assert(len(quant_cfg) == 16)
            conv_quant_cfg = quant_cfg[:13]
            line_quant_cfg = quant_cfg[-3:]
            model_mvm = dnn_model(pretrained=True, 
                                  quant_cfg=conv_quant_cfg, 
                                  linear_quant=line_quant_cfg)
        else:
            model_mvm = dnn_model(pretrained=True, 
                                  quant_cfg=quant_cfg)

        model_mvm.cuda()

        (data, target) = next(self.iter_test)
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        output = mvm_model(data_var)
        loss= criterion(output, target_var)

        return loss.data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture:' + ' | '.join(model_names) + ' (default: resnet18)')

    parser.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')

    parser.add_argument('--data', action='store', default='/home/shuang91/data/imagenet/',
            help='dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 8)')

    args = parser.parse_args()

    # model = models.__dict__[args.arch](pretrained=True)

    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    # pretrained_model = model.state_dict()
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    cudnn.benchmark = True

    # model.load_state_dict(pretrained_model, strict=True)

    # num_conv2d = 0
    # num_bn     = 0
    # num_linear = 0
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         num_conv2d += 1
    #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         num_bn += 1
    #     elif isinstance(m, nn.Linear):
    #         num_linear += 1
    # print("num_conv2d = ", num_conv2d)
    # print("num_bn = ", num_bn)
    # print("num_linear = ", num_linear)

    # model_resnet18_mvm = resnet18_mvm(pretrained=True)
    # model_vgg16_mvm = vgg16(pretrained=True)
    model_vgg16_mvm = vgg16(pretrained=True, quant_cfg=[(8,8,32,24,32,24,9,16,24)]*13, linear_quant=[(8,8,32,24,32,24,9,16,24)]*3)
    # model_vgg16 = vgg16(pretrained=True)

    # num_conv2d = 0
    # num_bn     = 0
    # num_linear = 0
    # for m in model_resnet18_mvm.modules():
    #     if isinstance(m, Conv2d_mvm):
    #         num_conv2d += 1
    #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         num_bn += 1
    #     elif isinstance(m, Linear_mvm):
    #         num_linear += 1
    # print("num_conv2d = ", num_conv2d)
    # print("num_bn = ", num_bn)
    # print("num_linear = ", num_linear)

    # model.cuda()
    # model_resnet18_mvm.cuda()
    model_vgg16_mvm.cuda()

    traindir = os.path.join(args.data, 'train')
    print(traindir)
    valdir = os.path.join(args.data, 'val')
    print(valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    torch_seed = 40#torch.initial_seed()
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.batch_size)))

    # print("test_dataset len = ",  test_dataset)
    # print("test_dataset_sub len = ",  len(test_dataset_sub))
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=True)

    print('Data Loading done')
    criterion = nn.CrossEntropyLoss()

    test_mvm(model_vgg16_mvm)
