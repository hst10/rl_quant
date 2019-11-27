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
    def __init__(self, batch_size=16, workers=4, dnn_model=vgg16):
        self.dnn_model = dnn_model
        self.model_golden = self.dnn_model(pretrained=True).cuda()

        self.global_index = 0

        self.data_path = '/home/shuang91/data/imagenet/'
        traindir = os.path.join(self.data_path, 'train')
        print(traindir)
        valdir = os.path.join(self.data_path, 'val')
        print(valdir)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        cudnn.benchmark = True

        print("BATCH SIZE === ", batch_size)

        self.criterion = nn.CrossEntropyLoss()

        self.test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
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

    def evaluate(self, quant_cfg=None, quiet=False):

        self.global_index += 1

        log_file = open("./logs/"+str(self.global_index)+".txt", "w")

        t_start = time.time()

        print(quant_cfg)
        log_file.write(str(quant_cfg)+'\n')


        print("AAAAAAAAAAAAAAAAAAAAAA")

        model_mvm = None
        if self.dnn_model is vgg16 and quant_cfg is not None:
            assert(len(quant_cfg) == 13)
            conv_quant_cfg = quant_cfg[:13]
            # line_quant_cfg = quant_cfg[-3:]
            line_quant_cfg = None
            model_mvm = self.dnn_model(pretrained=True, 
                                  quant_cfg=conv_quant_cfg, 
                                  linear_quant=line_quant_cfg)
        else:
            model_mvm = self.dnn_model(pretrained=True, 
                                  quant_cfg=quant_cfg)

        print("BBBBBBBBBBBBBBBBBBBBBBBBBBB")

        model_mvm.cuda()

        (data, target) = next(self.iter_test)
        data_var = torch.autograd.Variable(data.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        output_golden = self.model_golden(data_var)
        loss_golden   = self.criterion(output_golden, target_var)

        # print("output_golden = ", output_golden)
        # print("target_var = ", target_var.item())

        print("CCCCCCCCCCCCCCCCCCCCCCCCCCC")

        if not quiet:
            print("Original model loss = ", loss_golden.data.item())
            log_file.write(str(loss_golden.data.item())+'\n')

        output = model_mvm(data_var)
        loss   = self.criterion(output, target_var)
        # print("output = ", output)

        if not quiet:
            print("Quantized model loss = ", loss.data.item())
            print("Evaluation time = ", time.time() - t_start)
            log_file.write(str(loss.data.item())+'\n')
            log_file.write(str(time.time() - t_start)+'\n')

        log_file.close()

        return loss.data.item(), loss_golden.data.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                         metavar='N', help='mini-batch size (default: 16)')

    args = parser.parse_args()

    QE = QuantEvaluator(batch_size=args.batch_size)
    cfg = [[1, 1, 14, 8, 10, 4, 9, 12, 8], \
           [1, 1, 12, 6, 8, 4, 9, 14, 8], \
           [1, 1, 10, 4, 14, 8, 6, 12, 6], \
           [1, 1, 14, 8, 10, 6, 6, 10, 6], \
           [1, 1, 14, 8, 14, 8, 7, 10, 6], \
           [1, 1, 10, 6, 14, 8, 5, 12, 6], \
           [1, 1, 14, 8, 12, 6, 5, 12, 6], \
           [1, 1, 12, 8, 14, 8, 6, 12, 8], \
           [1, 1, 14, 8, 8, 4, 8, 10, 4], \
           [1, 1, 12, 6, 12, 6, 5, 10, 4], \
           [1, 1, 14, 8, 10, 6, 9, 14, 8], \
           [1, 1, 10, 6, 12, 6, 5, 10, 6], \
           [1, 1, 14, 8, 12, 6, 8, 10, 4]]
    loss = QE.evaluate(quant_cfg=cfg)
    # loss = QE.evaluate(quant_cfg=[(2,1,8,4,8,4,9,8,4)]*13)
    # loss = QE.evaluate()
    print(loss)

