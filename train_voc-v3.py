import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from datasetclass17 import Datasetee17
# D:\Anaconda3\pkgs\torchvision-0.2.2-py_3\site-packages\torchvision\models

def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr*(0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    loss = 0
    for batch_idx, (data, target,fn) in enumerate(train_loader):
        print('batch_idx',batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.softmax(model(data),dim=1)
        criten = nn.MultiLabelSoftMarginLoss(size_average=True)
        loss = criten(output,target)
        loss.backward()
        print('loss,train',loss)
        optimizer.step()
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time.time()-start_time)))
        
def test(args, model, device, test_loader):
    model.eval()
    test_loss=0
    correct=[0]*20
    num=[0]*20
    num_p = 0
    tbs=args.test_batch_size
    print(tbs)
    with torch.no_grad():
        for data, target,_ in test_loader:
            data, target = data.to(device), target.to(device)
            #print(data,len(data))
            #print('tar',target,len(target))
            output = F.softmax(model(data),dim=1)
            #output = torch.clamp(output,0,1)
            #print('out',output)
            #criten = nn.BCELoss(size_average=True)
            criten = nn.MultiLabelSoftMarginLoss(size_average=True)
            test_loss += criten(output, target).item()
            #print('testloss',test_loss)
            output = output.cpu()
            target = target.cpu()
            #print('out',output)
            output = output.numpy()
            threshold=output.sum(axis=1)/10
            #print('thres',threshold,type(threshold))
            #output = output.numpy()
            #print('out_np',output,type(output))
            
            num_p = num_p + len(data)
            for b in range(len(data)):
                #print('outb',output[b])
                #print('theb',threshold[b])
                ToF= output[b]>threshold[b]
                #print('tof',ToF,type(ToF))
                temp = target[b].numpy()
                temp = temp.astype(int)
                num = num+temp
                #print('num',num,type(num))
                #print('tar_b',temp,type(temp))
                correct = correct+(1-(ToF^temp))
    print('testloss',test_loss)
    num = np.array(num)
    correct = np.array(correct)
    correct_rate = correct/num_p
    mAcc = sum(correct_rate)/20
    wAcc = sum(correct_rate*num)/sum(num)

    print('\nTest set: mAcc: {:.4f}, wAcc: {:.4f} \n'.format(mAcc,wAcc))
               
def main():
    # Training configurations
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=48, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='For visualization')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.cuda.set_device(2)
    net = models.resnet101(num_classes=20)
    net=net.to(device)
    
    # data loaders
    print('data loading...\n')
    transform1 = transforms.Compose([
                               transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])
    
    train_loader = torch.utils.data.DataLoader(
        Datasetee17('./PascalVOC/','train.txt',transform=transform1),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        Datasetee17('./PascalVOC/','test.txt',transform=transform1),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # construct optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.999),eps=1e-08)
    
    # train and test
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(args.lr, optimizer, epoch)
        train(args, net, device, train_loader, optimizer, epoch)
        test(args, net, device, test_loader)

    if args.save_model:
        print('Saving model in ./model/checkpoint.pt\n')
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(net.state_dict(),'model/checkpoint.pt')

if __name__ == '__main__':
    main()
    
    
