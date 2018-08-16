#############################################################
from deeplab_large_fov import Net
from utils import *
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import os
import time
import torch.optim as optim

docstr = """Train Deeplab-Large_FOV with augmented PASCAL VOC dataset.Version 1

Usage:
  train_v1.py <list_path> <im_path> <gt_path> [options]
  train_v1.py (-h | --help)
  train_v1.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --batch_size=<int>          batch_size for processing in gpu[default: 20]
  --gpu=<bool>                Which GPU to use[default: 3]
  --init_file=<str>           network inital weights path[default: ]
  --max_iter=<int>            maximum iterations[default: 6000]
  --wt_decay=<float>          wt_decay parameter to use[default: 0.0005]
  --momentum=<float>          momentum parameter to use[default: 0.9]
  --lr=<float>                learning rate to be used[default: 0.001]
  --snapshot_dir=<str>        directory to save snapshots[default: ]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))

    model = Net()
    if(args['--init_file']):
        params = torch.load(args['--init_file'])
        for k, v in params.items():
            if k.endswith('.bias'):
                v.squeeze_()
        model.load_state_dict(params)
        
    max_iter = int(args['--max_iter']) 
    batch_size = int(args['--batch_size'])
    wt_decay = float(args['--wt_decay'])
    momentum = float(args['--momentum'])
    base_lr = float(args['--lr'])
    lr = base_lr
    snapshot_dir = args['--snapshot_dir']
    if not snapshot_dir:
        if(args['--init_file']):
            snapshot_dir = os.path.dirname(args['--init_file'])
        else:
            snapshot_dir = '.'
    gt_path =  args['<gt_path>']
    img_path = args['<im_path>']
    
    img_list = read_file(args['<list_path>'])
    train_epoch = int(max_iter*batch_size/float(len(img_list)))
    data_list = []
    for i in range(train_epoch+1):
        np.random.shuffle(img_list)
        data_list.extend(img_list)
    data_list = data_list[:max_iter*batch_size]
    print(len(data_list))

    model.float()
    model.train()
    model.cuda()
    
    optimizer = optim.SGD([
            {'params': get_parameters(model), 'lr': base_lr,'weight_decay':wt_decay},
            {'params': get_parameters(model,bias=True), 'lr': 2*base_lr,'weight_decay':0},
            {'params': get_parameters(model,final=True), 'lr': base_lr*10,'weight_decay':wt_decay},
            {'params': get_parameters(model,bias=True,final=True), 'lr': base_lr*20,'weight_decay':0}
            ], lr=base_lr, momentum=momentum,weight_decay = wt_decay)
    optimizer.zero_grad()
    criterion = nn.NLLLoss2d()
    for iter, chunk in enumerate(chunker(data_list, batch_size)):

        images, labels = get_data_from_chunk_v2(chunk,gt_path,img_path)
        outputs = model.forward(images)
        labels = torch.squeeze(labels,1).long().cuda()
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter%10 == 0:
            print('Loss at ',iter,' is ',loss.data.cpu().numpy())
        if iter%2000==0 and iter!=0:
            lr = lr *0.1
            optimizer = adjust_learning_rate(optimizer, lr)
            print('learning rate kept at ', lr)
            print('time taken ',time.time()-start_time)
            print('saving snapshot')
            torch.save(model.state_dict(), os.path.join(snapshot_dir, 'deeplab_large_fov_{:d}.pth'.format(iter)))
    print('Last snapshot')
    torch.save(model.state_dict(), os.path.join(snapshot_dir, '/deeplab_large_fov_last.pth'))
    print('Total Time taken',time.time()-start_time)
