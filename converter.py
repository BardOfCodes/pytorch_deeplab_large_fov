#############################################################
from deeplab_large_fov import Net
import torch
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import sys
docstr = """Converts Deeplab Large FOV caffemodel file to .pth OrderedDictionary.

Usage:
  converter.py <caffe_path> <model_caffemodel> <model_prototxt> [options]

Options:
  -h --help     Show this screen.
  --version     Show version.
  --gpu=<bool>                Which GPU to use[default: 3]
  --save_dir=<str>            directory to save snapshots[default: ]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))
    caffe_path = args['<caffe_path>']
    sys.path.insert(0, caffe_path + 'python')
    import caffe
    model_wts_path = args['<model_caffemodel>']
    model_def_path = args['<model_prototxt>']
    caffe_model = caffe.Net(model_def_path, model_wts_path)
    
    pytorch_model = Net()
    dict = pytorch_model.state_dict()
    for i in range(len(caffe_model.params.keys())):
        caffe_name = caffe_model.params.keys()[i]
        
        wts = caffe_model.params[caffe_name][0].data
        pytorch_name = dict.keys()[2*i]
        dict[pytorch_name] = torch.from_numpy(wts)
        
        bias = caffe_model.params[caffe_name][1].data
        pytorch_name = dict.keys()[2*i+1]
        dict[pytorch_name] = torch.from_numpy(bias)
    torch.save(dict, args['--save_dir']+'deeplab_vgg_init.pth')
    
