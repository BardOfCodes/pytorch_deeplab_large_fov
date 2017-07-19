from deeplab_large_fov import Net
from utils import *
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import skimage.io as sio

docstr = """Test Deeplab-Large_FOV for PASCAL VOC dataset.Saves the output predictions for each image. Version 1.

Usage:
  test.py <model_path> <im_path> <im_list> <save_path> [options]
  test.py [options]

Options:
  -h --help     Show this screen.
  --version     Show version.save snapshots[default: False]
  --gpu=<bool>  Which GPU to use[default: 2]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))
    im_path = args['<im_path>']

    model = Net()
    model.load_state_dict(torch.load(args['<model_path>']))
    batch_size = 1
    
    model.float()
    model.eval()
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    f = open(args['<im_list>'])
    file_list = []
    for l in f.readlines():
        file_list.append(l[:-1])
    max_iter = len(file_list)
    for iter, chunk in enumerate(chunker(file_list, batch_size)):
        images= get_test_data_from_chunk_v2(chunk,im_path)
        outputs = model.forward_test(images)
        x = outputs.cpu().data.numpy()[0]
        x = np.argmax(x,0)
        image = cv2.imread(im_path+file_list[iter]+'.jpg')
        image = image[:,:,0]
        # image[image ==255] = 0
        height = image.shape[0]
        width = image.shape[1]
        sio.imsave(args['<save_path>']+file_list[iter]+'.png',x[:height,:width])
        if(iter%50 == 0):
            print('Current Iteration ',iter,' of max_iter ',max_iter)
print('Saved all images in ',time.time() - start_time, ' seconds.')