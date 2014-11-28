__author__ = 'chensi'
import numpy as np
import sys
caffe_root = '../'
sys.path.insert(0,caffe_root + 'python')
import caffe
import glob
from optparse import OptionParser
import time
import scipy.io as sio
import os.path
import os
from scipy.sparse import csr_matrix

def initial_network_vgg():
    net = caffe.Net(caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_conv.prototxt',
                           caffe_root+'models/3785162f95cd2d5fee77/vgg_19_conv_model.caffemodel')

    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data',np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'),'channel')
    net.set_channel_swap('data',(2,1,0))
    net.set_raw_scale('data',255.0)
    return net
    
def initial_network_alex():
    net = caffe.Net(caffe_root+'examples/imagenet/bvlc_caffenet_full_conv.prototxt',
                           caffe_root+'models/bvlc_reference_caffenet/bvlc_caffenet_full_conv.caffemodel')

    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data',np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'),'channel')
    net.set_channel_swap('data',(2,1,0))
    net.set_raw_scale('data',255.0)
    return net

def initial_network_custom(prototxt_path, model_path):
    net = caffe.Net(prototxt_path,
                           model_path)

    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data',np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'),'channel')
    net.set_channel_swap('data',(2,1,0))
    net.set_raw_scale('data',255.0)
    return net
    
def get_options_parser():
    parser = OptionParser()
    parser.add_option('-i','--input_path',dest='img_input_path')
    parser.add_option('-l','--input_list',dest='img_input_list')
    parser.add_option('-o','--output',dest='feature_output_path',default=None)
    parser.add_option('--layer',dest='layer',default='conv5')
    parser.add_option('--mode',dest='mode',default='custom')
    parser.add_option('-s',dest='small_size',default=None)
    parser.add_option('--prototxt', dest='prototxt', default=None)
    parser.add_option('--model', dest='conv_model', default=None)
    return parser

def calculate_location_table():
    print 'calculating locations for each layer...'

    print 'done'
    return {'conv1':[4,6],'conv2':[8,10],'conv3':[16,18],'conv4':[16,18],'conv5':[16,18],
            'fc6-conv':[32,112],'fc7-conv':[32,112],'fc8-conv':[32,112],'prob':[32,112]}


def save_sparse_file(csr_matrix,filename):
    data = csr_matrix.data
    rows, cols = csr_matrix.nonzero()
    f = open(filename,'w')
    
    
    for i in range(len(data)):
        f.write(str(rows[i]+1)+' ')
        f.write(str(cols[i]+1)+' ')
        f.write(str(data[i])+'\n')
        
    f.close()

    




def main():
    parser = get_options_parser()

    (options, args) = parser.parse_args()

    with open(options.img_input_list,'r') as f:
        file_names = [options.img_input_path+line.strip() for line in f]
    with open(options.img_input_list,'r') as f:
	file_names_rel = [line.strip() for line in f]
    print 'total_image_detected: ' + str(len(file_names))
    out_path = options.feature_output_path
    small_size = options.small_size
    if not out_path:
	out_path = options.img_input_path
    file_name, fileExtension = os.path.splitext(file_names[0])
    if small_size:
        small_size = float(small_size)
    layer = options.layer
    if options.mode == 'AlexNet' :
        net = initial_network_alex()
    elif options.mode == 'VggNet':
        net = initial_network_vgg()
    else :
        net = initial_network_custom(options.prototxt, options.conv_model)
        
         
    start_time_total = time.time()
    for i in range(len(file_names)):
        start_time = time.time()
        print 'extracting the CNN feature, layer: %s, image No. %d/%d' %(layer,i+1,len(file_names))
        net.extract_features('data',file_names[i],small_size)
        feature  = net.blobs[layer].data
        
        out_path1 = out_path+file_names_rel[i].replace(fileExtension,'_CNN_'+layer+'_feature'+'.mat')
        try:
            sio.savemat(out_path1,{'CNN_feature':feature})
        except:
            os.makedirs(os.path.dirname(out_path1))
            sio.savemat(out_path1,{'CNN_feature':feature})

        print 'time used:'+str(time.time()-start_time)+'s'
    print 'time used totally:'+ str(time.time()-start_time_total)




if __name__ == '__main__':
    main()
