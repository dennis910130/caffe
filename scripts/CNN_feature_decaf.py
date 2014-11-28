__author__ = 'chensi'
import numpy as np
import sys
caffe_root = '../'
sys.path.insert(0,caffe_root + 'python')
import caffe
import glob
imageset_path = ''
import cPickle
from optparse import OptionParser
import time
import scipy.io as sio
import os.path
import os
from scipy.sparse import csr_matrix


def initial_network_custom(proto_path, model_path):

    net = caffe.Classifier(proto_path,
                           model_path,
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    net.set_phase_test()
    net.set_mode_gpu()
    return net
def initial_network_vgg_center():

    net = caffe.Classifier(caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy_center.prototxt',
                           caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    net.set_phase_test()
    net.set_mode_gpu()
    return net
def initial_network_vgg_ten_crops():

    net = caffe.Classifier(caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt',
                           caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    net.set_phase_test()
    net.set_mode_gpu()
    return net
def initial_network_alex_center():

    net = caffe.Classifier(caffe_root+'models/bvlc_reference_caffenet/deploy_center.prototxt',
                           caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    net.set_phase_test()
    net.set_mode_gpu()
    return net
def initial_network_alex_ten_crops():

    net = caffe.Classifier(caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt',
                           caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    net.set_phase_test()
    net.set_mode_gpu()
    return net
def get_options_parser():
    parser = OptionParser()
    parser.add_option('-i','--input_path',dest='img_input_path')
    parser.add_option('-l','--input_list',dest='img_input_list')
    parser.add_option('-o','--output',dest='feature_output_path',default=None)
    parser.add_option('--layer',dest='layer',default='conv5')
    parser.add_option('--mode',dest='mode',default='custom')
    parser.add_option('--prototxt', dest = 'prototxt_path')
    parser.add_option('--model', dest = 'model')
    parser.add_option('--center', dest = 'center', default = False);
    return parser



def main():
    parser = get_options_parser()

    (options, args) = parser.parse_args()

    with open(options.img_input_list,'r') as f:
        file_names = [options.img_input_path+line.strip() for line in f]
    with open(options.img_input_list,'r') as f:
	file_names_rel = [line.strip() for line in f]
    print 'total_image_detected: ' + str(len(file_names))
    out_path = options.feature_output_path
    if not out_path:
	out_path = options.img_input_path
    file_name, fileExtension = os.path.splitext(file_names[0])

    layer = options.layer
    oversample = False;
    
    if options.mode == 'AlexNetCenter' :
        net = initial_network_alex_center()
    elif options.mode == 'VggNetCenter' : 
        net = initial_network_vgg_center()
    elif options.mode == 'AlexNetTen' :
        net = initial_network_alex_ten_crops()
        oversample = True
    elif options.mode == 'VggNetTen' : 
        net = initial_network_vgg_ten_crops()
        oversample = True
    else :
        net = initial_network_custom(options.prototxt_path, options.model)
        oversample = not options.center
        
    start_time_total = time.time()
    for i in range(len(file_names)):
    #for i in range(18474,30607):
        start_time = time.time()
        print 'extracting the CNN feature, layer: %s, image No. %d/%d' %(layer,i+1,len(file_names))
        net.predict([caffe.io.load_image(file_names[i])],oversample)        
        feature_temp = net.blobs[layer].data
        out_path1 = out_path+file_names_rel[i].replace(fileExtension,'_CNN_'+layer+'_feature'+'.mat')
        try:
            sio.savemat(out_path1,{'CNN_feature':feature_temp})
        except:
            os.makedirs(os.path.dirname(out_path1))
            sio.savemat(out_path1,{'CNN_feature':feature_temp})

        print 'time used:'+str(time.time()-start_time)+'s'
    print 'time used totally:'+ str(time.time()-start_time_total)




if __name__ == '__main__':
    main()





