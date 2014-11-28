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

def get_options_parser():
	parser = OptionParser()
	parser.add_option('-i','--input_path',dest='input_model')
	parser.add_option('-f','--fc_prototxt',dest='fc_proto')
	parser.add_option('-c','--conv_prototxt',dest='conv_proto')
	parser.add_option('-o','--output',dest='output_model')
	return parser
	
def main():
	parser = get_options_parser()
	(options, args) = parser.parse_args()
	net = caffe.Net(options.fc_proto,options.input_model)
	params = ['fc6','fc7','fc8']
	fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
	
	net_full_conv = caffe.Net(options.conv_proto,options.input_model)
	params_full_conv = ['fc6-conv','fc7-conv','fc8-conv']
	conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
	
	for pr, pr_conv in zip(params, params_full_conv):
		conv_params[pr_conv][1][...] = fc_params[pr][1]
	
	for pr, pr_conv in zip(params, params_full_conv):
		out, in_, h, w = conv_params[pr_conv][0].shape
		W = fc_params[pr][0].reshape((out, in_, h, w))
		conv_params[pr_conv][0][...] = W
	
	try:
		net_full_conv.save(options.output_model)
	except:
		os.makedirs(os.path.dirname(options.output_model))
		net_full_conv.save(options.output_model)
		
if __name__ == '__main__':
	main()
