import open3d as o3d
import argparse
import os
import sys
import numpy as np
import torch
import torchvision
from model import PointLK, PointNet_features
import data

def visualize_result(template, source, est_T):
	template, source, est_T = template[0], source[0], est_T[0]
	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]

	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	transformed_source_ = o3d.geometry.PointCloud()
	
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
	
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	transformed_source_.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([template_, source_, transformed_source_])

# Read names of files in given data_dictionary.
def read_files(data_dict):
	with open(os.path.join('data',data_dict,'files.txt')) as file:
		files = file.readlines()
		files = [x.split()[0] for x in files]
	return files[0]

# Read data from h5 file and return as templates.
def read_h5(file_name):
	import h5py
	f = h5py.File(file_name, 'r')
	templates = np.array(f.get('templates'))
	f.close()
	return templates

# Main function to load data and return as templates array.
def loadData(data_dict):
	files = read_files(data_dict)	# Read file names.
	print(files)
	templates = read_h5(files)		# Read templates from h5 file using given file_name.
	return templates

def read_data(args):
	# template:		Nx3 (torch.Tensor)
	# source: 		Nx3 (torch.Tensor)

	# Change this variable to change category of point cloud.
	test_idx = 0

	template = loadData('test_data')					# 100x2048x3 (Data read from folder "data/test_data/templates.h5")
	template = torch.tensor(template)[test_idx]			# 2048x3

	rigid_transform = data.RandomTransformSE3(args.mag, mag_randomly=True)
	source = rigid_transform(template)					# 2048x3

	return template, source

def test(args, model, data):
	model.eval()
	template, source = data

	template = template.view(1,-1,3)
	source = source.view(1,-1,3)
	template = template.to(args.device)
	source = source.to(args.device)

	result = model(template, source)
	est_T = result['est_T']
	r = result['r']

	visualize_result(template.detach().cpu().numpy(), source.detach().cpu().numpy(), est_T.detach().cpu().numpy())

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')

	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--mag', default=0.8, type=float,
						metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('-c', '--categoryfile', type=str, default='./sampledata/modelnet40_half1.txt',
						metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')

	# settings for PointNet
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for LK
	parser.add_argument('--max_iter', default=10, type=int,
						metavar='N', help='max-iter on LK. (default: 10)')
	parser.add_argument('--delta', default=1.0e-2, type=float,
						metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
	parser.add_argument('--learn_delta', dest='learn_delta', action='store_true',
						help='flag for training step size delta')

	# settings for on training
	parser.add_argument('--pretrained', default='pretrained/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()

	torch.backends.cudnn.deterministic = True

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNetLK Model.
	ptnet = PointNet_features(emb_dims=args.emb_dims, symfn=args.symfn)
	model = PointLK(ptnet=ptnet)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	data = read_data(args)
	test(args, model, data)

if __name__ == '__main__':
    main()