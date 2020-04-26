import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import h5py
import transforms3d.euler as t3d

def visualize_result(template, source):
	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source)
	
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	o3d.visualization.draw_geometries([template_, source_])

# PyTorch based Code
def sphere_torch(p):
	# args: p, batch of pointclouds [B x N x 3]
	# returns: mask of visible points in p [B x N]
	p_trans = p + 2
	p_mag = torch.norm(p_trans, 2, 2)
	p_COM = torch.mean(p_trans, 1, keepdim=True)
	p_COM_mag = torch.norm(p_COM, 2, 2) + 0.08 			# Change this value to control amount of data removal.
	mask = p_mag < p_COM_mag
	data = [d[mask[idx]].view(-1,3) for idx, d in enumerate(p)]
	size = min([d.shape[0] for d in data])
	data = [d[:size] for d in data]
	data = torch.stack(data)
	return mask, data

# Numpy based Code
def sphere(data):
	# args: data (batch of pointclouds [B x N x 3])
	# returns: mask of visible points in data [B x N] and partial data
	data_trans = data + 2
	data_mag = np.linalg.norm(data_trans, 2, 2)
	data_COM = np.mean(data_trans, 1, keepdims=True)
	data_COM_mag = np.linalg.norm(data_COM, 2, 2)
	mask = data_mag < data_COM_mag
	data = [d[mask[idx]] for idx, d in enumerate(data)]
	return mask, data
	
if __name__ == '__main__':
	file = h5py.File('data/test_data/templates.h5', 'r')
	data = file.get('templates')
	data_original = np.array(data)[:, :1024]

	R = t3d.euler2mat(np.pi/3, -np.pi/4, 0, 'szyx')
	data = np.matmul(R, data_original[0].T).T + np.array([0.2, 0.3, 0.1])
	data = data.reshape(1,-1,3)


	data = torch.tensor(data).float()
	mask, partial_data = sphere_torch(data)

	data, partial_data = data.numpy(), partial_data.numpy()
	print(partial_data.shape)
	# mask, partial_data = sphere(data)
	visualize_result(data_original[0], partial_data[0])