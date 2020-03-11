import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import se3, so3, invmat, data_utils

def flatten(x):
	return x.view(x.size(0), -1)

def symfn_max(x):
	# [B, K, N] -> [B, K, 1]
	a = torch.nn.functional.max_pool1d(x, x.size(-1))
	return a

def symfn_avg(x):
	a = torch.nn.functional.avg_pool1d(x, x.size(-1))
	return a


class PointNet_features(torch.nn.Module):
	def __init__(self, emb_dims=1024, use_tnet=False, symfn='max'):
		super().__init__()
		self.h1 = [torch.nn.Conv1d(3, 64, 1), torch.nn.BatchNorm1d(64),
				   torch.nn.ReLU(),
				   torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64),
				   torch.nn.ReLU()]
		self.h1 = torch.nn.Sequential(*self.h1)

		self.h2 = [torch.nn.Conv1d(64, 64, 1), torch.nn.BatchNorm1d(64),
				   torch.nn.ReLU(),
				   torch.nn.Conv1d(64, 128, 1), torch.nn.BatchNorm1d(128),
				   torch.nn.ReLU(),
				   torch.nn.Conv1d(128, emb_dims, 1), torch.nn.BatchNorm1d(emb_dims),
				   torch.nn.ReLU()]
		self.h2 = torch.nn.Sequential(*self.h2)

		if symfn == 'max': self.sy = symfn_max
		elif symfn == 'avg': self.sy = symfn_avg

	def forward(self, points):
		""" points -> features
			[B, N, 3] -> [B, K]
		"""
		x = points.transpose(1, 2) # [B, 3, N]
		x = self.h1(x)
		x = self.h2(x)
		x = flatten(self.sy(x))
		return x


class PointNet_classifier(torch.nn.Module):
	def __init__(self, ptnet=PointNet_features(), num_c=40, emb_dims=1024):
		super().__init__()
		self.ptnet = ptnet
		list_layers = [torch.nn.Linear(emb_dims, 512), torch.nn.BatchNorm1d(512),
					   torch.nn.ReLU(),
					   torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256),
					   torch.nn.ReLU(),
					   torch.nn.Linear(256, num_c)]
		self.classifier = torch.nn.Sequential(*list_layers)

	def forward(self, points):
		feat = self.ptnet(points)
		out = self.classifier(feat)
		return out

	def loss(self, out, target, w=0.001):
		loss = torch.nn.functional.nll_loss(
			torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)
		return loss


class PointLK(nn.Module):
	def __init__(self, ptnet=PointNet_features(), delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max'):
		super().__init__()
		self.ptnet = ptnet
		self.inverse = invmat.InvMatrix.apply
		self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
		self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

		w1, w2, w3, v1, v2, v3 = delta, delta, delta, delta, delta, delta
		twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
		self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

		# results
		self.last_err = None
		self.g_series = None # for debug purpose
		self.prev_r = None
		self.g = None # estimation result
		self.itr = 0
		self.xtol = xtol
		self.p0_zero_mean = p0_zero_mean
		self.p1_zero_mean = p1_zero_mean

	def forward(self, template, source, maxiter=10):
		template, source, template_mean, source_mean = data_utils.preprocess_data(template, source, 
																			 self.p0_zero_mean, self.p1_zero_mean)

		result = self.iclk(template, source, maxiter)
		result = data_utils.postprocess_data(result, template, source, template_mean, source_mean, 
											 self.p0_zero_mean, self.p1_zero_mean)
		return result

	def iclk(self, template, source, maxiter):
		batch_size = template.size(0)

		est_T0 = torch.eye(4).to(template).view(1, 4, 4).expand(template.size(0), 4, 4).contiguous()
		est_T = est_T0
		self.est_T_series = torch.zeros(maxiter+1, *est_T0.size(), dtype=est_T0.dtype)
		self.est_T_series[0] = est_T0.clone()

		training = self.handle_batchNorm(template, source)

		# re-calc. with current modules
		template_features = self.ptnet(template) # [B, N, 3] -> [B, K]

		# approx. J by finite difference
		dt = self.dt.to(template).expand(batch_size, 6)
		J = self.approx_Jic(template, template_features, dt)

		self.last_err = None
		pinv = self.compute_inverse_jacobian(J, template_features, source)

		itr = 0
		r = None
		for itr in range(maxiter):
			self.prev_r = r
			transformed_source = self.transform(est_T.unsqueeze(1), source) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
			source_features = self.ptnet(transformed_source) # [B, N, 3] -> [B, K]
			r = source_features - template_features

			pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

			check = pose.norm(p=2, dim=1, keepdim=True).max()
			if float(check) < self.xtol:
				if itr == 0:
					self.last_err = 0 # no update.
				break

			est_T = self.update(est_T, pose)
			self.est_T_series[itr+1] = est_T.clone()

		rep = len(range(itr, maxiter))
		self.est_T_series[(itr+1):] = est_T.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

		self.ptnet.train(training)
		self.est_T = est_T

		result = {'est_R': est_T[:,0:3,0:3],
				  'est_t': est_T[:,0:3,3],
				  'est_T': est_T,
				  'r': r,
				  'transformed_source': self.transform(est_T.unsqueeze(1), source),
				  'itr': itr+1,
				  'est_T_series': self.est_T_series}
		
		return result

	def update(self, g, dx):
		# [B, 4, 4] x [B, 6] -> [B, 4, 4]
		dg = self.exp(dx)
		return dg.matmul(g)

	def approx_Jic(self, template, template_features, dt):
		# p0: [B, N, 3], Variable
		# f0: [B, K], corresponding feature vector
		# dt: [B, 6], Variable
		# Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]

		batch_size = template.size(0)
		num_points = template.size(1)

		# compute transforms
		transf = torch.zeros(batch_size, 6, 4, 4).to(template)
		for b in range(template.size(0)):
			d = torch.diag(dt[b, :]) # [6, 6]
			D = self.exp(-d) # [6, 4, 4]
			transf[b, :, :, :] = D[:, :, :]
		transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
		p = self.transform(transf, template.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

		#f0 = self.ptnet(p0).unsqueeze(-1) # [B, K, 1]
		template_features = template_features.unsqueeze(-1) # [B, K, 1]
		f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

		df = template_features - f # [B, K, 6]
		J = df / dt.unsqueeze(1)

		return J

	def compute_inverse_jacobian(self, J, template_features, source):
		# compute pinv(J) to solve J*x = -r
		try:
			Jt = J.transpose(1, 2) # [B, 6, K]
			H = Jt.bmm(J) # [B, 6, 6]
			B = self.inverse(H)
			pinv = B.bmm(Jt) # [B, 6, K]
			return pinv
		except RuntimeError as err:
			
			self.last_err = err
			g = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
			
			source_features = self.ptnet(source) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.ptnet.train(training)
			return {}

	def handle_batchNorm(self, template, source):
		training = self.ptnet.training
		if training:
			# first, update BatchNorm modules
			template_features, source_features = self.ptnet(template), self.ptnet(source)
		self.ptnet.eval()   # and fix them.
		return training

if __name__ == '__main__':
	pnlk = PointNet_classifier()
	template = torch.rand(10, 1024, 3)
	result = pnlk(template)
	import ipdb; ipdb.set_trace()

#EOF