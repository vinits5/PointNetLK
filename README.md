# PointNetLK: Point Cloud Registration using PointNet

Source Code Author: Vinit Sarode

This code is tested using torch 1.3.0+cu92 and torchvision 0.4.1+cu92.

### Code use:
1. Train PointNet:
> python train_classifier.py
2. Train PointNetLK:
> python main.py
3. Test PointNetLK:
> python test.py

### Pretrained Models:
1. best_model: 			PointNetLK pretrained model.
2. best_ptnet_model: 	PointNet_features model.
3. classifier: 			PointNet classification model.

### Requirements:
1. PyTorch
2. ModelNet40 [[Link]](https://modelnet.cs.princeton.edu/)
3. SciPy
4. GPU Requirements: Approximately 6 GB of GPU memory for batch size = 16.

### References:
1. PointNetLK [[paper]](https://arxiv.org/abs/1903.05711)
2. PointNetLK [[github]](https://github.com/hmgoforth/PointNetLK)
3. PCRNet 	  [[github]](https://github.com/vinits5/pcrnet_pytorch)
3. DCP 		  [[github]](https://github.com/WangYueFt/dcp)

### Sample Results:
*Template: Red Point Cloud, Source: Green Point Cloud & Registration Result: Blue Point Cloud*

<p align="center">
	<img src="https://github.com/vinits5/PointNetLK/blob/master/images/sample_result.png" height="300">
</p>