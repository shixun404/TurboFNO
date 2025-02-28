import torch
import re
import matplotlib.pyplot as plt
def load_data_to_tensor(file_path):
	data_list = []
	with open(file_path, 'r') as file:
		for line in file:
			match = re.search(r'bs=(\d+)\s*, dimX=(\d+)\s*, DY=(\d+)\s*, N=(\d+)\s*, K=(\d+)\s*, TIME=\s*([\d.]+)ms', line)
			if match:
				bs, dimX, DY, N, K, time = map(float, match.groups())
				data_list.append([bs, dimX, DY, N, K, time])
	tensor_data = torch.tensor(data_list, dtype=torch.float32)
	return tensor_data
file_path = ['exp_data/1D_A', 'exp_data/1D_E']
data = []
for file in file_path:
	data.append(load_data_to_tensor(file_path))
plt.figure()
plt.plot(data[0][:, -1] / data[1][:, -1], label='A')