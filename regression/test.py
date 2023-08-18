import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

hidden_dim=[4,8]
print(hidden_dim)
print(len(hidden_dim))
print(hidden_dim[len(hidden_dim)-1])
for i in range(len(hidden_dim)):
    l+i = 22

# arr = np.load('l1_wtheta_sbnn.npy')
# epochs = arr.shape[0]
# features = arr.shape[1]
# hidden_dim = arr.shape[2]
# curr_dir = "/Users/Akanksha Mishra/Documents/GitHub/sbnn/regression/"

# plt.figure(figsize = (10,15))
# plt.subplot(2,3,1)
# plt.plot(arr[:,0,0], color = 'tab:brown', label = 'Hidden Dim 0')
# plt.plot(arr[:,0,1], color = 'tab:cyan', label = 'Hidden Dim 1')
# plt.plot(arr[:,0,2], color = 'tab:blue', label = 'Hidden Dim 2')
# plt.plot(arr[:,0,3], color = 'tab:green', label = 'Hidden Dim 3')
# plt.plot(arr[:,0,4], color = 'tab:purple', label = 'Hidden Dim 4')
# plt.plot(arr[:,0,5], color = 'tab:orange', label = 'Hidden Dim 5')
# plt.plot(arr[:,0,6], color = 'tab:red', label = 'Hidden Dim 6')
# plt.xlabel('Iterations/Epochs')
# plt.ylabel('Weights')
# plt.title('Feature 0')
# plt.legend()

# plt.subplot(2,3,2)
# plt.plot(arr[:,1,0], color = 'tab:brown', label = 'Hidden Dim 0')
# plt.plot(arr[:,1,1], color = 'tab:cyan', label = 'Hidden Dim 1')
# plt.plot(arr[:,1,2], color = 'tab:blue', label = 'Hidden Dim 2')
# plt.plot(arr[:,1,3], color = 'tab:green', label = 'Hidden Dim 3')
# plt.plot(arr[:,1,4], color = 'tab:purple', label = 'Hidden Dim 4')
# plt.plot(arr[:,1,5], color = 'tab:orange', label = 'Hidden Dim 5')
# plt.plot(arr[:,1,6], color = 'tab:red', label = 'Hidden Dim 6')
# plt.xlabel('Iterations/Epochs')
# plt.ylabel('Weights')
# plt.title('Feature 1')
# plt.legend()

# plt.subplot(2,3,3)
# plt.plot(arr[:,2,0], color = 'tab:brown', label = 'Hidden Dim 0')
# plt.plot(arr[:,2,1], color = 'tab:cyan', label = 'Hidden Dim 1')
# plt.plot(arr[:,2,2], color = 'tab:blue', label = 'Hidden Dim 2')
# plt.plot(arr[:,2,3], color = 'tab:green', label = 'Hidden Dim 3')
# plt.plot(arr[:,2,4], color = 'tab:purple', label = 'Hidden Dim 4')
# plt.plot(arr[:,2,5], color = 'tab:orange', label = 'Hidden Dim 5')
# plt.plot(arr[:,2,6], color = 'tab:red', label = 'Hidden Dim 6')
# plt.xlabel('Iterations/Epochs')
# plt.ylabel('Weights')
# plt.title('Feature 2')
# plt.legend()

# plt.subplot(2,3,4)
# plt.plot(arr[:,3,0], color = 'tab:brown', label = 'Hidden Dim 0')
# plt.plot(arr[:,3,1], color = 'tab:cyan', label = 'Hidden Dim 1')
# plt.plot(arr[:,3,2], color = 'tab:blue', label = 'Hidden Dim 2')
# plt.plot(arr[:,3,3], color = 'tab:green', label = 'Hidden Dim 3')
# plt.plot(arr[:,3,4], color = 'tab:purple', label = 'Hidden Dim 4')
# plt.plot(arr[:,3,5], color = 'tab:orange', label = 'Hidden Dim 5')
# plt.plot(arr[:,3,6], color = 'tab:red', label = 'Hidden Dim 6')
# plt.xlabel('Iterations/Epochs')
# plt.ylabel('Weights')
# plt.title('Feature 3')
# plt.legend()

# plt.subplot(2,3,5)
# plt.plot(arr[:,4,0], color = 'tab:brown', label = 'Hidden Dim 0')
# plt.plot(arr[:,4,1], color = 'tab:cyan', label = 'Hidden Dim 1')
# plt.plot(arr[:,4,2], color = 'tab:blue', label = 'Hidden Dim 2')
# plt.plot(arr[:,4,3], color = 'tab:green', label = 'Hidden Dim 3')
# plt.plot(arr[:,4,4], color = 'tab:purple', label = 'Hidden Dim 4')
# plt.plot(arr[:,4,5], color = 'tab:orange', label = 'Hidden Dim 5')
# plt.plot(arr[:,4,6], color = 'tab:red', label = 'Hidden Dim 6')
# plt.xlabel('Iterations/Epochs')
# plt.ylabel('Weights')
# plt.title('Feature 4')
# plt.legend()

# plt.savefig(f"l1_wtheta_sbnn.png")

# plt.show()