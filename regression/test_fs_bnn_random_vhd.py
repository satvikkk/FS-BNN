# sparse nonlinear function
import torch
import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0,'/Users/Akanksha Mishra/Documents/GitHub/sbnn/regression/')

from tools import sigmoid
from sparse_bnn_vhd import FeatureSelectionBNN
from sklearn.model_selection import train_test_split

start_time = time.time()
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

np.random.seed(123)
torch.manual_seed(456)

curr_dir = "/Users/Akanksha Mishra/Documents/GitHub/sbnn/regression/"
code = "Feature_selection"

# ------------------------------------------------------------------------------------------------------
# Generate target function
data_size = 4000
test_size = 1000
data_dim = 100
sigma_noise = 1.
rep = 1


# generating training data
# target function
def target(n, p):
    X = np.random.uniform(-1, 1, (n, p))
    epsilon_y = np.random.normal(0, sigma_noise, X.shape[0])
    y = 7 * X[:, 1] / (1 + X[:, 0]**2) + 5 * np.sin(X[:, 2] * X[:, 3]) + 2 * X[:, 4]
    # y = 7 * X[:, 1] / (1 + X[:, 0]**2) + 5 * np.sin(X[:, 2] * X[:, 3]) + 2 * X[:, 4] + torch.tensor(epsilon_y)
    return X, y


# x_train = torch.Tensor(data_size, data_dim).uniform_(-1, 1) * 5
trainsets = []
for i in range(rep):
    x, y = target(data_size + test_size, data_dim)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .8)
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    trainset = [x_train, y_train]
    trainsets.append(trainset)


# x_test = np.random.uniform(-1, 1, (test_size, data_dim))
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)
# y_test = target(x_test, sigma_noise)


# ------------------------------------------------------------------------------------------------------
batch_size = 512
num_batches = data_size / batch_size
learning_rate = torch.tensor(1e-3)
epochs = 5000
hidden_dim = [16,8,4]
L = 3
# total = (data_dim+1) * hidden_dim + (L-1)*((hidden_dim+1) * hidden_dim) + (hidden_dim+1) * 1
# a = np.log(total) + 0.1*((L+1)*np.log(hidden_dim) + np.log(np.sqrt(data_size)*data_dim))
total = (data_dim+1) * hidden_dim[0] + (hidden_dim[0]+1) * hidden_dim[1] + (hidden_dim[1]+1) * hidden_dim[2] + (hidden_dim[2]+1) * 1
a = np.log(total) + 0.1*np.log(hidden_dim[0]) + 0.1*np.log(hidden_dim[1]) + 0.1*np.log(hidden_dim[2]) + np.log(np.sqrt(data_size)*data_dim)
lm = 1/np.exp(a)
phi_prior = torch.tensor(lm)
temp = 0.5

train_MSEs = []
test_MSEs = []
sparse_overalls = []
sparse_overalls2 = []
FNRs = []
FPRs = []
w0 = np.zeros(data_dim)
w0[0:5] = 1.
P = 5
N = data_dim - P
# l1_wtheta = np.zeros(shape=(epochs, data_dim, hidden_dim))

print('Train size:{}, Test size:{}, Total Features:{}, Epochs:{}, Hidden Layers:{}, Hidden Dims:{}'.format(data_size, test_size, data_dim, epochs, L, hidden_dim))
for k in range(rep):

    print('------------ round {} ------------'.format(k))
    # create sparse BNN
    net = FeatureSelectionBNN(data_dim=data_dim, hidden_dim = hidden_dim, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    x_train = trainsets[k][0]
    y_train = trainsets[k][1]

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = []
        log_likelihoods = []
        kls = []
        permutation = torch.randperm(data_size)

        for i in range(0, data_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, nll_MC,kl_MC, _ = net.sample_elbo(
                batch_x, batch_y, 1, temp, phi_prior, num_batches)
            if torch.isnan(loss):
                break
            train_losses.append(loss.item())
            log_likelihoods.append(nll_MC.item())
            kls.append(kl_MC.item())
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            one1_w = (net.l1.w != 0).float()
            one1_b = (net.l1.b != 0).float()
            sparsity = torch.sum(one1_w) + torch.sum(one1_b)
            print('Epoch {}, Train_Loss: {}, Log_likelihood: {},KL: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), np.mean(log_likelihoods), np.mean(kls), phi_prior,
                                                                                 sparsity))
    print('Finished Training')

    # sparsity level
    one1_w = (sigmoid(net.l1.w_theta)).float()
    one1_b = (sigmoid(net.l1.b_theta) > 0.5).float()
    sparse_overall = torch.sum(one1_w) + torch.sum(one1_b)
    sparse_overalls.append(sparse_overall)
    sparse_overall2 = torch.sum(sigmoid(net.l1.w_theta)) + torch.sum(sigmoid(net.l1.b_theta))
    sparse_overalls2.append(sparse_overall2)
    torch.set_printoptions(profile="full")
    print("\n", "----------- Network Sparsity -----------")
    print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    # print('l1 w Edges: {}'.format(one1_w))
    p = torch.mean(one1_w, axis=1)
    sorted, indices = torch.sort(p,0, descending=True)
    print('features selected in the first layer: {}'.format(indices[0:10]))
    torch.save(indices, 'indices_tensor_fsbnn.pt')

    # t_np = one1_w.numpy()
    # df = pd.DataFrame(t_np)
    # df.to_csv("random_fs_sparse_one1w", index = False)

    print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
    print('l1 b Edges: {}'.format(one1_b))

    # prediction
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    _, _, _, pred = net.sample_elbo(x_train, y_train, 30,
                              temp, phi_prior, num_batches)
    pred = pred.mean(dim=0)
    train_mse = torch.sqrt(torch.mean((pred - y_train) ** 2))
    train_MSEs.append(train_mse.data)

    print("----------- Training -----------")
    print('y_train: {}'.format(y_train[0:20]))
    print('pred_train: {}'.format(pred[0:20]))
    print('MSE_train: {}'.format(train_mse))

    # ------------------------------------------------------------------------------------------------------
    print("\n", "----------- Testing -----------")
    # testing
    # prediction
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    _, _, _, pred2 = net.sample_elbo(
        x_test, y_test, 30, temp, phi_prior, num_batches)
    pred2 = pred2.mean(dim=0)
    test_mse = torch.sqrt(torch.mean((pred2 - y_test) ** 2))
    test_MSEs.append(test_mse.data)

    print('y_test: {}'.format(y_test[0:20]))
    print('pred_test: {}'.format(pred2[0:20]))
    print('MSE_test: {}'.format(test_mse))
    print("\n")
    one = torch.transpose(one1_w, 0, 1)
    w = (torch.sum(one, dim=0) > 0).float()
    w = w.detach().numpy()

    FN = np.sum((w - w0) < 0)
    FP = np.sum((w - w0) > 0)
    FNR = FN / P
    FPR = FP / N
    FNRs.append(FNR)
    FPRs.append(FPR)

# print(l1_wtheta.shape)
# np.save(f"{curr_dir}/l1_wtheta_fsbnn", l1_wtheta)

train_MSE = torch.tensor(train_MSEs)
test_MSE = torch.tensor(test_MSEs)
sparse_overalls = torch.tensor(sparse_overalls)
sparse_overalls2 = torch.tensor(sparse_overalls2)
FNRs = torch.tensor(FNRs)
FPRs = torch.tensor(FPRs)

print("\n", "----------- Summary -----------")
print('MSE_train: {}'.format(torch.mean(train_MSE)))
print('MSE_train_sd: {}'.format(torch.std(train_MSE)))
print('MSE_test: {}'.format(torch.mean(test_MSE)))
print('MSE_test_sd: {}'.format(torch.std(test_MSE)))
print('sparsity: {}'.format(torch.mean(sparse_overalls)))
print('sparsity2: {}'.format(torch.mean(sparse_overalls2)))
print('FNR: {}'.format(torch.mean(FNRs)))
print('FNR sd: {}'.format(torch.std(FNRs)))
print('FPR: {}'.format(torch.mean(FPRs)))
print('FPR sd: {}'.format(torch.std(FPRs)))


print('sparsity all: {}'.format(sparse_overalls))
print('sparsity all 2: {}'.format(sparse_overalls2))
print('MSE_train all: {}'.format(train_MSE))
print('MSE_test all: {}'.format(test_MSE))
print('FNRs: {}'.format(FNRs))
print('FPRs: {}'.format(FPRs))
