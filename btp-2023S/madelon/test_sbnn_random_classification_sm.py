# Added 12 August 2023
# Author: Akanksha Mishra
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0,'/Users/Akanksha Mishra/Documents/GitHub/sbnn/btp-2023S/')

from tools import sigmoid
from sparse_bnn_classification import SparseBNNClassification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

start_time = time.time()
torch.set_default_dtype(torch.float64)
if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
print(device)

curr_dir = "/Users/Akanksha Mishra/Documents/GitHub/sbnn/btp-2023S/madelon"
code = "sbnn_samplemean"

np.random.seed(123)
torch.manual_seed(456)

#------------------------------------------------------------------------------------------------------
# Create a simple dataset
def make_classification_dataset(n, d, d_imp, classes):
    return make_classification(n_samples = n, n_features = d, n_informative = d_imp, n_redundant = 0, n_classes = classes, n_clusters_per_class = 1, random_state = 10)

data_size = 4000
test_size = 1000
data_dim = 100
d_imp = 5
no_of_classes = 3
target_classes = ['Class 1', 'Class 2', 'Class 3']
sigma_noise = 1.
rep = 1

trainsets = []
for i in range(rep):
    x, y = make_classification_dataset(data_size + test_size, data_dim, d_imp=d_imp, classes=no_of_classes)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .8)
    x_train = torch.Tensor(x_train)
    y_train = torch.LongTensor(y_train)
    trainset = [x_train, y_train]
    trainsets.append(trainset)

x_test = torch.Tensor(x_test)
y_test = torch.LongTensor(y_test)

# ------------------------------------------------------------------------------------------------------
batch_size = 512
num_batches = data_size / batch_size
learning_rate = torch.tensor(5e-3)
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

train_Loss = []
test_Loss = []
sparse_overalls = []
sparse_overalls2 = []
FNRs = []
FPRs = []
no_of_features_selected = []
training_loss = []
train_accuracy = []
test_accuracy = []
# l1_wtheta = np.zeros(shape=(epochs, data_dim, hidden_dim))

for k in range(rep):
    print('------------ round {} ------------'.format(k))
    # create sparse BNN
    net = SparseBNNClassification(data_dim, hidden_dim = hidden_dim, target_dim = no_of_classes, device = device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    x_train = trainsets[k][0]
    y_train = trainsets[k][1]
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = []
        permutation = torch.randperm(data_size)

        for i in range(0, data_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, _ = net.sample_elbo(batch_x, batch_y, 1, temp, phi_prior, num_batches)
            if torch.isnan(loss):
                break
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        training_loss.append(np.mean(train_losses))
        # wtheta = np.array(net.l1.w_theta.detach().numpy())
        # l1_wtheta[epoch] = wtheta
        # one1_w = (sigmoid(net.l1.w_theta) > 0.5).float()
        # p = torch.sum(one1_w, axis=1)
        # no_of_features_selected.append(torch.sum(p>=1))

        _, pred = net.sample_elbo(x_train.to(device), y_train.to(device), 30,
                                temp, phi_prior, num_batches)
        pred = torch.mode(pred, dim=0).values
        train_accuracy.append(torch.sum(pred == y_train) / y_train.shape[0])

        _, pred2 = net.sample_elbo(x_test.to(device), y_test.to(device), 30,
                                temp, phi_prior, num_batches)
        pred2 = torch.mode(pred2, dim=0).values
        test_accuracy.append(torch.sum(pred2 == y_test) / y_test.shape[0])

        if epoch % 1000 == 0 or epoch==epochs-1:
            one1_w = (net.l1.w != 0).float()
            one1_b = (net.l1.b != 0).float()
            one2_w = (net.l2.w != 0).float()
            one2_b = (net.l2.b != 0).float()
            one3_w = (net.l3.w != 0).float()
            one3_b = (net.l3.b != 0).float()
            one4_w = (net.l4.w != 0).float()
            one4_b = (net.l4.b != 0).float()
            sparsity = (torch.sum(one1_w) + torch.sum(one2_w) + torch.sum(one3_w) + torch.sum(one4_w) +
                        torch.sum(one1_b) + torch.sum(one2_b) + torch.sum(one3_b) + torch.sum(one4_b)) / total
            print('Epoch {}, Train_Loss: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), phi_prior, sparsity))
    
    print('Finished Training')
    # sparsity level
    one1_w = (sigmoid(net.l1.w_theta)).float()
    one1_b = (sigmoid(net.l1.b_theta) > 0.5).float()
    one2_w = (sigmoid(net.l2.w_theta) > 0.5).float()
    one2_b = (sigmoid(net.l2.b_theta) > 0.5).float()
    one3_w = (sigmoid(net.l3.w_theta) > 0.5).float()
    one3_b = (sigmoid(net.l3.b_theta) > 0.5).float()
    one4_w = (sigmoid(net.l4.w_theta) > 0.5).float()
    one4_b = (sigmoid(net.l4.b_theta) > 0.5).float()
    sparse_overall = (torch.sum(one1_w) + torch.sum(one2_w) + torch.sum(one3_w) + torch.sum(one4_w) +
                      torch.sum(one1_b) + torch.sum(one2_b) + torch.sum(one3_b) + torch.sum(one4_b)) / total
    sparse_overalls.append(sparse_overall)
    sparse_overall2 = (torch.sum(sigmoid(net.l1.w_theta)) + torch.sum(sigmoid(net.l1.b_theta)) +
                       torch.sum(sigmoid(net.l2.w_theta)) + torch.sum(sigmoid(net.l2.b_theta)) +
                       torch.sum(sigmoid(net.l3.w_theta)) + torch.sum(sigmoid(net.l3.b_theta)))/total
    sparse_overalls2.append(sparse_overall2)
    torch.set_printoptions(profile="full")

    print("\n", "----------- Network Sparsity -----------")
    print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    # print('l1 w Edges: {}'.format(one1_w))
    p = torch.mean(one1_w, axis=1)
    sorted, indices = torch.sort(p,0, descending=True)
    print('features selected in the first layer: {}'.format(indices[0:10]))
    torch.save(indices, f"{curr_dir}/{code}/indices_tensor_sbnn.pt")

    print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
    print('l1 b Edges: {}'.format(one1_b))

    # prediction
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    _, pred = net.sample_elbo(x_train, y_train, 30,
                              temp, phi_prior, num_batches)
    print("shape of output -> ",pred.shape)
    
    pred = torch.mode(pred, dim=0).values
    train_loss = torch.sum(pred == y_train) / y_train.shape[0]
    train_Loss.append(train_loss)

    print("----------- Training -----------")
    print('y_train: {}'.format(y_train[0:20]))
    print('pred_train: {}'.format(pred[0:20]))
    print('binary_loss_train: {}'.format(train_loss))

    # ------------------------------------------------------------------------------------------------------
    print("\n", "----------- Testing -----------")
    # testing
    # prediction

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    _, pred2 = net.sample_elbo(
        x_test, y_test, 30, temp, phi_prior, num_batches)
    
    pred2 = torch.mode(pred2, dim=0).values
    test_loss = torch.sum(pred2 == y_test) / y_test.shape[0]
    test_Loss.append(test_loss)

    print('y_test: {}'.format(y_test[0:20]))
    print('pred_test: {}'.format(pred2[0:20]))
    print('binary_loss_test: {}'.format(test_loss))
    print("\n")

# np.save(f"{curr_dir}/{code}/l1_wtheta_sm", l1_wtheta)

train_LOSS = torch.tensor(train_Loss)
test_LOSS = torch.tensor(test_Loss)
sparse_overalls = torch.tensor(sparse_overalls)
sparse_overalls2 = torch.tensor(sparse_overalls2)
FNRs = torch.tensor(FNRs)
FPRs = torch.tensor(FPRs)

print("\n", "----------- Summary -----------")
print('binary_loss_MEAN_train: {}'.format(torch.mean(train_LOSS)))
print('binary_loss_std_train: {}'.format(torch.std(train_LOSS)))
print('binary_loss_MEAN_test: {}'.format(torch.mean(test_LOSS)))
print('binary_loss_std_test: {}'.format(torch.std(test_LOSS)))
print('sparsity: {}'.format(torch.mean(sparse_overalls)))
print('sparsity2: {}'.format(torch.mean(sparse_overalls2)))
print('FNR: {}'.format(torch.mean(FNRs)))
print('FNR sd: {}'.format(torch.std(FNRs)))
print('FPR: {}'.format(torch.mean(FPRs)))
print('FPR sd: {}'.format(torch.std(FPRs)))


print('sparsity all: {}'.format(sparse_overalls))
print('sparsity all 2: {}'.format(sparse_overalls2))
print('binary_loss_train all: {}'.format(train_LOSS))
print('binary_loss_test all: {}'.format(test_LOSS))
print('FNRs: {}'.format(FNRs))
print('FPRs: {}'.format(FPRs))


y_train = y_train.cpu()
pred = pred.cpu()

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_train, pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_train, pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_train, pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_train, pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_train, pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_train, pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_train, pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_train, pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_train, pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_train, pred, average='weighted')))

print('\nClassification Report\n')
print(classification_report(y_train, pred, target_names = target_classes))



y_test = y_test.cpu()
pred2 = pred2.cpu()

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, pred2)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, pred2, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, pred2, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, pred2, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, pred2, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, pred2, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, pred2, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, pred2, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, pred2, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, pred2, average='weighted')))

print('\nClassification Report\n')
print(classification_report(y_test, pred2, target_names = target_classes))

plt.plot(training_loss)
plt.xlabel("epochs \n Final Loss: {:.2f}".format(training_loss[epochs-1]))
plt.ylabel("Training Loss")
plt.tight_layout()
plt.show()

plt.plot(train_accuracy, label = 'train_accuracy')
plt.plot(test_accuracy, label = 'test_accuracy')
plt.xlabel("epochs \n Final Train Accuracy : {:.3f}, Final Test Accuracy: {:.3f}".format(train_accuracy[epochs-1], test_accuracy[epochs-1]))
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# plt.plot(no_of_features_selected)
# plt.xlabel("epochs \n Total no of Features selected: {}".format(no_of_features_selected[epochs-1]))
# plt.ylabel("no of Features selected")
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize = (8,8))
# cm = confusion_matrix(y_train, pred)
# cm_display = ConfusionMatrixDisplay(cm)
# cm_display.plot(ax = ax)
# plt.title("Confusion matrix for training data", fontdict = {'fontsize': 18, 'color': 'teal'}, pad = 15)
# plt.show()

# fig, ax = plt.subplots(figsize = (8,8))
# cm = confusion_matrix(y_test, pred2)
# cm_display = ConfusionMatrixDisplay(cm)
# cm_display.plot(ax = ax)
# plt.title("Confusion matrix for testing data", fontdict = {'fontsize': 18, 'color': 'teal'}, pad = 15)
# plt.show()


print("Total Time Consumed --> ", time.time() - start_time)
