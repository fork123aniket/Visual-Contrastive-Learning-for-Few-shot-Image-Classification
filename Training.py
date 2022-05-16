import torchvision
from torch.autograd import Variable
import torch
import pickle
import os
import random
import numpy as np
from Model import SiameseNetwork
from torch.optim import Adam
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(save_path):
    with open(os.path.join(save_path, "train.pickle"), "rb") as f:
        (Xtrain, train_classes) = pickle.load(f)
    with open(os.path.join(save_path, "val.pickle"), "rb") as f:
        (Xval, val_classes) = pickle.load(f)
    Xtrain, Xval = torch.tensor(Xtrain), torch.tensor(Xval)
    print(f'Xtrain size: {Xtrain.size()} and Xval size: {Xval.size()}')
    return Xtrain, Xval, train_classes, val_classes


Xtrain, Xval, train_classes, val_classes = load_data('data/')

Xtrain, Xval = Xtrain / 255.0, Xval / 255.0


def getBatch(batch_size, k_shot, s="train"):
    if s == 'train':
        X = Xtrain
    else:
        X = Xval
    (n_classes, n_examples, w, h), channels = X.shape, 1
    categories = np.random.choice(n_classes, size=(batch_size,), replace=False)
    pairs = torch.tensor(np.array([np.zeros((batch_size, channels, w, h)) for _ in range(k_shot + 1)]))
    targets = np.zeros((batch_size,))
    if s == 'train':
        targets[batch_size // 2:] = 1
    else:
        targets[0] = 1
        pairs[0] = X[categories[0], 4, :, :].repeat(batch_size, 1, 1).reshape(batch_size, channels, w, h)
    for i in range(batch_size):
        idx = np.random.randint(0, n_examples, k_shot + 1)
        if s != 'train':
            if i == 0:
                for shot in range(1, k_shot + 1):
                    pairs[shot][i, :, :, :] = X[categories[0], idx[shot]].reshape(channels, w, h)
            else:
                for shot1 in range(1, k_shot + 1):
                    pairs[shot1][i, :, :, :] = X[categories[i], idx[shot1]].reshape(channels, w, h)
        else:
            category = categories[i]
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + np.random.randint(1, n_classes)) % n_classes
            pairs[0][i, :, :, :] = X[category, idx[0]].reshape(channels, w, h)
            for j in range(1, k_shot + 1):
                pairs[j][i, :, :, :] = X[category_2, idx[j]].reshape(channels, w, h)
    pairs = pairs.float()
    targets = torch.from_numpy(np.array(targets, dtype=np.float32))
    if s != 'train':
        indices = torch.randperm(batch_size)
        for i in range(k_shot + 1):
            pairs[i] = pairs[i][indices]
        targets = targets[indices]
    return pairs, targets


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, -20, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


set_seed(9)
net = SiameseNetwork(1.0)
optimizer = Adam(net.parameters(), lr=1e-2)  # weight_decay=2e-4
val_instances, counter, loss_history = 10, [], []
train_N_way, t_shot, val_N_way, k_shot, trials, num_epochs = 15, 6, 1, 1, 1, 10

for epoch in range(num_epochs):
    net.train()
    optimizer.zero_grad()
    img0, label = getBatch(train_N_way, t_shot)
    loss = net(img0, label, t_shot)
    loss = Variable(loss, requires_grad=True)
    loss.backward()
    optimizer.step()
    print("Epoch number {}\nCurrent loss {}\n".format(epoch, loss.item()))
    counter.append(epoch)
    loss_history.append(loss.item())

show_plot(counter, loss_history)

net.eval()
n_correct = 0
print("Evaluating model on {} random {} way {}-shot learning tasks ... \n".format(trials, val_N_way,
                                                                                  k_shot))
with torch.no_grad():
    for i in range(trials):
        nimg0, nlabel = getBatch(val_N_way, k_shot, 'val')
        actual_distance = net(nimg0, nlabel, k_shot)
        if torch.argmin(actual_distance) == torch.argmax(nlabel):
            n_correct += 1
    percent_correct = (100.0 * n_correct / trials)
    print("Got an average of {}% {} way {}-shot learning accuracy \n".format(percent_correct, val_N_way,
                                                                             k_shot))


def extract_sample(batch_size, k_shot, it, indices):
    pairs = torch.tensor(np.array([np.zeros((batch_size, 1, *Xval.size()[-2:])) for _ in range(k_shot + 1)]))
    pairs[0] = Xval[15, 4, :, :].repeat(batch_size, 1, 1).reshape(batch_size, 1, *Xval.size()[-2:])
    if it in indices:
        ind = np.random.randint(Xval.size(1))
        pairs[1] = Xval[15, ind, :, :].repeat(batch_size, 1, 1).reshape(batch_size, 1, *Xval.size()[-2:])
    else:
        index = np.random.randint(68, Xval.size(0), (1,))
        pairs[1] = Xval[index, 12, :, :].repeat(batch_size, 1, 1).reshape(batch_size, 1, *Xval.size()[-2:])
    targets = np.zeros((batch_size,))
    targets[0] = 1 if it in indices else 0
    pairs, targets = pairs.float(), torch.from_numpy(np.array(targets, dtype=np.float32))
    return pairs, targets


net.eval()
pair_distance, pair_labels = [], []
val_images = torch.zeros((val_instances, k_shot + 1, val_N_way, 1, *Xval.size()[-2:]))
indices = [0, 5, 7]
with torch.no_grad():
    for it in range(val_instances):
        nimg0, nlabel = extract_sample(val_N_way, k_shot, it, indices)
        distance = net(nimg0, nlabel, k_shot)
        concatenated = torch.cat((nimg0[0], nimg0[1]), 0)
        imshow(torchvision.utils.make_grid(concatenated),
               'Dissimilarity: {:.4f}'.format(distance.item()))
        pair_distance.append(distance.item())
        pair_labels.append(nlabel)
        val_images[it] = nimg0
    pair_distance, pair_labels = torch.tensor(pair_distance), torch.tensor(pair_labels)
    min_distance = torch.topk(pair_distance, len(indices), largest=False, sorted=False)
    pair_labels = pair_labels[min_distance.indices]
    val_images = val_images[min_distance.indices]
    pair_distance = pair_distance[min_distance.indices]
    for val_img in range(len(indices)):
        imshow(torchvision.utils.make_grid(val_images[val_img].view(k_shot + 1, 1, *Xval.size()[-2:])),
               'Dissimilarity: {:.4f} \npredicted label: {}'.format(pair_distance[val_img].item(),
                                                                      pair_labels[val_img]))
