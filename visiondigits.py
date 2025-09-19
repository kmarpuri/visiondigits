import torch
import torch.nn.functional as F
import torch.nn as nn
import idx2numpy as idx

train_images = idx.convert_from_file("./data/train-images-ubyte")
train_labels = idx.convert_from_file("./data/train-labels-ubyte")
test_images = idx.convert_from_file("./data/test-images-ubyte")
test_labels = idx.convert_from_file("./data/test-labels-ubyte")
row_size, col_size = 28, 28

num_dev = 10000
perm = torch.randperm(len(train_images))
train_idx = perm[num_dev:]
dev_idx = perm[:num_dev]

def build_xy(image, label):
    return torch.tensor(image).float() / 255.0, torch.tensor(label).long()

def data_nll(X, Y, model):
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        return loss.item()

def predict(model, X, Y, wrong):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        for i in range(X.size(0)):
            if pred[i].item() != Y[i].item():
                wrong.append(i)

X, Y = build_xy(train_images, train_labels)
Xtr, Ytr = X[train_idx], Y[train_idx]
Xdev, Ydev = X[dev_idx], Y[dev_idx]
Xte, Yte = build_xy(test_images, test_labels)

n_hidden = 64

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(row_size * col_size, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU6(),
    nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU6(),
    nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU6(),
    nn.Linear(n_hidden, 10)
)

with torch.no_grad():
    model[-1].weight *= 0.1

print(sum(p.numel() for p in model.parameters()))

for p in model.parameters():
    p.requires_grad = True

steps = 200000
batch_size = 32

for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    loss = F.cross_entropy(model(Xb), Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 50000 else 0.01 if i < 100000 else 0.001
    for p in model.parameters():
        p.data -= lr * p.grad

    if i % 10000 == 0:
        print(i, loss.item())

for layer in model:
    if isinstance(layer, nn.BatchNorm1d):
        layer.training = False

print(data_nll(Xtr, Ytr, model))
print(data_nll(Xdev, Ydev, model))

wrong = []
predict(model, Xte, Yte, wrong)
print(len(wrong))

print(data_nll(Xte, Yte, model))

percentage_correct = (len(test_labels) - len(wrong)) / len(test_labels) * 100
print(f'{percentage_correct:.2f}%')