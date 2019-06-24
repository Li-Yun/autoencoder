import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # two sequential models: encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            # compress to 3 features which can be visualized in plt
            nn.Linear(16, 8),)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 28*28),
            # compress to a range (0, 1)
            nn.Sigmoid(),)
    def forward(self, in_put):
        encoded = self.encoder(in_put)
        decoded = self.decoder(encoded)
        return encoded, decoded

# hyper-parameter settings
BATCH_SIZE = 64
learning_rate = 0.005
os.environ["CUDA_VISIBLE_DEVICES"]="0"
N_TEST_IMG = 5
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_loading():
    # Mnist digits dataset
    trainData = torchvision.datasets.MNIST(
            root='/home/li-yun/Downloads/dl/mnist/',
            train=True,
            # Cast PIL.Image or numpy to tensor (C x H x W)
            transform=torchvision.transforms.ToTensor(),
            download=True)
    # Data Loader (the image batch shape will be (100, 1, 28, 28))
    train_loader = Data.DataLoader(dataset = trainData, batch_size = BATCH_SIZE, shuffle=True, num_workers = 6)
    return train_loader, trainData

def network_training(data_loader, trainData, model, max_itr):
    # declare a loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()   # continuously plot
    
    # original data (first row) for viewing
    view_data = trainData.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], 
            (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

    # training the network
    for epoch_num in range(max_itr):
        print('====================')
        # get inputs
        for index, data in enumerate(data_loader, 0):
            inputs, _ = data
            x_batch = inputs.view(-1, 28*28)   # batch x, shape (batch, 28*28)
            y_batch = inputs.view(-1, 28*28)   # batch y, shape (batch, 28*28)
            x_batch = Variable(x_batch).cuda(async=True)
            y_batch = Variable(y_batch).cuda(async=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            encoded, decoded = model(x_batch)
            loss = criterion(decoded, y_batch)
            loss.backward()
            # update the weights (apply the gradients)
            optimizer.step()

            # show the loss values
            if index % 100 == 0:
                print('Epoch: ', epoch_num, '| train loss: %.4f' % loss.item())
                # show decoded images
                view_data = Variable(view_data).cuda()
                _, decoded_data = model(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(()); a[1][i].set_yticks(())
                    plt.draw(); plt.pause(0.05)
    plt.ioff()
    plt.show()

def reconstruction(in_data, autoencoder_model):
    f, a = plt.subplots(2, 5, figsize=(5, 2))
    plt.ion()   # continuously plot

    # visualize the testing images
    for i in range(5):
        a[0][i].imshow(np.reshape(in_data.data.numpy()[i],
            (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
    # get reconstruction images
    
    in_data = Variable(in_data).cuda()
    _, decoded_data = autoencoder_model(in_data)
    for i in range(5):
        a[1][i].clear()
        a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw()
    plt.ioff()
    plt.show()

def main():
    # data loader
    train_data_loader, trainData = data_loading()

    # load the model
    autoencoder = Autoencoder()
    autoencoder.cuda()
    # training the network
    network_training(train_data_loader, trainData, autoencoder, 60)
    # image reconstruction
    test_data = trainData.train_data[5:].view(-1, 28*28).type(torch.FloatTensor)/255.
    reconstruction(test_data, autoencoder)

if __name__ == "__main__":
    main()
