import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        # two sequential models: encoder and decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride = 3, padding = 1), # 10x10x16
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2), # 5x5x16
            nn.Conv2d(32, 16, 3, stride = 2, padding = 1), # 3x3x16
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1), # 2x2x8
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride = 2), # 5x5x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 5, stride = 3, padding = 1), # 15x15x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride = 2, padding = 1), # 28x28x1
            nn.Tanh(),)

    def forward(self, in_put):
        encoded = self.encoder(in_put)
        decoded = self.decoder(encoded)
        return encoded, decoded

# hyper-parameter settings
BATCH_SIZE = 128
learning_rate = 0.001 # 0.005
os.environ["CUDA_VISIBLE_DEVICES"]="0"
N_TEST_IMG = 5

def data_loading():
    # Mnist digits dataset
    trainData = torchvision.datasets.MNIST(
            root='/home/li-yun/Downloads/dl/mnist/',
            train=True,
            # Cast PIL.Image or numpy to tensor (C x H x W)
            transform=torchvision.transforms.ToTensor())
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
    view_data = trainData.train_data[:N_TEST_IMG].type(torch.FloatTensor)/255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], 
            (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

    # training the network
    for epoch_num in range(max_itr):
        print('====================')
        # get inputs
        for index, data in enumerate(data_loader, 0):
            inputs, _ = data
            x_batch = inputs
            y_batch = inputs
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
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded.cpu().data.numpy()[i], (28, 28)), cmap='gray')
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
    in_data = in_data.unsqueeze_(1)
    #print(in_data.shape)
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
    conv_autoencoder = ConvolutionalAutoencoder()
    conv_autoencoder.cuda()
    # training the network
    network_training(train_data_loader, trainData, conv_autoencoder, 200)
    # image reconstruction
    test_data = trainData.train_data[:5].type(torch.FloatTensor)/255.
    reconstruction(test_data, conv_autoencoder)

if __name__ == "__main__":
    main()
