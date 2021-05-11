from model import *
from loss import *
from dataset import  *
from torch import optim
import matplotlib.pyplot as plt


net = siamese_network().cuda()
criterion = Contrastive_loss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)


counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, num_epochs+1):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("epoch number {}\n current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

plt.plot(counter, loss_history)
plt.show()