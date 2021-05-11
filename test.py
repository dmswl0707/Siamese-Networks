
from dataset import *
from train import net, F

test_dataset = ImageFolder(root=test_dir)
siamese_dataset = SiameNetworkDataset(imageFolderDataset = test_dataset,
                                      transform = transforms.Compose([transforms.Resize(100, 100),
                                                  transforms.ToTensor()]), should_invert = False)

test_dataloader = DataLoader(siamese_dataset, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ =next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concate = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concate), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))