import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler, RandomSampler
import torch.nn as nn
import torch.optim as optim
# -----------------------------
from datasets.dataset import *
from fixmatch import * # Train func
from models.model import WideEffnet
from models.ema import EMA
from config import *

def main():
    # Config
    config = Config()
    #Data
    root_labeled = 'datasets/train/Images'
    root_unlabeled = 'datasets/phase1-test-images'
    one_hot_label = one_hot_labels(root_labeled)
    train_labeled_set, train_unlabeled_set = get_dataset(root_labeled, root_unlabeled, one_hot_label)
    labeled_trainloader = DataLoader(train_labeled_set,
                                     sampler = RandomSampler(train_labeled_set),
                                     batch_size = config.batch_size,
                                     drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_set,
                                       sampler= RandomSampler(train_unlabeled_set),
                                       batch_size= config.batch_size,
                                       drop_last=True)
    # Model - EMA model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideEffnet(num_class=10, model_name="efficientnet_b0", pretrained=False)
    # # Load - GPU
    # model.load_state_dict(torch.load(''))
    # # Load - CPU
    # model.load_state_dict(torch.load('', map_location=device))
    model.to(device)
    # model.eval()
    ema = EMA(model, config.ema_decay)

    # # # # # Train
    train_criterion = nn.CrossEntropyLoss()
    if config.optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), config.lr)
    if not os.path.exists("save_model"):
        os.mkdir("save_model")

    list_train_losses = []
    list_train_losses_w = []
    list_train_losses_u = []
    for epoch in range(config.epochs):
        # training
        train_losses, train_losses_x, train_losses_u = train(labeled_trainloader, unlabeled_trainloader,
                                                             model, optimizer, ema, train_criterion,
                                                             device, config)
        print('Epochs: {}, Train_loss: {}, loss_x: {}'.format(epoch, train_losses, train_losses_x))
        torch.save(ema.model.state_dict(), "save_model/FixMatch_model_epoch_{}_loss{}.pth".format(epoch, train_losses))

        list_train_losses.append(train_losses)
        list_train_losses_w.append(train_losses_x)
        list_train_losses_u.append(train_losses_u)


if __name__ == '__main__':
    main()