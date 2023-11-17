import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import evaluate
from utils import metrics_evaluation

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')


from data_utils.dataset import get_loader
from data_utils.utils import LoadConfig, load_model, get_optimer

def get_loader(train = True):
    transform = T.Compose([T.Resize((config['image_size'], config['image_size'])),
                           T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.ToTensor()])

    train_loader, val_loader = create_dataloader(config['input_dir'], config['datasets']['neg_samples_dir'], transform, config["datasets"]["train"]['batch_size'], shuffle=True)
    return train_loader, val_loader

def train_one_epoch(model, optimizer, crit, metric, dataloader):
    loss = 0
    model.train()

    for i, (img, name) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # load it to the active device
        img = img.to(device)
        outputs = model(img)

        train_loss = criterion(outputs, img)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        wandb.log({"Train Loss": train_loss.item()}, step = len(dataloader)*len(img)*(epoch)+i*len(img))

    # compute the epoch training loss
    return loss, res

def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for img, name in tqdm(dataloader):
            img = img.to(device)
            outputs = model(img)

            val_loss = criterion(outputs, img)
            loss += val_loss.item()
    
        # save the last batch input and output of every epoch
        for j in range(outputs.size(0)):
            save_image(outputs[j], f'{config["output_dir"]}/{epoch}_{j}.png')
                    
    return loss, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run3')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    with wandb.init(project='test', config=config, name=args.test_name) as run:
    
    
        data_train = Data(data, partitions['train'])
        data_valid = Data(data, partitions['valid'])
        data_test = Data(data, partitions['test'])
        dataloader_train = '''write a proper dataloader, same for valid and test'''
        model = Model().to(DEVICE)
        model.train()
        optimizer = '''choose a proper optimizer'''
        crit = nn.CrossEntropyLoss()
        metric = Metric()
        for epoch in range(EPOCHS):
            loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
            print(f'train loss: {loss:.2f}, metric: {res:.2f}, epoch: {epoch}')
            loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid)
            print(f'valid loss: {loss:.2f}, metric: {res:.2f}')
            wandb.log({"Epoch Train Loss": train_loss, "Epoch Validation Loss": val_loss, "epoch":epoch+1}, step=(epoch+1)*len(train_loader)*config["datasets"]["train"]['batch_size'])
        loss_t, res_t = eval_epoch(model, crit, metric, testloader_test)
        print(f'test loss: {loss:.2f}, metric: {res:.2f}')
    
        # if config["network"]["checkpoint"] != None: 
        #     model.load_state_dict(torch.load(config["network"]["checkpoint"]))
        #     print("Load model from checkpoint {}".format(config["network"]["checkpoint"]))

        wandb.watch(model)
        optimizer = nn.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            print("Epoch {}/{}".format(epoch, config['epochs']))
            print("Training")
            train_loss = train(model, train_loader, optimizer, criterion, epoch=epoch)
            print("Validation")
            val_loss = validation(model, val_loader, criterion, epoch = epoch)
            print("epoch : {}| Train loss = {:.6f}| Val loss = {:.6f}".format(epoch, train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best validation loss")
                torch.save(model.state_dict(), f'{config["weights_dir"]}/CNN_autoencoder_epoch_{epoch}.pth')
    

    
def train_one_epoch(model, optimizer, crit, metric, dataloader):
    '''finish the code'''
    return loss, res

def eval_epoch(model, crit, metric, dataloader):
    '''finish the code'''
    return loss, res
