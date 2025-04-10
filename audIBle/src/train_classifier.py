import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audIBle.data.datasets import ESC_50,WHAMDataset,combine_batches
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tr

from datetime import datetime
from audIBle.nn.encoder_simple import Encoder

from audIBle.nn.classifier_simple import Classifier
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Classif(nn.Module):
    
    def __init__(self,emb_dim=512,nclass=50):
        super(Classif, self).__init__()
        self.encoder = Encoder(dim=emb_dim)
        self.classi = Classifier(dim=emb_dim,n_class=nclass)
    def forward(self,X):
        h = self.encoder(X)
        out=self.classi(h)
        return out
    def compute_objectives(self,pred,labels,stage="Train"):
        acc = torch.sum(pred.squeeze().argmax(-1) == labels)/len(pred)
        return F.cross_entropy(pred,labels.unsqueeze(1)),acc
def train_one_epoch(epoch_index, training_loader,model,optimizer):
    running_loss = 0.
    last_loss = 0.
    running_acc = 0.
    last_acc = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        pred = model(inputs.to(DEVICE))# X logits M h

        # Compute the loss and its gradients
        loss,acc = model.compute_objectives(pred, labels.to(DEVICE), stage="train")
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_acc +=acc
        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            last_acc = running_acc/10
            print(f'    batch {i+1} loss: {last_loss} accuracy(#50): {last_acc*100}%')
            running_loss = 0.
            running_acc = 0.

    return last_loss

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = Classif(emb_dim=512,nclass=50)
    model.to(DEVICE)
    train_esc50_set = ESC_50(root="/mnt/data",part="train")
    test_esc50_set = ESC_50(root="/mnt/data",part="test")
    valid_esc50_set = ESC_50(root="/mnt/data",part="valid")
    train_esc50_loader = DataLoader(train_esc50_set,batch_size=16)
    valid_esc50_loader = DataLoader(valid_esc50_set,batch_size=16)
    

    #dummy = torch.rand((8,1,44100*5))
    #res = lmac(dummy)
    #spec_gen = lmac.interpret_computation_steps(dummy)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCH = 100
    best_loss = 100
    best_epoch=0
    
    for e in range(EPOCH):
        path_to_model = 'model_{}_{}'.format(timestamp, e+1)
        l_loss = train_one_epoch(epoch_index=e+1,training_loader=train_esc50_loader,model=model,optimizer=optimizer)
        model.eval()
        running_vloss=0.
        running_vacc=0.
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valid_esc50_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(DEVICE))
                vloss,vacc = model.compute_objectives(voutputs, vlabels.to(DEVICE),stage='valid')
                running_vloss += vloss
                running_vacc += vacc

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc/(i+1)
        print(f'EPOCH {e} LOSS train {l_loss} valid {avg_vloss}, Acc : {avg_vacc}')
        if avg_vloss<best_loss:
            best_loss = avg_vloss
            best_epoch = e
            model_path = path_to_model
            torch.save(model.state_dict(), model_path)

        model.train()
