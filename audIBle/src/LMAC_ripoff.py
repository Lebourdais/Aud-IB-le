import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audIBle.data.datasets import ESC_50
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tr

from datetime import datetime
from encoders import Cnn14
from decoders import CNN14PSI_stft
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def tv_loss(mask, tv_weight=1, power=2, border_penalty=0.3):
    if tv_weight is None or tv_weight == 0:
        return 0.0
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/PiotrDabkowski/pytorch-saliency/blob/bfd501ec7888dbb3727494d06c71449df1530196/sal/utils/mask.py#L5
    w_variance = torch.sum(torch.pow(mask[:, :, :-1] - mask[:, :, 1:], power))
    h_variance = torch.sum(torch.pow(mask[:, :-1, :] - mask[:, 1:, :], power))

    loss = tv_weight * (h_variance + w_variance) / float(power * mask.size(0))
    return loss
    
class SpecMag(nn.Module):
    def __init__(self,power=1.0,log=False):
         super(SpecMag, self).__init__()
         self.power = power
         self.log = log
    def forward(self,X):
        eps = 1e-8
        spectr = X.pow(2)# .sum(-1) not complex ?

        # Add eps avoids NaN when spectr is zero
        if self.power < 1:
            spectr = spectr + eps
        spectr = spectr.pow(self.power)

        if self.log:
            return torch.log(spectr + eps)
        return spectr
    
class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    input_size : int
        Expected size of input dimension.
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    nn.BatchNorm1d(num_features=input_size),
                    nn.Linear(input_size,lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.Tensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.

        Returns
        -------
        out : torch.Tensor
            Output probabilities over speakers.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)

def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False
class LMAC(nn.Module):
    L_IN_W = 4
    L_OUT_W = 0.2
    REG_W_TV = 0.0
    REG_W_L1 = 0.4
    G_W = 4
    CROSSCOR_TH=0.6
    BIN_TH=0.35
    def __init__(self,embedding_path,classifier_path,emb_dim,n_class,verbose=True):
        super(LMAC, self).__init__()
        self.verbose = verbose
        embedding_model = torch.load(embedding_path,map_location=DEVICE)
        self.encoder = Cnn14(mel_bins=80, emb_dim=emb_dim, return_reps=True)
        self.encoder.load_state_dict(embedding_model)
        freeze(self.encoder)
        self.classifier = Classifier(input_size=emb_dim,lin_blocks=1,out_neurons=n_class,lin_neurons=192)
        classifier_model = torch.load(classifier_path,map_location=DEVICE)
        self.classifier.load_state_dict(classifier_model,strict=False)
        freeze(self.classifier)
        self.decoder = CNN14PSI_stft(dim=2048)
        self.stft = tr.Spectrogram(n_fft=1024,hop_length=512,win_length=1024) # not exactly the same as the paper but cleaner

        self.stft_power = SpecMag(power=0.5)

        self.mel = tr.MelScale(n_mels=80,sample_rate=44100,n_stft=513,f_min=0,f_max=8000)#magic value


    def interpret_computation_steps(self,X):
        print("Compute interpretable spec")
        X_stft = self.stft(X)
        print(f"{X_stft.shape=}")
        X_stft_power = self.stft_power(X_stft)
        print(f"{X_stft_power.shape=}")
        X_mel = torch.log1p(self.mel(X_stft)).transpose(2,3)
        print(f"{X_mel.shape=}")
        X_stft_logpower = torch.log1p(X_stft_power)
        print(f"{X_stft_logpower.shape=}")
        X_stft_phase = torch.atan2(X_stft[:, :, :, 1], X_stft[:, :, :, 0]) #Getting the phase
        print(f"{X_stft_phase.shape=}")
        h,intermediate_repr = self.encoder(X_mel) # dims
        print(f"{h.shape=}")

        logits = self.classifier(h.transpose(1,2).squeeze(2)) # dims
        print(f"{logits.shape=}")
        class_preds = logits.argmax(-1).squeeze() # dims
        if self.verbose:
            predictions = F.softmax(logits).squeeze(1)
            print(f"{predictions.shape=},{class_preds.shape=}")
            class_prob = predictions[0,class_preds[0]].item()
            print(f"classifier_prob: {class_prob}")
        
        M = self.decoder(intermediate_repr).squeeze(1)#needs every repr
        M = F.sigmoid(M)
        Tmax = M.shape[1]
        
        X_hat = M * X_stft_logpower[:,:Tmax,:]
        return (
            X_hat.transpose(1, 2),
            M.transpose(1, 2),
            X_stft_phase,
            X_stft_logpower,
        )
    def compute_forward(self,X):
        #print("Compute forward")
        X_stft = self.stft(X)
        
        X_mel = torch.log1p(self.mel(X_stft)).transpose(2,3)
         

        h,intermediate_repr = self.encoder(X_mel) # dims
        
        logits = self.classifier(h.transpose(1,2).squeeze(2)) # dims
        
        #class_preds = logits.argmax() # dims
        
        M = self.decoder(intermediate_repr).squeeze(1)#needs every repr
        M = F.sigmoid(M)

        return (X, logits, M.transpose(1,2), h)
    
    def forward(self,X):
        return self.compute_forward(X)
    
    def compute_objectives(self, pred, label, stage='train'):
        """Helper function to compute the objectives"""
        (
            X,
            predictions,
            xhat,
            _,
        ) = pred


        #uttid = batch.id
        labels = label
        
        X_stft = self.stft(X)
        # X_stft : Batch 1 Freq Time
        X_stft_power = self.stft_power(X_stft.transpose(1,2))
        X_stft_logpower = torch.log1p(X_stft_power).squeeze(2)
        # X_stft_logpower : Batch Freq Time
        # xhat: Batch Freq time
        Tmax = xhat.shape[2]

        # map clean to same dimensionality
        X_stft_logpower = X_stft_logpower[:, :, :Tmax]

        mask_in = xhat * X_stft_logpower
        #mask_in : Batch freq time
        mask_out = (1 - xhat) * X_stft_logpower
        X_in = torch.expm1(mask_in).transpose(1,2)
        #X_in : Batch time freq
        fbank = torchaudio.functional.melscale_fbanks(n_freqs = 513,
                        f_min = 0,
                        f_max=8000,
                        n_mels=80, 
                        sample_rate=44100,).to(DEVICE)
        mask_in_mel = torch.log1p(X_in @ fbank)
        #mask in mel: Batch time mel
        #mask_in_mel = torch.log1p(mask_in_mel)

        X_out = torch.expm1(mask_out).transpose(1,2)
        mask_out_mel = torch.log1p(X_out @ fbank)

        rec_loss = 0
        crosscor_mask = torch.zeros(xhat.shape[0], device=DEVICE)

        h_in,_ = self.encoder(mask_in_mel) # dims
        
        mask_in_preds = self.classifier(h_in.transpose(1,2).squeeze(2)) # dims
        h_out,_ = self.encoder(mask_out_mel) # dims
        
        mask_out_preds = self.classifier(h_out.transpose(1,2).squeeze(2)) # dims
        class_pred = predictions.argmax(1)

        l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)
        l_out = -F.nll_loss(mask_out_preds.log_softmax(1), class_pred)

        ao_loss = l_in * self.L_IN_W + self.L_OUT_W * l_out

        r_m = (
            xhat.abs().mean((-1, -2, -3))
            * self.REG_W_L1
            * torch.logical_not(crosscor_mask)
        ).sum()
        r_m += (
            tv_loss(xhat)
            * self.REG_W_TV
            * torch.logical_not(crosscor_mask)
        ).sum()

        mask_in_preds = mask_in_preds.softmax(1)
        mask_out_preds = mask_out_preds.softmax(1)
        

        # self.in_masks.append(uttid, c=crosscor_mask)
        # self.acc_metric.append(
        #     uttid,
        #     predict=predictions,
        #     target=labels,
        # )

        # if stage != sb.Stage.TEST:
        #     if hasattr(self.hparams.lr_annealing, "on_batch_end"):
        #         self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return ao_loss + r_m + rec_loss

def train_one_epoch(epoch_index, training_loader,model,optimizer):
    running_loss = 0.
    last_loss = 0.

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
        loss = model.compute_objectives(pred, labels.to(DEVICE), stage="train")
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print(f'    batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

if __name__ == "__main__":
    embedding_path = "/lium/raid-b/tahon/audIBle/checkpoints-lmac/embedding_model.ckpt"
    classif_path ="/lium/raid-b/tahon/audIBle/checkpoints-lmac/classifier.ckpt"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lmac = LMAC(embedding_path=embedding_path,classifier_path=classif_path,emb_dim=2048,n_class=50)
    lmac.to(DEVICE)
    train_esc50_set = ESC_50(root="/lium/corpus/vrac/tmario/",part="train")
    test_esc50_set = ESC_50(root="/lium/corpus/vrac/tmario/",part="test")
    valid_esc50_set = ESC_50(root="/lium/corpus/vrac/tmario/",part="valid")
    train_esc50_loader = DataLoader(train_esc50_set,batch_size=16)
    valid_esc50_loader = DataLoader(valid_esc50_set,batch_size=16)
    

    #dummy = torch.rand((8,1,44100*5))
    #res = lmac(dummy)
    #spec_gen = lmac.interpret_computation_steps(dummy)
    
    optimizer = torch.optim.Adam(lmac.parameters(), lr=0.0002, weight_decay=0.000002)
    EPOCH = 100
    best_loss = 100
    best_epoch=0
    for e in range(EPOCH):
        path_to_model = '.../models/model_{}_{}'.format(timestamp, e+1)
        l_loss = train_one_epoch(epoch_index=e+1,training_loader=train_esc50_loader,model=lmac,optimizer=optimizer)
        lmac.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valid_esc50_loader):
                vinputs, vlabels = vdata
                voutputs = lmac(vinputs.to(DEVICE))
                vloss = lmac.compute_objectives(voutputs, vlabels.to(DEVICE),stage='valid')
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'EPOCH {e} LOSS train {l_loss} valid {avg_vloss}')
        if avg_vloss<best_loss:
            best_loss = avg_vloss
            best_epoch = e
            model_path = path_to_model
            torch.save(lmac.state_dict(), model_path)

        lmac.train()
