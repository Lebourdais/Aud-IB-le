import torch.nn as nn
class CNN14PSI_stft(nn.Module):
    """
    This class estimates a saliency map on the STFT domain, given classifier representations.

    Arguments
    ---------
    dim : int
        Dimensionality of the input representations.
    outdim : int
        Defines the number of output channels in the saliency map.

    Example
    -------
    >>> from speechbrain.lobes.models.Cnn14 import Cnn14
    >>> classifier_embedder = Cnn14(mel_bins=80, emb_dim=2048, return_reps=True)
    >>> x = torch.randn(2, 201, 80)
    >>> _, hs = classifier_embedder(x)
    >>> psimodel = CNN14PSI_stft(2048, 1)
    >>> xhat = psimodel.forward(hs)
    >>> print(xhat.shape)
    torch.Size([2, 1, 201, 513])
    """

    def __init__(self, dim=128, outdim=1):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(dim, dim, 3, (2, 4), 1)
        self.convt2 = nn.ConvTranspose2d(dim // 2, dim, 3, (2, 4), 1)
        self.convt3 = nn.ConvTranspose2d(dim, dim, (7, 4), (2, 4), 1)
        self.convt4 = nn.ConvTranspose2d(dim // 4, dim, (5, 4), (2, 4), 1)
        self.convt5 = nn.ConvTranspose2d(dim, dim // 2, (3, 5), (2, 2), 1)
        self.convt6 = nn.ConvTranspose2d(dim // 8, dim // 2, (3, 3), (2, 4), 1)
        self.convt7 = nn.ConvTranspose2d(
            dim // 2, dim // 4, (4, 3), (2, 2), (0, 5)
        )
        self.convt8 = nn.ConvTranspose2d(
            dim // 4, dim // 8, (3, 4), (2, 2), (0, 2)
        )
        self.convt9 = nn.ConvTranspose2d(dim // 8, outdim, (1, 5), (1, 4), 0)

        self.nonl = nn.ReLU(True)

    def forward(self, hs):
        """
        Forward step to estimate the saliency map

        Arguments
        --------
        hs : torch.Tensor
            Classifier's representations.

        Returns
        --------
        xhat : torch.Tensor
            An Estimate for the saliency map
        """

        h1 = self.convt1(hs[0])
        h1 = self.nonl(h1)

        h2 = self.convt2(hs[1])
        h2 = self.nonl(h2)
        h = h1 + h2

        h3 = self.convt3(h)
        h3 = self.nonl(h3)

        h4 = self.convt4(hs[2])
        h4 = self.nonl(h4)
        h = h3 + h4

        h5 = self.convt5(h)
        h5 = self.nonl(h5)

        h6 = self.convt6(hs[3])
        h6 = self.nonl(h6)

        h = h5 + h6

        h = self.convt7(h)
        h = self.nonl(h)

        h = self.convt8(h)
        xhat = self.convt9(h)

        return xhat