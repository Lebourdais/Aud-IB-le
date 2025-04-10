import torch

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
        self.blocks = torch.nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    torch.nn.BatchNorm1d(num_features=input_size),
                    torch.nn.Linear(input_size,lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_neurons, input_size, device=device)
        )
        torch.nn.init.xavier_uniform_(self.weight)

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
        x = torch.nn.functional.linear(torch.nn.functional.normalize(x.squeeze(1)), torch.nn.functional.normalize(self.weight))
        return x.unsqueeze(1)