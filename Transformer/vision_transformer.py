import torch
import torch.nn as nn
import torch.optim as optim


# define Patch Embeddings
class PatchEmbedding(nn.Module):
    """Split the image into patches and then embed them
    Parameters
    ----------
    image_size: int
        size of the image (the window/length size of image in terms of square )

    patch_size: int
        Size of the patches (it is a square)

    in_chans: int
        Number of input channels

    embed_dim: int
        The embedding dimension

    Attributes
    ----------
    n_patches: int
        number of patches

    proj: nn.Conv2D
        A convolutional layer that splits the image into patches and embeds it
        Setting the stride = patch_size allows us to split image to non-overlapping patches


    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass

        Paramters:
        ----------
        x: torch.tensor
            Shape (n_sample, in_chans, img_size, img_size)

        Return:
        torch.tensor
            Shape (n_samples, n_patches, embed_dim)
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x
