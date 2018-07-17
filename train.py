import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from utils import to_var
from models import VAE

def main(args):

    ts = time.time()

    datasets = OrderedDict()
    datasets['train'] = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)

        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return BCE + KLD

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels= 10 if args.conditional else 0
        )

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)


    tracker_global = defaultdict(torch.FloatTensor)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for split, dataset in datasets.items():

            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')

            for iteration, (x, y) in enumerate(data_loader):

                x = to_var(x)
                y = to_var(y)

                x = x.view(-1, 784)
                y = y.view(-1, 1)

                if args.conditional:
                    recon_x, mean, log_var, z = vae(x, y)
                else:
                    recon_x, mean, log_var, z = vae(x)

                for i, yi in enumerate(y.data):
                    id = len(tracker_epoch)
                    tracker_epoch[id]['x'] = z[i, 0].item()
                    tracker_epoch[id]['y'] = z[i, 1].item()
                    tracker_epoch[id]['label'] = yi.item()


                loss = loss_fn(recon_x, x, mean, log_var)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if( tracker_global['loss'].dim() < 2 ):
                    tracker_global['loss'] = loss.data/x.size(0)
                else:
                    tracker_global['loss'] = torch.cat( (tracker_global['loss'], loss.data/x.size(0)) )

                if( tracker_global['it'].dim() < 2 ):
                    tracker_global['it'] = torch.Tensor([epoch*len(data_loader)+iteration])
                else:
                    tracker_global['it'] = torch.cat((tracker_global['it'], torch.Tensor([epoch*len(data_loader)+iteration])))

                if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                    print("Batch %04d/%i, Loss %9.4f"%(iteration, len(data_loader)-1, loss.item()))


                    if args.conditional:
                        c=to_var(torch.arange(0,10).long().view(-1,1))
                        x = vae.inference(n=c.size(0), c=c)
                    else:
                        x = vae.inference(n=10)

                    plt.figure()
                    plt.figure(figsize=(5,10))
                    for p in range(10):
                        plt.subplot(5,2,p+1)
                        if args.conditional:
                            plt.text(0,0,"c=%i"%c.data[p][0], color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(x[p].view(28,28).data.numpy())
                        plt.axis('off')


                    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                        if not(os.path.exists(os.path.join(args.fig_root))):
                            os.mkdir(os.path.join(args.fig_root))
                        os.mkdir(os.path.join(args.fig_root, str(ts)))

                    plt.savefig(os.path.join(args.fig_root, str(ts), "E%iI%i.png"%(epoch, iteration)), dpi=300)
                    plt.clf()
                    plt.close()


            df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            g = sns.lmplot(x='x', y='y', hue='label', data=df.groupby('label').head(100), fit_reg=False, legend=True)
            g.savefig(os.path.join(args.fig_root, str(ts), "E%i-Dist.png"%epoch), dpi=300)

            plt.close('all')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
