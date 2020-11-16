"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use('Agg')
import logging
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import get_model_path


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class LatentKeys(nn.Module):
    def __init__(self,conv_dim=64):
        super(LatentKeys, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(4):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        layers.append(nn.AvgPool2d(16,stride=1))

        self.main = nn.Sequential(*layers)
        self.dense = nn.Linear(1024,512*18)

    def forward(self,x):
        x = self.main(x)
        x = x.view(-1,1024)
        x = self.dense(x)
        return x


class pSpL(nn.Module):

    def __init__(self, opts):
        super(pSpL, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.latent_avg = None
        self.encoder = self.set_encoder()
        #self.latent_keys = LatentKeys()
        self.decoder = Generator(1024, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()


    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder


    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            #self.latent_keys.load_state_dict(get_keys(ckpt, 'latent_keys'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(get_model_path('ir_se50'))
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            #self.encoder2.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=18)


    def forward(self, x, y, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        codes_img = self.encoder(x)
        #codes_land = self.latent_keys(y)
        #codes_land = codes_land.view(-1,18,512)
        # normalize with respect to the center of an average face
        #codes = codes_img*codes_land
        images, result_latent = self.decoder([codes_img],
                                         input_is_latent=True,
                                         randomize_noise=randomize_noise,
                                         return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images


    def set_opts(self, opts):
        self.opts = opts


    def __load_latent_avg(self, ckpt, repeat=None):
        pass
