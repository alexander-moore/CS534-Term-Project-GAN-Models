import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)

class Decoder(nn.Module):
    # Adapted from g_arches.py
    def __init__(self, nf, nc, zdim, cdim):
        super(Decoder, self).__init__()
        ## Decoding:
        self.deconv1v = nn.ConvTranspose2d(zdim, nf*8, 4, 1, 0, bias = False) # Not sure how this looks
        
        self.deconv1_bn = nn.BatchNorm2d(nf*8)
        self.deconv2 = nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias = False)
        self.deconv2_bn = nn.BatchNorm2d(nf*4)
        self.deconv3 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias = False)
        self.deconv3_bn = nn.BatchNorm2d(nf*2)
        self.deconv4 = nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias = False)
        self.deconv4_bn = nn.BatchNorm2d(nf)
        #self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1, 0)
        self.deconv5 = nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias = False)
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, v):
        x = self.deconv1_bn(self.deconv1v(v))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x)) # this output is correct given [0.5] scale in preprocess
        return x

class Conditional_Decoder(nn.Module):
    # Adapted from g_arches.py
    def __init__(self, nf, nc, zdim, cdim):
        super(Conditional_Decoder, self).__init__()
        self.nf = nf
        self.nc = nc
        self.zdim = zdim
        self.cdim = cdim
        ## Decoding:
        self.deconv1v = nn.ConvTranspose2d(zdim, nf*8, 4, 1, 0, bias = False) # Not sure how this looks
        self.deconv1c = nn.ConvTranspose2d(cdim, nf*8, 4, 1, 0, bias = False) # Input: (bs, cdim+v_dim, 1, 1)
        
        self.deconv1_bn = nn.BatchNorm2d(nf*8)
        self.deconv2 = nn.ConvTranspose2d(nf*8*2, nf*4, 4, 2, 1, bias = False)
        self.deconv2_bn = nn.BatchNorm2d(nf*4)
        self.deconv3 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias = False)
        self.deconv3_bn = nn.BatchNorm2d(nf*2)
        self.deconv4 = nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias = False)
        self.deconv4_bn = nn.BatchNorm2d(nf)
        #self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1, 0)
        self.deconv5 = nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias = False)
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, v, c):
        v = self.deconv1_bn(self.deconv1v(v))
        c = self.deconv1_bn(self.deconv1c(c))
        x = torch.cat((v, c), dim = 1) #stack on channel dim, should be (bs, vdim+cdim, 1, 1). Not sure here
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x)) # this output is correct given [0.5] scale in preprocess
        return x

class Encoder(nn.Module):
    # Taken from `https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html` : this is a Discriminator: likely need
    # to generalize to some desired output dimension to be an "encoder"
    # an encoder may also be a supervised learner with zdim = cdim

    # Encoder(zdim) to z
    # Discriminator(1) to p
    # Supervised(cdim) to c
    def __init__(self, nf, nc, zdim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias = False)
        self.conv2 = nn.Conv2d(nf, nf*2, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(nf*2)
        self.conv3 = nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(nf*4)
        self.conv4 = nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(nf*8)
        self.conv5 = nn.Conv2d(nf*8, zdim, 5, 2, 1) # output (n,100,1,1) #5?

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x

    def predict(self, x):
        return self.forward(x)

class Conditional_Encoder(nn.Module):
    def __init__(self, nf, nc, zdim, cdim, enc_activation = torch.nn.Sigmoid()): # For an AC Discriminator, zdim = 1
        super(Conditional_Encoder, self).__init__()
        
        # Discriminator:
        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias = False) # (bs, 3 + , img_size, img_size)
        self.conv2 = nn.Conv2d(nf, nf*2, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(nf*2)
        self.conv3 = nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(nf*4)
        self.conv4 = nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(nf*8)

        self.conv5_enc = nn.Conv2d(nf*8, zdim, 5, 2, 1)
        self.conv5_aux = nn.Conv2d(nf*8, cdim, 5, 2, 1)
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        encoding = enc_activation(self.conv5_enc(x))
        label = torch.Softmax()(self.conv5_aux(x))
        return encoding, label

    def predict(self, x):
        z, c = self.forward(x)
        return c

class VAE_Encoder(nn.Module):
    def __init__(self, nf, nc, zdim): # For an AC Discriminator, zdim = 1
        super(VAE_Encoder, self).__init__()
        
        # Discriminator:
        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias = False) # (bs, 3 + , img_size, img_size)
        self.conv2 = nn.Conv2d(nf, nf*2, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(nf*2)
        self.conv3 = nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(nf*4)
        self.conv4 = nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(nf*8)

        self.conv5_mu = nn.Conv2d(nf*8, zdim, 5, 2, 1)
        self.conv5_sd = nn.Conv2d(nf*8, zdim, 5, 2, 1)
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        mu = self.conv5_mu(x)
        sd = self.conv5_sd(x)
        return mu, sd

class VAE_Decoder(nn.Module):
    # Adapted from g_arches.py
    def __init__(self, nf, nc, zdim):
        super(VAE_Decoder, self).__init__()
        ## Decoding:
        self.deconv1v = nn.ConvTranspose2d(zdim, nf*8, 4, 1, 0, bias = False) # Not sure how this looks
        
        self.deconv1_bn = nn.BatchNorm2d(nf*8)
        self.deconv2 = nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias = False)
        self.deconv2_bn = nn.BatchNorm2d(nf*4)
        self.deconv3 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias = False)
        self.deconv3_bn = nn.BatchNorm2d(nf*2)
        self.deconv4 = nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias = False)
        self.deconv4_bn = nn.BatchNorm2d(nf)
        self.deconv5 = nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias = False)
    
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
      
    def forward(self, v):
        x = self.deconv1_bn(self.deconv1v(v))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x)) # this output is correct given [0.5] scale in preprocess
        return x

class VAE(nn.Module):
    def __init__(self, nf, nc, zdim):
        super(VAE, self).__init__()
        self.zdim = zdim

        self.Enc = VAE_Encoder(nf, nc, zdim)
        self.Dec = VAE_Decoder(nf, nc, zdim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        # encoding
        mu, sd = self.Enc(x)
        z = self.reparameterize(mu, sd)

        # decoding
        x = self.Dec(z)

        return x, (mu, sd)


class Conditional_Autoencoder(nn.Module):
    def __init__(self, nf, nc, zdim, cdim):
        super(Conditional_Autoencoder, self).__init__()
        self.cdim = cdim
        self.zdim = zdim

        self.Enc = Conditional_Encoder(nf, nc, zdim, cdim)
        self.Dec = Conditional_Decoder(nf, nc, zdim, cdim)

    def forward(self, input):
        # Likely want to return ((z,c), x)
        z, c = self.Enc(input)
        return (z,c), self.Dec(z,c)

    def predict(self, input):
        z, c = self.Enc(input)
        return c