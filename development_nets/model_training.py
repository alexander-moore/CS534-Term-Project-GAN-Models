# Model training
# Each model type needs to be trained:
# - supervised learning
# - conditional AE
# - ACGAN
# - AE ACGAN

# Ideally each one is self-contained, and we send in data and parameters and it returns trained models and testable outcomes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Global Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print_stride = 10

n_epochs = 200 # 100? idk check graphs
lr = 0.002
beta1 = 0.5
bs = 32

print_samp = True

criterion = nn.BCELoss()
supervised_criterion = nn.BCELoss()
reconstruction_criterion = nn.MSELoss()
accuracy = torchmetrics.Accuracy()

def data(dset = 'CIFAR', amount = 1.0):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Permute(2,0,1), # NOTE?
        torchvision.transforms.Resize(64),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

    #train_dataset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=True, transform=transform, download=True)
    #test_dataset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=False, transform=transform, download=False)

    if dset == 'CIFAR':
        train_dataset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=False, download=False)

        Xtr = torchvision.transforms.Resize(64)(torchvision.transforms.Normalize(mean=[0.5], std=[0.5])(torch.FloatTensor(train_dataset.data).permute(0,3,1,2)))
        ytr = F.one_hot(torch.FloatTensor(train_dataset.targets).to(torch.int64), num_classes = 10).unsqueeze(-1).unsqueeze(-1).float()
        ytr_int = train_dataset.targets

        Xte = torchvision.transforms.Resize(64)(torchvision.transforms.Normalize(mean=[0.5], std=[0.5])(torch.FloatTensor(test_dataset.data).permute(0,3,1,2)))
        yte = F.one_hot(torch.FloatTensor(test_dataset.targets).to(torch.int64), num_classes = 10).unsqueeze(-1).unsqueeze(-1).float()
        yte_int = test_dataset.targets

        # Downsample if desired
        k = int(Xtr.shape[0]*amount)
        perm = torch.randperm(Xtr.shape[0])
        idx = perm[:k]
        Xtr = Xtr[idx]
        ytr = ytr[idx]

        train_loader = torch.utils.data.DataLoader(dataset=list(zip(Xtr, ytr, ytr_int)), batch_size=bs, shuffle=True, num_workers = 1)
        test_loader = torch.utils.data.DataLoader(dataset=list(zip(Xte, yte, yte_int)), batch_size=1024, shuffle=True, num_workers = 1)


    if dset == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, download=True)

        Xtr = torchvision.transforms.Resize(64)(train_dataset.data.float()).unsqueeze(1)
        ytr = F.one_hot(train_dataset.targets, num_classes = 10).unsqueeze(-1).unsqueeze(-1).float()
        ytr_int = train_dataset.targets

        Xte = torchvision.transforms.Resize(64)(test_dataset.data.float()).unsqueeze(1)
        yte = F.one_hot(test_dataset.targets, num_classes = 10).unsqueeze(-1).unsqueeze(-1).float()
        yte_int = test_dataset.targets

        # Want yte_int to be like "1 if y==2 else 0"
        new_yint_tr = []
        for ele in ytr_int:
            new_yint_tr.append(1 if ele == 2 else 0)
        new_yint_tr = torch.FloatTensor(new_yint_tr)

        new_yint_te = []
        for ele in yte_int:
            new_yint_te.append(1 if ele == 2 else 0)
        new_yint_te = torch.FloatTensor(new_yint_te)

        
        # Downsample if desired
        k = int(Xtr.shape[0]*amount)
        perm = torch.randperm(Xtr.shape[0])
        idx = perm[:k]
        Xtr = Xtr[idx]
        ytr = ytr[idx]

        train_loader = torch.utils.data.DataLoader(dataset=list(zip(Xtr, new_yint_tr)), batch_size=bs, shuffle=True, num_workers = 1)
        test_loader = torch.utils.data.DataLoader(dataset=list(zip(Xte, new_yint_te)), batch_size=1024, shuffle=False, num_workers = 1)

    return train_loader, test_loader

def train_supervised(model, train_loader, test_loader):
    print(device)
    # Trains and returns a supervised model optimized on input data w hyperparameters given by gloval
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []
    for epoch in range(n_epochs):

        for i, data in enumerate(train_loader, 0):

            for param in model.parameters():
                param.grad = None

            x = data[0].to(device)
            label = data[1].to(device)

            loss = supervised_criterion(model(x).squeeze(), label)
            loss.backward()
            opt.step()

        if epoch % print_stride == 0:

            training_curve.append(loss.data.item())

            testbce, testacc = test_model(model, test_loader)
            testing_curve.append(testbce)
            testing_acc.append(testacc)

    return model, training_curve, testing_curve, testing_acc

def train_conditional_autoencoder(model, train_loader, test_loader):
    print(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []
    for epoch in range(n_epochs):

        for i, data in enumerate(train_loader, 0):

            for param in model.parameters():
                param.grad = None

            x = data[0].to(device)
            label = data[1].to(device)

            (z,c), xhat = model(x)
            #print(z.shape, c.shape, xhat.shape, x.shape)

            # Supervised
            supervised_loss = supervised_criterion(c, label)

            # Reconstructive
            reconstruction_loss = reconstruction_criterion(xhat, x)

            loss = supervised_loss + reconstruction_loss
            loss.backward()
            opt.step()

        if epoch % print_stride == 0:

            training_curve.append(supervised_loss.data.item())

            testbce, testacc = test_model(model, test_loader)
            testing_curve.append(testbce)
            testing_acc.append(testacc)

    return model, training_curve, testing_curve, testing_acc

def train_vae(model, train_loader, test_loader, lam = 0.0):
    print(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []
    for epoch in range(n_epochs):

        for i, data in enumerate(train_loader, 0):

            for param in model.parameters():
                param.grad = None

            x = data[0].to(device)
            label = data[1].to(device)

            xhat, (mu, sd) = model(x)

            # Supervised
            #supervised_loss = supervised_criterion(c, label)

            # KLD
            kld_loss = kl = torch.mean(-0.5 * torch.sum(1 + sd - mu ** 2 - sd.exp(), dim = 1), dim = 0)

            # Reconstructive
            reconstruction_loss = reconstruction_criterion(xhat, x)

            loss = reconstruction_loss + lam*kld_loss
            loss.backward()
            opt.step()

        if epoch % print_stride == 0:

            print(reconstruction_loss.data.item(), kld_loss.data.item())

            # print_sample
            with torch.no_grad():
                rand_v = torch.randn((9, model.zdim, 1, 1), device = device)                
                X_fake = model.Dec(rand_v)

                for i in range(9):
                    plt.subplot(330 + 1 + i)
                    element = X_fake[i].permute(1,2,0) + 1
                    element = element / 2 # this gets it to [0,1]
                    plt.imshow(element.cpu())
                plt.show()

    return model, training_curve, testing_curve, testing_acc

# 2nd
def train_acgan(G, D, train_loader, test_loader):
    G.to(device)
    D.to(device)

    Gopt = torch.optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))
    Dopt = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []

    hflipper = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    resize_cropper = torchvision.transforms.RandomResizedCrop((64,64), scale=(0.88,1.0), ratio=(0.8,1.2))
    rotation = torchvision.transforms.RandomRotation(degrees=(-5,5))

    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []

        for i, (X,code) in enumerate(train_loader, 0):
            #print(X, code)
            
            X = hflipper(X)

            X = resize_cropper(X)

            X = rotation(X)

            mini_batch = X.size()[0]
            
            X = X.to(device)
            code = code.to(device).float()
            #code = F.one_hot(code).float()
            
            valid = Variable(torch.FloatTensor(mini_batch, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.FloatTensor(mini_batch, 1).fill_(0.0), requires_grad=False).to(device)

            ## Discriminator Training
            for param in D.parameters():
                param.grad = None

            rand_v = torch.randn((mini_batch, G.zdim, 1, 1), device = device)
            rand_c = F.one_hot(torch.randint(low=0, high=10, size = (mini_batch,), device = device), num_classes = G.cdim).view(mini_batch, G.cdim, 1, 1).float()
            
            X_fake = G(rand_v, rand_c)
                
            real_pred, real_aux = D(X)
            d_real_validity_loss = criterion(real_pred.squeeze(), valid.squeeze())
            d_real_supervised_loss = supervised_criterion(real_aux.squeeze(), code.squeeze())

            d_real_loss = (d_real_validity_loss + d_real_supervised_loss) / 2
            #d_real_loss = d_real_validity_loss
            
            fake_pred, fake_aux = D(X_fake.detach())
            d_fake_validity_loss = criterion(fake_pred.squeeze(), fake.squeeze())
            d_fake_loss =  d_fake_validity_loss + supervised_criterion(fake_aux.squeeze(), rand_c.squeeze())
            d_fake_loss = d_fake_loss / 2
            #d_fake_loss = d_fake_validity_loss
            
            d_loss = (d_real_loss + d_fake_loss) / 2

            D_loss = d_loss
            D_loss.backward()
            Dopt.step()

            d_validity_losses = d_real_validity_loss + d_fake_validity_loss
            d_losses.append(d_validity_losses.data)
            
            ## Generator Training
            for param in G.parameters():
                param.grad = None
                
            
            
            validity, pred_label = D(X_fake)
            g_validity_loss = criterion(validity.squeeze(), valid.squeeze())
            G_loss = 0.5*( g_validity_loss + supervised_criterion(pred_label.squeeze(), rand_c.squeeze()))
            #G_loss = g_validity_loss
            
            model_loss = G_loss
            model_loss.backward()
            Gopt.step()

            g_losses.append(g_validity_loss.data)
            
            
        if epoch % print_stride == 0:
            print('d loss', torch.mean(torch.FloatTensor(d_losses)))
            print('g loss', torch.mean(torch.FloatTensor(g_losses)))

            #training_curve.append(d_real_supervised_loss.data.item())

            if test_loader is not None:
                testbce, testacc = test_model(D, test_loader)
                testing_curve.append(testbce)
                testing_acc.append(testacc)

            if print_samp == True:
                with torch.no_grad():
                    rand_v = torch.randn((9, G.zdim, 1, 1), device = device)
                    rand_c = F.one_hot(torch.randint(low=0, high=G.cdim, size = (9,), device = device), num_classes = G.cdim).view(9, G.cdim, 1, 1).float()
                    
                    X_fake = G(rand_v, rand_c)
                    #print(X_fake.shape)

                    classes = ['entities_humanoid', 'entities_humanoid_human', 'entities_humanoid_humanlike', 
                    'entities_humanoid_humanlike_demonspawn', 'entities_humanoid_humanlike_spriggan', 
                    'entities_humanoid_undead', 'entities_nonhumanoid', 'entities_nonhumanoid_aberration', 
                    'entities_nonhumanoid_abyss', 'entities_nonhumanoid_amorphous', 'entities_nonhumanoid_animals', 
                    'entities_nonhumanoid_aquatic', 'entities_nonhumanoid_demons', 'entities_nonhumanoid_Dragon', 
                    'entities_nonhumanoid_eyes', 'entities_nonhumanoid_fungi_plants', 'entities_nonhumanoid_half-human', 
                    'entities_nonhumanoid_holy', 'entities_nonhumanoid_log', 'entities_nonhumanoid_nonliving']

                    classes = [x[9:] for x in classes]

                    c = rand_c.squeeze()
                    #print(c, c.shape)
                    q = torch.argmax(c, dim = 1)
                    #print(q.shape)

                    for i in range(9):
                        #c = rand_c.squeeze()
                        #print(c, c.shape)
                        #q = torch.argmax(c[i])
                        #print(q)
                        plt.subplot(330 + 1 + i).set_title(str(classes[q[i]]))
                        # plot raw pixel data
                        #print(X_fake[i].shape)
                        element = X_fake[i].permute(1,2,0) + 1
                        element = element / 2 # this gets it to [0,1]
                        plt.imshow(element.cpu())
                    plt.show()
                

    return D, training_curve, testing_curve, testing_acc

def train_caegan(AE, D, train_loader, test_loader):
    AE.to(device)
    D.to(device)

    AEopt = torch.optim.Adam(AE.parameters(), lr = lr, betas = (beta1, 0.999))
    Dopt = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []

    for epoch in range(n_epochs):

        for i, (X,code) in enumerate(train_loader, 0):

            mini_batch = X.size()[0]
            
            X = X.to(device)
            code = code.to(device).float()
            #code = F.one_hot(code).float()
            
            valid = Variable(torch.FloatTensor(mini_batch, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.FloatTensor(mini_batch, 1).fill_(0.0), requires_grad=False).to(device)

            ## Discriminator Training
            for param in D.parameters():
                param.grad = None

            real_pred = D(X)
            d_real_loss = criterion(real_pred.squeeze(), valid.squeeze())

            rand_v = torch.randn((mini_batch, AE.zdim, 1, 1), device = device)
            rand_c = F.one_hot(torch.randint(low=0, high=10, size = (mini_batch,), device = device), num_classes = AE.cdim).view(mini_batch, AE.cdim, 1, 1).float()
            
            X_fake = AE.Dec(rand_v, rand_c)
            fake_pred = D(X_fake)

            d_fake_loss = criterion(fake_pred.squeeze(), fake.squeeze())
            
            D_loss = d_real_loss + d_fake_loss
            D_loss.backward(retain_graph = True)
            Dopt.step()

            
            ## Generator Training
            for param in AE.parameters():
                param.grad = None
            

            # Reconstruction Loss    
            (_, chat), Xhat = AE(X)
            recon_loss = reconstruction_criterion(Xhat, X)
            # Supervised Loss
            supervised_loss = supervised_criterion(chat, code)
            # Adversarial Loss
            validity = D(X_fake)
            G_loss = criterion(validity.squeeze(), valid.squeeze())
            
            model_loss = G_loss + recon_loss + supervised_loss
            model_loss.backward(retain_graph = True)
            AEopt.step()
            
            
            
        if epoch % print_stride == 0:

            training_curve.append(supervised_loss.data.item())

            testbce, testacc = test_model(AE, test_loader)
            testing_curve.append(testbce)
            testing_acc.append(testacc)

    return AE, training_curve, testing_curve, testing_acc

#CAE_ACGAN
def train_cae_acgan(G, D, train_loader, test_loader):
    G.to(device)
    D.to(device)

    Gopt = torch.optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))
    Dopt = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))

    training_curve = []
    testing_curve = []
    testing_acc = []

    for epoch in range(n_epochs):

        for i, (X,code) in enumerate(train_loader, 0):

            mini_batch = X.size()[0]
            
            X = X.to(device)
            code = code.to(device).float()
            #code = F.one_hot(code).float()
            
            valid = Variable(torch.FloatTensor(mini_batch, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.FloatTensor(mini_batch, 1).fill_(0.0), requires_grad=False).to(device)

            ## Discriminator Training
            for param in D.parameters():
                param.grad = None

            rand_v = torch.randn((mini_batch, G.zdim, 1, 1), device = device)
            rand_c = F.one_hot(torch.randint(low=0, high=10, size = (mini_batch,), device = device), num_classes = G.cdim).view(mini_batch, G.cdim, 1, 1).float()
            
            X_fake = G(rand_v, rand_c)
                
            real_pred, real_aux = D(X)
            d_real_validity_loss = criterion(real_pred.squeeze(), valid.squeeze())
            d_real_supervised_loss = supervised_criterion(real_aux.squeeze(), code.squeeze())

            d_real_loss = (d_real_validity_loss + d_real_supervised_loss) / 2
            
            fake_pred, fake_aux = D(X_fake.detach())
            d_fake_loss = criterion(fake_pred.squeeze(), fake.squeeze())
            
            d_loss = (d_real_loss + d_fake_loss) / 2

            D_loss = d_loss
            D_loss.backward()
            Dopt.step()

            
            ## Generator Training
            for param in G.parameters():
                param.grad = None
                
            
            
            validity, pred_label = D(X_fake)
            G_loss = 0.5*(criterion(validity.squeeze(), valid.squeeze()) + supervised_criterion(pred_label.squeeze(), rand_c.squeeze()))
            
            model_loss = G_loss
            model_loss.backward()
            Gopt.step()
            
            
            
        if epoch % print_stride == 0:

            training_curve.append(d_real_supervised_loss.data.item())

            testbce, testacc = test_model(D, test_loader)
            testing_curve.append(testbce)
            testing_acc.append(testacc)

    return D, training_curve, testing_curve, testing_acc



def test_model(model, data_loader):

    metric = torchmetrics.Accuracy()

    with torch.no_grad():
        loss, acc = 0, 0
        n = 0
        for data in data_loader:
            n+=1

            x = data[0]
            y = data[1]
            #yint = data[2]
            yhat = model.predict(x.to(device))

            # Could do something cute like know what metrics to return based on what input-output is
            loss += supervised_criterion(yhat.squeeze(), y.to(device)).data.item()

            #acc = metric(torch.round(yhat.squeeze().cpu()).int(), y)

    return loss/n, 0#metric.compute()#, acc/i