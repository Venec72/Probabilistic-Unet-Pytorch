import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from numpy import zeros,array
import os


lr = 1e-4
epochs = 50

out_dir = 'outputs/test2'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('Folder already exists. Existing models and training logs will be replaced')


# logging

# lié à la partie 'end of validation' à la fin des epochs, il faut encore comprendre ce que ça fait
train_loss = []
test_loss = []
best_val_loss = 999.0



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("device is %s" %(device))

dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)      # si on veut vérifier qu'on obtient le même résultat que l'autre code que nous avons trouvé, il faudra enlever le shuffle et comparer



train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)    
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0)

"""
# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
secheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
"""

save_loss = zeros(epochs, int)

for epoch in range(epochs):
    net.train()
    loss_train = 0
    loss_segmentation = 0

    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.detach().cpu().item()

        if step%10==0:
            print('[Ep ', epoch+1, (step+1), ' of ', len(train_loader) ,'] train loss: ', loss_train/(step+1))
            #print ("Epoch %d, step %d" %(epoch,step))




    # end of training loop
    loss_train /= len(train_loader)
    
    # valdiation loop
    net.eval()
    loss_val = 0
    
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(test_loader): 
            patch = patch.to(device)
            mask = mask.to(device)
            mask = torch.unsqueeze(mask,1)
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss_train = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss_train

            loss_val += loss.detach().cpu().item()
            

    # end of validation

    loss_val /= len(test_loader)
    
    train_loss.append(loss_train)
    test_loss.append(loss_val)
    
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)   
    


    # save best model checkpoint
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        fname = 'model_dict.pth'
        torch.save(net.state_dict(), os.path.join(out_dir, fname))
        print('model saved at epoch: ', epoch+1)

    """
    state = {
    'epoch' : epoch +1,
    'state_dict' : net.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'loss' : loss,
    }
    torch.save(state, 'checkpoint/checkpoint.pth.tar')
    """
# save_loss[epoch] = loss

# torch.save(net.state_dict(), 'model/net1.pt')



print('Finished training')
# save loss curves        
plt.figure()
plt.plot(train_loss)
plt.title('train loss')
fname = os.path.join(out_dir,'loss_train.png')
plt.savefig(fname)
plt.close()

plt.figure()
plt.plot(test_loss)
plt.title('val loss')
fname = os.path.join(out_dir,'loss_val.png')
plt.savefig(fname)
plt.close()
# plt.show()

# Saving logs
log_name = os.path.join(out_dir, "logging.txt")
with open(log_name, 'w') as result_file:
    result_file.write('Logging... \n')
    result_file.write('Validation loss ')
    result_file.write(str(test_loss))
    result_file.write('\nTraining loss  ')
    result_file.write(str(train_loss))