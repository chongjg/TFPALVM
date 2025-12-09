from utils import *

test_worm = [i for i in range(6)]
train_worm = [i for i in range(6,21,1)]

infer = 1
gener = 1
random_init_index = 0
ucsFlag = False

network = TFPALVM(Args(infer=infer,gener=gener,test_worm=test_worm, train_worm=train_worm, constraint='unconstrained' if ucsFlag else 'weight'))
network.updateRandomInitIndex(random_init_index)
network.loadParam()

metrics = network.Test(OnlyLatent = False)

CorrData  = metrics['CorrData']
MSEData   = metrics['MSEData']

corr = np.sum(CorrData[0][CorrData[1] == 1]) / np.sum(CorrData[1])
mse  = np.sum(MSEData[0][MSEData[1] == 1]) / np.sum(MSEData[1])
loss_mean  = metrics['loss_mean']
recon_mean = metrics['recon_mean']
KLD_mean   = metrics['KLD_mean']

print('wormholdout -- infer : ' + str(infer) + ' gener : ' + str(gener))
print('corr      : ' + str(corr))
print('mse       : ' + str(mse))
print('loss_mean : ' + str(loss_mean))
print('recon_mean: ' + str(recon_mean))
print('KLD_mean  : ' + str(KLD_mean))
