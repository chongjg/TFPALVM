from utils import *

test_worm = [1]
train_worm = [1]

infer = 1
gener = 1
random_init_index = 0
ucsFlag = False

neuron_holdout = ['ADAL', 'ADAR']

network = TFPALVM(Args(infer=infer,gener=gener,test_worm=test_worm, train_worm=train_worm, removeLen=30,constraint='unconstrained' if ucsFlag else 'weight'))
network.updateRandomInitIndex(random_init_index)
network.updateNeuronHoldout(neuron_holdout)
network.loadParam()

metrics = network.Test(OnlyLatent = True)

CorrData  = metrics['CorrData']
MSEData   = metrics['MSEData']

corr = np.sum(CorrData[0][CorrData[2] == 1]) / np.sum(CorrData[2])
mse = np.sum(MSEData[0][MSEData[2] == 1]) / np.sum(MSEData[2])

print(f'neuronholdout ({neuron_holdout}) -- infer : ' + str(infer) + ' gener : ' + str(gener))
print('corr      : ' + str(corr))
print('mse       : ' + str(mse))
