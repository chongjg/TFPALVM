import os
import sys
import torch
sys.path.append('../')
from model import *
from model import data_loader
import numpy as np
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / ((np.sum(a ** 2) * np.sum(b ** 2)) ** 0.5)
    return r

class Args:
    def __init__(self, neuron_holdout = [], test_worm = [1], train_worm = [1], model_type = 'conductance', constraint = 'weight', random_init_index = 0, infer = 0, gener = 0, removeLen = 0, ablatedNeurons = None):
        self.neuron_holdout = neuron_holdout
        self.test_worm = test_worm
        self.train_worm = train_worm
        self.model_type = model_type
        self.constraint = constraint
        self.random_init_index = random_init_index
        self.infer = infer
        self.gener = gener
        self.removeLen = removeLen
        self.ablatedNeurons = ablatedNeurons

class TFPALVM:

    def updateSavePath(self):
        args = self.args
        exp_name =  f'worm_vae_neuron_held_out_{args.neuron_holdout}_train_worm_{args.train_worm}_{args.model_type}_{args.constraint}_rand_{str(args.random_init_index)}_infer_{str(args.infer)}_gener_{str(args.gener)}'
        self.savepath = './checkpoints/'+exp_name+'_checkpoint_'

    def updateNeuronHoldout(self, neuron_holdout):
        self.args.neuron_holdout = neuron_holdout
        self.updateSavePath()
    
    def updateRandomInitIndex(self, random_init_index):
        self.args.random_init_index = random_init_index
        self.updateSavePath()

    def __init__(self, args):
        self.args = args
        self.updateSavePath()

        window_size = 784

        activity_path = '../data/worm_activity/'
        cnctm_path = '../data/worm_connectivity/'

        worm_data_loader = Worm_Data_Loader(activity_path)
        connectome_data = WhiteConnectomeData(cnctm_path, 'cuda', args.gener)
        self.connectome_data = connectome_data
        stim_channels = worm_data_loader.odor_channels
        acquisition_dt = worm_data_loader.step
        N = connectome_data.N
        self.N = N

        # Simulation hyparameters
        upsample_factor = 40
        self.upsample_factor = 40
        dt = acquisition_dt/upsample_factor

        # Training hyperparameters
        device = 'cuda'

        # Inference network
        # network architecture hyperparameters 
        inference_net_params = {'k1': 11, 'k2': 21, 'k3': 21, 'k4': 41, 'k5': 41, 'up1_factor': 4, 'up2_factor': 10}
        channels_dict = {'conv1': 2 * N,'conv2': 2 * N,'conv3': 2 * N, 'conv2d1' : 20, 'conv2d2' : 40, 'dilation' : 1}
        kernel_size_dict = {'conv1': inference_net_params['k1'],
                            'conv2': inference_net_params['k2'],
                            'conv3': inference_net_params['k3'],
                            'conv4': inference_net_params['k4'],
                            'conv5': inference_net_params['k5']}
        scale_factor_dict = {'upsample1': inference_net_params['up1_factor'], 'upsample2': inference_net_params['up2_factor']}
        inference_network = Worm_Inference_Network(
            neuron_num = N,
            inputs_channels = 2 * N,
            channels_dict = channels_dict,
            kernel_size_dict = kernel_size_dict,
            scale_factor_dict = scale_factor_dict,
            nonlinearity = F.relu,
            infer = args.infer)

        # Sensory encoder
        encoder = Worm_Sensory_Encoder(stim_channels,N)

        # Worm VAE : inference network + generative model
        initialization = {'neuron_bias': 1e-3*torch.rand((1, N)).to(device),
                        'neuron_tau': 1e-2*torch.rand((1, N)).to(device),
                        'calcium_bias': 1e-2*torch.rand((1, N)).to(device),
                        'voltage_to_calcium_nonlinearity': F.softplus}

        signs_c = torch.empty(N,N).uniform_(-0.01,0.01)
        self.network = WormVAE(
            connectome = connectome_data,
            window_size = window_size,
            initialization = initialization,
            upsample_factor = upsample_factor,
            signs_c = signs_c,
            encoder = encoder,
            inference_network = inference_network,
            device = device,
            dt = dt,
            nonlinearity = F.softplus if args.gener else F.leaky_relu,
            model_type = args.model_type,
            constraint =args.constraint,
            gener = args.gener)

        self.network.to(device)
        self.network.eval()

        upsampled_df_list = worm_data_loader.interpolate_odor(worm_data_loader.odor_worms[args.test_worm],upsample_factor)
        stim_features = torch.from_numpy(upsampled_df_list).to(torch.float32).cuda()
        target_raw = torch.from_numpy(worm_data_loader.activity_worms[args.test_worm]).to(torch.float32).cuda()
        full_target = connectome_data.make_full_connectome_activity(
            activity_data = target_raw,
            activity_neuron_list = worm_data_loader.neuron_names,
            holdout_list = [])
        full_target = full_target.to(torch.float32).cuda()
        target, missing_target = data_loader.generate_target_mask(full_target)
        train_dataset = data_loader.TimeSeriesDataloader4Test(data_param_dict = inference_net_params,
                                            data = [stim_features, target, missing_target],
                                            window_size = [int(window_size * upsample_factor), window_size, window_size])
        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False, drop_last = False, generator = torch.Generator(device='cuda'))

    def fluorescenceTensor(self):
        fluorescenceList = []
        missingwindowList = []
        for odor_input_window, target_window_padding, missing_window_padding, target_window, missing_window in self.trainloader:
            fluorescenceList.append(target_window_padding.detach().cpu().numpy())
            missingwindowList.append(missing_window_padding.detahc().cpu().numpy())
        return fluorescenceList, missingwindowList

    def loadParam(self, filename = None):
        if filename is None:
            filename = self.savepath + 'epoch_299.pt'
        network_params = torch.load(filename)
        self.network.load_state_dict(network_params)

    def independentInferEnable(self):
        for i in range(300):
            if self.args.infer:
                self.network.inference_network.conv1d2.weight.data[i, [idx for idx in range(3000-600) if idx % 300 != i]] *= 0
            else:
                self.network.inference_network.conv4.weight.data[i, [idx for idx in range(1200-600) if i % 300 != i]] *= 0

    def reverseIndependentInferEnable(self):
        for i in range(300):
            if self.args.infer:
                self.network.inference_network.conv1d2.weight.data[i, [idx for idx in range(3000-600) if idx % 300 == i]] *= 0
                self.network.inference_network.conv2d3.weight.data *= 0
                self.network.inference_network.conv2d3.bias.data *= 0
            else:
                self.network.inference_network.conv4.weight.data[i, [idx for idx in range(1200-600) if i % 300 == i]] *= 0

    def independentInfer(self, keep_dim = 0):
        if self.args.infer:
            return
        keep_dim = max(0, min(keep_dim, 600))
        for i in range(300):
            self.network.inference_network.conv4.weight.data[i, np.random.permutation(600)[keep_dim:]] *= 0

    def Metrics(self, predList, targetList, missing_windowList, latentList = []):
        upsample_factor = self.upsample_factor
        CorrData = np.zeros((3, 21, 300))
        MSEData  = np.zeros((3, 21, 300)) # 1st dimension -- 0:data 1:exist mask 2:holdout mask
        TraceData = np.zeros((2, 21, 300, 784)) # 1st dimension -- 0:target 1:pred
        if len(latentList) > 0:
            VoltageData = np.zeros((21, 300, 784*40))

        args = self.args
        removeLen = args.removeLen
        for idx in range(len(args.test_worm)):
            worm_id = args.test_worm[idx]
            pred = predList[idx][:,:,removeLen:]
            target = targetList[idx][:,:,removeLen:]
            missing_window = missing_windowList[idx][:,:,removeLen:]

            if len(latentList) > 0:
                latent = latentList[idx][:,:,removeLen*upsample_factor:]
                VoltageData[worm_id] = latent[0]
            TraceData[0,worm_id] = targetList[idx][0]
            TraceData[1,worm_id] = predList[idx][0]

            for neuron_id in range(self.N):
                pred_tmp   = pred[0,neuron_id,missing_window[0,neuron_id] < 0.5]
                target_tmp = target[0,neuron_id,missing_window[0,neuron_id] < 0.5]

                if len(target_tmp) > 1 and np.max(target_tmp) - np.min(target_tmp) > 1e-8:
                    MSEData[0, worm_id, neuron_id] = np.mean((pred_tmp - target_tmp) ** 2)
                    MSEData[1, worm_id, neuron_id] = 1
                    
                    CorrData[0, worm_id, neuron_id] = correlation(pred_tmp, target_tmp)
                    CorrData[1, worm_id, neuron_id] = 1
                    if neuron_id in [self.connectome_data.name_neuron_dict[name].index for name in args.neuron_holdout]:
                        CorrData[2, worm_id, neuron_id] = 1
                        MSEData[2, worm_id, neuron_id] = 1

        metrics = {
            'CorrData' : CorrData,
            'MSEData'  : MSEData,
            'TraceData': TraceData,
        }
        if len(latentList) > 0:
            metrics['VoltageData'] = VoltageData
        return metrics

    def Test(self, OnlyLatent = True, need_activity = False, ablation_neurons = []):
        upsample_factor = self.upsample_factor
        if not OnlyLatent:
            loss_sum = 0
            recon_sum = 0
            KLD_sum = 0
        fluorescence_targetList = []
        pred_fluorescenceList = []
        missing_windowList = []
        latentList = []

        for odor_input_window, target_window_padding, missing_window_padding, target_window, missing_window in self.trainloader:
            with torch.no_grad():
                torch.cuda.empty_cache()

                fluorescence_targetList.append(target_window.detach().cpu().numpy())
                missing_windowList.append(missing_window.detach().cpu().numpy())

                neuronidList = [self.connectome_data.name_neuron_dict[name].index for name in (self.args.neuron_holdout + ablation_neurons)]

                target_window_padding[:,neuronidList,:] = 0.0
                target_window[:,neuronidList,:] = 0.0
                missing_window_padding[:,neuronidList,:] = 1.0

                outputs = self.network.forward(fluorescence_full_target = target_window_padding,
                                            fluorescence_target = target_window,
                                            missing_target_mask = missing_window_padding,
                                            odor_input = odor_input_window,
                                            is_Training = False,
                                            OnlyLatent = OnlyLatent)

                pred_fluorescenceList.append(outputs['pred_fluorescence_mu'][:,:,::upsample_factor].detach().cpu().numpy())

                if not OnlyLatent:
                    latentList.append(outputs['voltage_latent_mu'].detach().cpu().numpy())
                    loss_minibatch, recon_likelihood_minibatch, KLD_latent_minibatch = self.network.loss(
                        fluorescence_trace_target = target_window,
                        missing_fluorescence_target =  missing_window,
                        outputs = outputs,
                        recon_weight = 1,
                        KLD_weight = 1)
                    loss_sum += loss_minibatch.detach().cpu().numpy()
                    recon_sum += recon_likelihood_minibatch.detach().cpu().numpy()
                    KLD_sum += KLD_latent_minibatch.detach().cpu().numpy()

        metrics = self.Metrics(pred_fluorescenceList, fluorescence_targetList, missing_windowList, latentList = latentList)

        if not OnlyLatent:
            num_worms = len(self.args.test_worm)
            metrics['loss_mean']  = -loss_sum / num_worms
            metrics['recon_mean'] = recon_sum / num_worms
            metrics['KLD_mean']   = KLD_sum / num_worms
        
        if need_activity:
            metrics['voltage_latent_mu'] = outputs['voltage_latent_mu'].detach().cpu().numpy()
            print(metrics['voltage_latent_mu'].shape)

        return metrics
    
    def nodeDegree(self):
        chem = (self.network.network_dynamics.sparsity_c.data * self.network.network_dynamics.magnitudes_c.data).detach().cpu().numpy()
        elec = (self.network.network_dynamics.sparsity_e.data * self.network.network_dynamics.magnitudes_e.data).detach().cpu().numpy()
        
        chem_degree = np.sum(chem, axis=0) + np.sum(chem, axis=1)
        elec_degree = np.sum(elec, axis=0)

        chem_index = np.argsort(chem_degree)
        elec_index = np.argsort(elec_degree)

        return chem_index, chem_degree, elec_index, elec_degree
        
neuronHoldoutList = [
    ['ADAL', 'ADAR'],
    ['ADEL', 'ADER'],
    ['ADFL', 'ADFR'],
    ['ADLL', 'ADLR'],
    ['AFDL', 'AFDR'],
    ['AIAL', 'AIAR'],
    ['AIBL', 'AIBR'],
    ['AIML', 'AIMR'],
    ['AINL', 'AINR'],
    ['AIYL', 'AIYR'],
    ['AIZL', 'AIZR'],
    ['ALA'],
    ['AQR'],
    ['ASEL', 'ASER'],
    ['ASGL', 'ASGR'],
    ['ASHL', 'ASHR'],
    ['ASIL', 'ASIR'],
    ['ASJL', 'ASJR'],
    ['ASKL', 'ASKR'],
    ['AUAL', 'AUAR'],
    ['AVAL', 'AVAR'],
    ['AVBL', 'AVBR'],
    ['AVDL', 'AVDR'],
    ['AVEL', 'AVER'],
    ['AVFL', 'AVFR'],
    ['AVHL', 'AVHR'],
    ['AVJL', 'AVJR'],
    ['AVKL', 'AVKR'],
    ['AVL'],
    ['AWAL', 'AWAR'],
    ['AWBL', 'AWBR'],
    ['AWCL', 'AWCR'],
    ['BAGL', 'BAGR'],
    ['CEPDL', 'CEPDR'],
    ['CEPVL', 'CEPVR'],
    ['DB02'],
    ['FLPL', 'FLPR'],
    ['I1L', 'I1R'],
    ['I2L', 'I2R'],
    ['I3'],
    ['I4'],
    ['I6'],
    ['IL1L', 'IL1R'],
    ['IL1DL', 'IL1DR'],
    ['IL1VL', 'IL1VR'],
    ['IL2L', 'IL2R'],
    ['IL2DL', 'IL2DR'],
    ['IL2VL', 'IL2VR'],
    ['M1'],
    ['M2L', 'M2R'],
    ['M3L', 'M3R'],
    ['M4'],
    ['M5'],
    ['MCL', 'MCR'],
    ['MI'],
    ['NSML', 'NSMR'],
    ['OLLL', 'OLLR'],
    ['OLQDL', 'OLQDR'],
    ['OLQVL', 'OLQVR'],
    ['RIAL', 'RIAR'],
    ['RIBL', 'RIBR'],
    ['RICL', 'RICR'],
    ['RID'],
    ['RIML', 'RIMR'],
    ['RIR'],
    ['RIS'],
    ['RIVL', 'RIVR'],
    ['RMDL', 'RMDR'],
    ['RMDDL', 'RMDDR'],
    ['RMDVL', 'RMDVR'],
    ['RMEL', 'RMER'],
    ['RMED'],
    ['RMEV'],
    ['RMFL', 'RMFR'],
    ['RMGL', 'RMGR'],
    ['RMHL', 'RMHR'],
    ['SAADL', 'SAADR'],
    ['SAAVL', 'SAAVR'],
    ['SABVL', 'SABVR'],
    ['SIADL', 'SIADR'],
    ['SIAVL', 'SIAVR'],
    ['SIBDL', 'SIBDR'],
    ['SIBVL', 'SIBVR'],
    ['SMBDL', 'SMBDR'],
    ['SMBVL', 'SMBVR'],
    ['SMDDL', 'SMDDR'],
    ['SMDVL', 'SMDVR'],
    ['URADL', 'URADR'],
    ['URAVL', 'URAVR'],
    ['URBL', 'URBR'],
    ['URXL', 'URXR'],
    ['URYDL', 'URYDR'],
    ['URYVL', 'URYVR'],
    ['VA01'],
    ['VB01'],
    ['VB02'],
]

def standard_error(data):
    std = np.std(data)
    SE = std / (len(data) ** 0.5)
    return SE

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
