## Temporal-Filter and Physiology-Aligned Latent Variable Model (TF-PALVM)

This is the official PyTorch implementation of the paper: ["Temporal-Filter Enhanced Prediction of Whole-Brain Neural Activity Using the Physiology-Aligned Latent Variable Model"]()

### Abstract

<div align="center">
<img src=".\assets\diagram.png" alt="diagram" width="75%" />
</div>

Understanding the neural mechanisms behind intelligent behaviors in nematodes requires a comprehensive framework that integrates connectome data with recordings of all neuronal activities. However, obtaining full membrane potentials for all labeled neurons is challenging, and even in C. elegans, which has only 302 neurons, current calcium imaging techniques typically capture just over half of these labeled neurons in vivo. Existing advances primarily focus on using spatial correlation across the whole brain to causally predict unknown neuronal activities by using measured activities of half-numbered neurons, but they struggle to achieve high prediction accuracy. By introducing a quasi-independent temporal coding property of populational neurons in living brains, e.g., C. elegans, we establish the Temporal-Filter and Physiology-Aligned Latent Variable Model (TF-PALVM), a new algorithmic leap that synergizes the spatial connectome structure with temporal coding functions of individual neurons to infer the electrical activities of the entire neural ensemble. This model employs an autoencoder network with temporal kernels, rather than relying on spatial correlations, refined to reflect individual neuronal temporal coding functions and predict their future temporal activities, while embedding experimentally derived synaptic weights into a biologically coherent framework. When tested, our model demonstrates unprecedented reconstruction accuracy, surpassing existing models by approximately 75% in the worm holdout evaluation and 51% in neuron holdout performance. Moreover, it precisely predicts synaptic polarities, with 75% of them to be excitatory, matching experimental excitatory synapse data. It is also able to identify the top neuron pairs with the most influence on behavioral correlations, consistent with previous experimental research. Our TF-PALVM stands as a transformative tool for neuroscientific exploration, capable of predicting missing neuronal activity with high fidelity. The success of the model confirms that the temporal response history of individual neurons contains more valuable information than the population network in predicting their present and future responses to sensory inputs. It offers a scalable approach to potentially unravel the complexities of larger, more intricate brains.

### Installation

Download this repository and create the environment:

```bash
git clone https://github.com/chongjg/TFPALVM.git
cd TFPALVM
conda env create -f environment.yml
conda activate TFPALVM
```

### Getting Start

* Training:
    ```bash
    cd experiment
    python -u main.py --neuron_holdout list_of_neuron_holdout --train_worm list_of_worm_id --constraint constraint --infer inference_network_type --gener generative_network_type --random_init_index random_init_index 
    ```
  * `--neuron_holdout` defines the set of neurons to be held out during training. You can specify a list of neuron names to choose which neurons (typically a pair of symmetric neurons) to be held out, e.g. `--neuron_holdout 'ADAL','ADAR'`.
  * `--train_worm` defines the set of worms to be trained. You can specify a list of worm ids to choose which worms to be trained on, e.g. `--train_worm 1`.
  * `--constraint` defines the constraint to synapse weight. `--constraint weight` indicates using connectome synapse count to constrain the weight; `--constraint unconstrained` indicates without connectome constrained, the weight is fully-connected matrix and the magnitude is trainable.
  * `--infer` defines the inference network type. `--infer 0` uses the original CCLVM inference network; `--infer 1` enables the proposed temporal-filter inference network.
  * `--gener` defines the generative network type. `--gener 0` uses the original CCLVM generative network; `--gener 1` enables the proposed physiology-aligned generative network.
  * `--random_init_index` defines the index of random initialization in each training trial. In the paper, we use 4 different random initializations to evaluate the models, e.g. `--random_init_index 0`.

```bash
# neuron holdout training : holdout ADAL & ADAR
python -u main.py --neuron_holdout 'ADAL' 'ADAR' --train_worm 1 --constraint weight --infer 1 --gener 1 --random_init_index 0
# worm holdout training :
python -u main.py --train_worm 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  --constraint weight --infer 1 --gener 1 --random_init_index 0
```

* Evaluation:

```bash
# neuron holdout evaluation:
python -u test_neuronholdout.py

# worm holdout evaluation:
python -u test_wormholdout.py
```

### Results

* Reconstruction results for different neuron categories in worm holdout evaluation. 

  <div align="center">
  <img src=".\assets\wormholdout.png" alt="wormholdout" width="75%" />
  </div>

* Reconstruction results for different neuron categories in neuron holdout evaluation. 

  <div align="center">
  <img src=".\assets\neuronholdout.png" alt="neuronholdout" width="75%" />
  </div>

* Prediction of reversal potential, synaptic polarity, and time constant.

  (a) Predicted reversal potential distributions in TF-PALVM and CCLVM.

  (b) Comparison of excitatory–inhibitory ratios from experiments and models.

  (c) Time constant distributions from TF-PALVM, CCLVM, and OpenWorm.

  (d) Synaptic polarity map of the C. elegans neural network (color: E/I ratio; width: strength).

  <div align="center">
  <img src=".\assets\physiology-align.png" alt="physiology-align" width="75%" />
  </div>

### Acknowledgement

This project is built on [wormvae](https://github.com/TuragaLab/wormvae). Thanks for their excellent work.