Code for [Deep Predictive Motion Tracking in Magnetic Resonance Imaging: Application to Fetal Imaging](https://arxiv.org/abs/1909.11625) to model robust spatiotemporal motion in fetal MRI

![tSNE projection of internal states of encoder lstm](./imgs/tsne_encoder_lstm_states_perplexity5.png)


### Model Architecture
![model_architecture](./imgs/model_architecture.png)

### Results
![results](./imgs/est_pred_imposed_slices_test29_msk5.png)

### Getting started
0. Install all the dependencies in python >v3.6 using `python setup.py install`
1. The `main.py` script runs all training and evaluation OR just evaluation using a pretrained model.
2. Simply configure the `sample_experiment.json` in `./experiments` directory and pass it as an argument to the main 
script
```python
>>> python main.py ./experiments/sample_experiment.json 
```
3. If step 2 is training then once the model is trained, the results of held out test set, a copy of experiments
.json and model itself will be persisted to given directory.
4. An example of how to configure experiments is put in ./experiments directory.
5. If a pretrained model path is provided then only evaluation will be done by loading the pretrained model otherwise
 training from scratch will be performed.

### Notes
* We have used CUDA implementation of LSTM which needs a NVIDIA GPU to train these models however in `
./models/our_model.py` you can switch `CuDNNLSTM` by generic `LSTM` to train on CPU. 
* In `./experiments/sample_experiment.json` you will find `network` section where you can define what kind of network
 you would like to train from `resnet18`, `vgg16`, `direct_lstm`, and `our_model` where in `our_model` you can 
 specify whether to train a `single_head: true` network or `single_head: false` model as well has size of layers 
 using `hidden_units: 512`.
* In data section of `./experiments/sample_experiment.json` you can also set `mask: true/false` to enable training 
models on masked/unmasked data.

> To download pretrained models please visit this [gdrive link](https://drive.google.com/drive/folders/1CCKWFLDZ-BoqmThGCpapcw7jBJ_a83cR?usp=sharing) and place them in preferably `./models` directory

