# DeepMotionTracking
Code for Deep Predictive Motion Tracking in Magnetic Resonance Imaging:Application to Fetal Imaging
=======
> Code for [Deep Predictive Motion Tracking in Magnetic Resonance Imaging: Application to Fetal Imaging]
(https://arxiv.org/abs/1909.11625) to model robust spatiotemporal motion in fetal MRI

## Getting started
---
1. The `main.py` script runs all training and evaluation OR just evaluation using a pretrained model.
2. Simply configure the `sample.json` in experiments directory and pass it as an argument to he main script
>> python main.py --config ./experiments/sample_experiment.json 
