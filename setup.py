import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='deep-predictive-motion-tracking',
    version='1.0',
    packages=['data', 'utils', 'models'],
    url='https://github.com/singhay/DeepPredictiveMotionTracking',
    license='MIT',
    author='Ayush Singh',
    author_email='ayush.singh@childrens.harvard.edu',
    description='Code for paper: Deep Predictive Motion Tracking in Magnetic Resonance Imaging: Application to Fetal '
                'Imaging (https://arxiv.org/abs/1909.11625) ',
    long_description=read('README'),
    install_requires=['cv2', 'SimpleITK==1.2.2', 'tqdm==4.23.1',
                      'numpy==1.21.0', 'keras==2.2.0', 'scipy==1.3.0',
                      'nibabel==2.5.1', 'nilearn==0.5.1']
)
