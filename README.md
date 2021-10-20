Objective:
Real time complex video anomaly detection from surveillance videos.

Datasets:
1. UCSD dataset link: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
2. CHUK Aveneue dataset link: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
3. ShanghaiTech dataset link : https://svip-lab.github.io/dataset/campus_dataset.html

Pre-requisites:

Skills:
Some familiarity with concepts and frameworks of neural networks:
1. Framework: Keras and Tensorflow
2. Concepts: convolutional, Recurrent Neural Network and Generative Adversarial Networks.
3. Moderate skills in coding with Python and machine learning using Python.

Software Dependencies:
Various python modules. We recommend working with a conda environement
Python 3.7
PIL
glob
cv2
numpy
matplotlib 3.2.2
sklearn
CUDA Toolkit 10.1
cudnn 7.6.5
TensorFlow 2.2.0

Configurations:
1. Configurations in ano_train.py
     video_dir: Path to the folder containing video samples for training
     model_dir: Path to save the model for every 50 epochs
     image_dir: Path to save the generated image for every 50 epochs
     niter: Set the number of epochs for training
2. Configurations in ano_test.py
     load_model: load the trained model
     df.to_csv: Path to save anomaly score in .CSV format
3. Configurations in filereader.py
     rootpath: Path to the folder containing video samples for testing


Evaluation:
After obtaining the anomaly score for the test data, run evaluation.py to get the frame-level evaluation, in terms of AUC and EER for the chosen sample test video.
Note: The evaluation.py should be in the same folder, where the CSV files are stored.


Hardware Dependencies
A GPU machine with a minimum of 24 GB VRAM.

File organisation:
src folder: It contains the code for training, testing and evaluation.
Trained_sample_models folder: Contains some sample trained weights for validation
Generated_anomaly_score folder: Contains anomaly scores and ground truth file for all the test data used.
Results folder: Contains the obtained results like ROC, AUC and EER for all the test data. 
