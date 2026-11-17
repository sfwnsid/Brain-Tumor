🧠 Brain Tumor MRI Classification

A simple CNN built with Keras to classify brain MRI scans into four classes:

Glioma · Meningioma · Pituitary · No Tumor

Dataset

MRI images organised into Training/ (1,400 per class) and Testing/ (400 per class) folders. Images are resized to 150×150 grayscale.

Model
Layer	Details
Conv2D + MaxPool	32 filters, 3×3
Conv2D + MaxPool	64 filters, 3×3
Dense	128 → 64 (ReLU)
Dropout	0.5
Output	4-class softmax

Optimizer: Adam · Loss: sparse categorical crossentropy · 15 epochs, batch size 200

Results
Training accuracy: ~99%
Validation accuracy: ~87%

The gap between the two suggests overfitting. Data augmentation, more regularization, or a deeper/pretrained model (e.g. transfer learning) could help.

Usage
bash
pip install tensorflow scikit-image numpy pandas matplotlib
jupyter notebook brain_tumor.ipynb

Update the dataset path in the notebook to point to your local folder.

Tech Stack

Python · Keras/TensorFlow · scikit-image · NumPy · Matplotlib
