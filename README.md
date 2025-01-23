# AI-Real-_Image-Classifier
---

AI Real vs AI-Generated Image Classifier

This project is a deep learning-based image classification system that distinguishes between real and AI-generated images. It uses Convolutional Neural Networks (CNNs) to analyze and classify images, and is implemented with TensorFlow and Keras.


---

Features

Replaces spaces in file names with underscores for consistency.

Resizes images to a uniform size (48x48) and normalizes pixel values.

Trains a CNN model to classify images as "Real" or "AI-Generated."

Includes performance evaluation metrics like accuracy and classification reports.

Saves and loads the trained model for future predictions.



---

Technologies Used

Python

TensorFlow/Keras

OpenCV

Scikit-learn

Matplotlib

NumPy



---

Setup Instructions

1. Clone the Repository

git clone https://github.com/Harshita-Rupani29/AI-Real-_Image-Classifier.git
cd AI-Real-Image-Classifier

2. Download the Dataset

The training and testing datasets are provided as a ZIP file.

1. Download the dataset_train.zip and dataset_test.zip files from the "Releases" section of this repository or the provided link.


2. Extract the ZIP files in the root folder of this repository.
The directory structure after extraction should look like this:



AI-Real-Image-Classifier/
├── dataset_train/
│   ├── AIGenerated/
│   ├── Real/
├── dataset_test/
│   ├── AIGenerated/
│   ├── Real/

3. Install Dependencies

Install the required Python libraries:

pip install -r requirements.txt


---

Model Training

1. Preprocessing

Replaces spaces in filenames with underscores.

Resizes images to 48x48 and normalizes pixel values to [0, 1].


2. Training the CNN

Run the training script to train the model:

python train_model.py

3. Saving the Model

The trained model is saved as AIGeneratedModel.h5.


---

Model Evaluation

Evaluate the model's performance on the testing dataset:

python evaluate_model.py


---

Making Predictions

To classify an individual image as "Real" or "AI-Generated":

1. Place the image in the desired folder.


2. Run the prediction script:



python predict_image.py --image <path_to_image>


---

Results and Visualization

Training vs. Validation Loss and Accuracy plots are generated.

Detailed classification reports for test data are displayed.



---

