# Multivariate (Faces) Classification with PCA LDA GMM SVM and CNN

## Introduction
This is the AY2024/2025 NUS **EE5907 Pattern Recognition** CA2 project. The project explores the deployment of several machine learning techniques, including Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Convolutional Neural Networks (CNN), and Gaussian Mixture Models (GMM) for dimensionality reduction, classification, and clustering tasks.

## Instruction
In this project, we have implemented an `ImageProcessor` class in `preprocessing.py` to handle images from the CMU PIE dataset and custom selfies. The class performs essential preprocessing tasks, including converting images to grayscale (using `IMREAD_GRAYSCALE`), resizing them to a standard 32×32 resolution, and splitting them into training and test datasets.

Each of the five tasks in the project corresponds to a separate method, and we have created five independent scripts to conduct the experiments. While PCA and LDA are custom-built implementations, the other methods are realized using existing Python packages.

To use this project, place 10 of your own photos (preferably cropped to focus on your face, minimizing unnecessary background) in the `Selfie` folder. Additionally, unzip the `PIE.zip` file into the `PIE` folder located at the root directory.

### Task 1: PCA for Dimensionality Reduction
In the first task, 500 random samples are selected from the CMU PIE training data and combined with user-provided selfies training data to perform PCA. The dimensionality of the vectorized images is reduced to 2 and 3 dimensions, respectively, and the resulting 2D and 3D projections are visualized with the selfies highlighted. Additionally, the 3 primary components -- eigenfaces are visualized.

To run this task, uncomment the following line in `PCA.py`:

```python
 main_test(cmu_pie_dir, selfie_dir, processed_selfie_dir)
```

After running the script, the following results will be displayed:
<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="margin: 5px;">
        <img src="Result\1-1 PCA Reduce to 2D.png" alt="1-1" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <img src="Result\1-2 PCA Reduce to 3D.png" alt="1-2" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">PCA 2D/3D Projection</p>    
    </div>
    <div style="margin: 5px;">
        <img src="Result\1-3 PCA 3 Eigenfaces.png" alt="1-3" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">3 Eigenfaces in PCA_3D</p> 
    </div>
</div>

After dimensionality reduction, PCA is applied to reduce the face images to 40, 80, and 200 components, followed by classification of test images using the nearest neighbor rule. Here are the results:

    Accuracy on CMU PIE test images with 40 components: 93.76%
    Accuracy on Selfie test images with 40 components: 100.00%
    Accuracy on CMU PIE test images with 80 components: 94.53%
    Accuracy on Selfie test images with 80 components: 66.67%
    Accuracy on CMU PIE test images with 200 components: 95.15%
    Accuracy on Selfie test images with 200 components: 66.67%

### Task 2: LDA for Dimensionality Reduction
Run the `LDA.py` script to visualize the distribution of the sampled data (similar to PCA) after reducing dimensionality to 2 and 3 components.

<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="margin: 5px;">
        <img src="Result\2-1 LDA Reduce to 2D.png" alt="2-1" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <img src="Result\2-2 LDA Reduce to 3D.png" alt="2-2" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">LDA 2D/3D Projection</p>    
    </div>
</div>

The classification accuracy for LDA is as follows:

    Accuracy on CMU PIE test images with 2 LDA components: 18.46%
    Accuracy on Selfie test images with 2 LDA components: 0.00%
    Accuracy on CMU PIE test images with 3 LDA components: 37.85%
    Accuracy on Selfie test images with 3 LDA components: 0.00%
    Accuracy on CMU PIE test images with 9 LDA components: 92.38%
    Accuracy on Selfie test images with 9 LDA components: 0.00%

### Task 3: GMM for Clustering
In this task, GMM models with 3 Gaussian components are trained using both raw face images and PCA-preprocessed images (reduced to 80 and 200 components). The clustering results are visualized along with the grouped face images assumed to belong to the same class.

To run the task, execute the `GMM.py` script. The following results will be generated:

<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <div style="margin: 5px;">
        <img src="Result\3-1 GMM Clustering on Raw Data.png" alt="3-1" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <img src="Result\3-2 Face Images Grouped from Raw Data.png" alt="3-2" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">Raw Data</p>
    </div>
    <div style="margin: 5px;">
        <img src="Result\3-3 GMM Clustering with 200 PCA Components.png" alt="3-3" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <img src="Result\3-4 Face Images Grouped from 200 PCA Components.png" alt="3-4" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">PCA 200</p>
    </div>
    <div style="margin: 5px;">
        <img src="Result\3-5 GMM Clustering with 80 PCA Components.png" alt="3-5" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <img src="Result\3-6 Face Images Grouped from 80 PCA Components.png" alt="3-6" style="width: 100%; max-width: 500px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2);"/>
        <p style="text-align: center; font-style: italic;">PCA 80</p>
    </div>
</div>

### Task 4: SVM for Linear Classification
Before running the SVM, make sure to install the required library:

```bash
pip install -U libsvm-official
```

The SVM classifier is implemented using the `svm_train` and `svm_predict` functions. Here are some optional parameters for these functions:

- `-s 0`: C-Support Vector Classification (C-SVC) for classification problems.
- `-t 0`: Linear kernel (for linear decision boundary).
- `-c {C}`: Penalty parameter for the SVM.
- `-q`: Quiet mode (suppresses detailed training information).

Evaluation of SVM on different datasets:

    Evaluating SVM on raw data:
    Raw Data Accuracies: {0.01: 99.30928626247122, 0.1: 99.30928626247122, 1: 99.30928626247122}
    Evaluating SVM on PCA-reduced data (80 components):
    PCA 80 Accuracies: {0.01: 99.0023023791251, 0.1: 99.0023023791251, 1: 99.0023023791251}
    Evaluating SVM on PCA-reduced data (200 components):
    PCA 200 Accuracies: {0.01: 99.23254029163469, 0.1: 99.23254029163469, 1: 99.23254029163469}

### Task 5: CNN for Classification (Optional)
This task involves implementing a Convolutional Neural Network (CNN) for classification.

 The architecture consists of:
- Two convolutional layers with a kernel size of 5.
- Max pooling with a kernel size of 2 and stride 2.
- A fully connected layer with ReLU activation and 26 output nodes.

This model is trained with the face images for 26-category classification. A basic implementation using PyTorch is provided in `CNN.py`.

---

### Conclusion
This project explores various machine learning techniques applied to face recognition and classification tasks. Each task is executed step-by-step, utilizing PCA, LDA, SVM, and GMM, with results visualized for a better understanding of the classifiers' behavior and accuracy.
