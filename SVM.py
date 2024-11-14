import numpy as np
from libsvm.svmutil import svm_train, svm_predict
from PCA import PCA  # Assume you have an existing PCA class
from preprocess import ImageProcessor  # Assume you have a preprocessing class to load and vectorize images


# Run PCA and return reduced datasets
def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced, X_test_reduced

# Function to train and evaluate SVM
def evaluate_svm(X_train, y_train, X_test, y_test, C_values):
    accuracies = {}
    for C in C_values:
        # Train SVM with linear kernel (-t 0) and given C
        model = svm_train(y_train, X_train, f'-s 0 -t 0 -c {C} -q')
        #  suppress the detailed output from svm_train and svm_predict, we can add the '-q' option (quiet mode) to the options string in svm_train
        
        # Predict on test data
        _, accuracy, _ = svm_predict(y_test, X_test, model,f'-q')
        
        # Record accuracy for this C
        accuracies[C] = accuracy[0]  # Store only the accuracy percentage
    return accuracies

# Main function
def main():
    # Load data
    cmu_pie_dir = "D:/NUS/5907/CA2/PIE"
    selfie_dir = "D:/NUS/5907/CA2/Raw_Selfie"
    processed_selfie_dir = "D:/NUS/5907/CA2/Selfie"
    
    subjects = range(40, 65)  # Choose 25 subjects as an example
    img_processor = ImageProcessor(cmu_pie_dir, selfie_dir, processed_selfie_dir)
    X_train, X_test, y_train, y_test = img_processor.get_dataset(subjects)
    
    # Define values for C
    C_values = [0.01, 0.1, 1]
    
    # Evaluate on raw data
    print("Evaluating SVM on raw data:")
    raw_accuracies = evaluate_svm(X_train, y_train, X_test, y_test, C_values)
    print("Raw Data Accuracies:", raw_accuracies)
    
    # Evaluate on PCA-reduced data with 80 components
    print("Evaluating SVM on PCA-reduced data (80 components):")
    X_train_80, X_test_80 = apply_pca(X_train, X_test, 80)
    pca_80_accuracies = evaluate_svm(X_train_80, y_train, X_test_80, y_test, C_values)
    print("PCA 80 Accuracies:", pca_80_accuracies)
    
    # Evaluate on PCA-reduced data with 200 components
    print("Evaluating SVM on PCA-reduced data (200 components):")
    X_train_200, X_test_200 = apply_pca(X_train, X_test, 200)
    pca_200_accuracies = evaluate_svm(X_train_200, y_train, X_test_200, y_test, C_values)
    print("PCA 200 Accuracies:", pca_200_accuracies)

if __name__ == "__main__":
    main()
