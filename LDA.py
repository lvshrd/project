import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from preprocess import ImageProcessor
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LDA():
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.linear_discriminants = None
    
    def fit(self, X, y):
        n_features = X.shape[1]
        N_samples = X.shape[0]
        class_labels = np.unique(y)
        # Compute the overall mean
        mean_overall = np.mean(X, axis=0)
        
        # Initialize the within-class and between-class scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        # Calculate S_W and S_B for each class
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            n_c = X_c.shape[0]
            # Within-class scatter matrix
            S_W += (n_c / N_samples) * (X_c - mean_c).T @ (X_c - mean_c)
            # Between-class scatter matrix         
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B +=  (n_c / N_samples) * (mean_diff @ mean_diff.T)

        # Solve the eigenvalue problem for S_W^-1 S_B
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[sorted_indices], eigvecs[:, sorted_indices]
        
        # Select the top 'n_components' eigenvectors
        self.linear_discriminants = eigvecs[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants)
    

def apply_lda(X_train, y_train, X_test, n_components):
    lda = LDA(n_components=n_components)
    lda.fit(X_train, y_train)  # Fit LDA on the training data
    X_train_lda = np.real(lda.transform(X_train))  # Transform the training data
    X_test_lda = np.real(lda.transform(X_test))    # Transform the test data
    
    return X_train_lda, X_test_lda

def visualize_lda(X_lda, y, highlight_indices=None, plot_3d=False):
    if X_lda.shape[1] < 2:
        raise ValueError("Number of components should be at least 2 for visualization.")
    
    if plot_3d and X_lda.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], c=y)
        if highlight_indices is not None:
            ax.scatter(X_lda[highlight_indices, 0], X_lda[highlight_indices, 1], X_lda[highlight_indices, 2],
                       color='red', s=100, label="Selfie")
        plt.legend()
        plt.show()
    else:
        plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
        if highlight_indices is not None:
            plt.scatter(X_lda[highlight_indices, 0], X_lda[highlight_indices, 1], color='red', s=50, label="Selfie")
        plt.legend()
        plt.show()

def classify_and_report(X_train, y_train, X_test_pie, y_test_pie, X_test_selfie, y_test_selfie):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    # Predict and calculate accuracy for CMU PIE test images
    y_pred_pie = knn.predict(X_test_pie)
    accuracy_pie = np.mean(y_pred_pie == y_test_pie)
    
    # Predict and calculate accuracy for selfie test images
    y_pred_selfie = knn.predict(X_test_selfie)
    accuracy_selfie = np.mean(y_pred_selfie == y_test_selfie)
    
    return accuracy_pie, accuracy_selfie

def main():
    # Set paths
    cmu_pie_dir = "D:/NUS/5907/CA2/PIE"
    selfie_dir = "D:/NUS/5907/CA2/Raw_Selfie"
    processed_selfie_dir = "D:/NUS/5907/CA2/Selfie"
    
    # Initialize image processor
    subjects = range(40, 65)  # Select 25 subjects
    img_processor = ImageProcessor(cmu_pie_dir, selfie_dir, processed_selfie_dir)
    
    # Load training and test data, with labels
    X_train, X_test, y_train, y_test = img_processor.get_dataset(subjects)
    
    # Separate CMU PIE data and selfie data from combined dataset
    y_test_pie, y_test_selfie = y_test[:-3], y_test[-3:]

    for n_components in [2, 3, 9]:
        # Apply LDA
        X_train_lda, X_test_lda = apply_lda(X_train, y_train, X_test, n_components)
        
        # Separate test data for CMU PIE and selfies
        X_test_pie_lda, X_test_selfie_lda = X_test_lda[:-3], X_test_lda[-3:]
        
        # Visualize data distribution for 2D and 3D
        if n_components in [2, 3]:
            highlight_indices = list(range(X_train.shape[0] - 7, X_train.shape[0]))  # Last 7 images are selfies
            visualize_lda(X_train_lda, y_train, highlight_indices, plot_3d=(n_components == 3))
        
        # Classify and report accuracy
        accuracy_pie, accuracy_selfie = classify_and_report(X_train_lda, y_train, X_test_pie_lda, y_test_pie, X_test_selfie_lda, y_test_selfie)
        print(f"Accuracy on CMU PIE test images with {n_components} LDA components: {accuracy_pie * 100:.2f}%")
        print(f"Accuracy on Selfie test images with {n_components} LDA components: {accuracy_selfie * 100:.2f}%")

if __name__ == "__main__":
    main()