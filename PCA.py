import numpy as np
import matplotlib.pyplot as plt
from preprocess import ImageProcessor
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

class PCA:
    # PCA的计算涉及特征值分解或奇异值分解（SVD），这些方法有时会因为数值精度带来小量的复数部分

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, X):
        # Step 1: Standardize the data (subtract the mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: Sort the eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance


    def transform(self, X):
        # Step 6: Project the data onto the selected components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def visualize_projection(self, X, labels, highlight_indices=None, plot_3d=False):
        if self.n_components < 2:
            raise ValueError("Number of components should be at least 2 for visualization.")
        

        # X_projected = self.transform(X)  出现comlpex number，leads to no scatter nodes in 3d projection visualization
        X_projected = np.real(self.transform(X))
        
        
        if plot_3d and self.n_components >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], c=labels)
            if highlight_indices is not None:
                ax.scatter(X_projected[highlight_indices, 0], X_projected[highlight_indices, 1], X_projected[highlight_indices, 2], color='red', s=100, label="Selfie")
            plt.legend()
            plt.show()
        else:
            plt.scatter(X_projected[:, 0], X_projected[:, 1], c=labels)
            if highlight_indices is not None:
                plt.scatter(X_projected[highlight_indices, 0], X_projected[highlight_indices, 1], color='red', s=50, label="Selfie")
            plt.legend()
            plt.show()

    def visualize_eigenfaces(self, image_shape=(32, 32)):
        fig, axes = plt.subplots(1, self.n_components, figsize=(self.n_components*self.n_components, self.n_components))
        for i, component in enumerate(self.components.T):
            # eigenface = component.reshape(image_shape)
            eigenface = np.real(component.reshape(image_shape))
            ax = axes[i]
            ax.imshow(eigenface, cmap='gray')
            ax.axis('on')
        plt.show()
    

    def classify_nearest_neighbor(self, X_train, y_train, X_test, y_test):
        # Calculate distances between test and training samples in the reduced space
        distances = cdist(X_test, X_train, metric='euclidean')
        nearest_neighbors = distances.argmin(axis=1)
        y_pred = y_train[nearest_neighbors]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

class NearestNeighborClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # 训练阶段只是存储训练集数据
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # 计算每个测试样本与所有训练样本的欧氏距离
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # 找到最近的训练样本的索引
            nearest_neighbor_index = np.argmin(distances)
            # 使用最近邻的标签作为预测
            predictions.append(self.y_train[nearest_neighbor_index])
        return np.array(predictions)
    

def main_test(cmu_pie_dir,selfie_dir,processed_selfie_dir):

    # Initialize image processor and PCA objects
    subjects = range(40, 65)  # Choose 25 subjects
    img_processor = ImageProcessor(cmu_pie_dir, selfie_dir,processed_selfie_dir)
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)

    # Load and process data
    X_combined_train,_ , _ , _ = img_processor.get_dataset(subjects)
    # X_train = X_combined_train[:(X_combined_train.shape[0]-7),:]
    X_combined_train_excluded = X_combined_train[:-7]
    X_sampled_500 = X_combined_train[np.random.choice(X_combined_train_excluded.shape[0], 500, replace=False)]
    X_sampled = np.vstack((X_sampled_500, X_combined_train[-7:]))

    # Apply PCA
    pca_2d.fit(X_sampled)
    X_2d = pca_2d.transform(X_sampled)
    pca_3d.fit(X_sampled)
    X_3d = pca_3d.transform(X_sampled)
    
    # Visualize projections
    # highlight_indices = list(range(len(X_train) - 7, len(X_train)))  # Last 7 images are selfies
    highlight_indices = list(range(X_sampled.shape[0]-7, X_sampled.shape[0]))  # Last 7 images are selfies
    pca_2d.visualize_projection(X_sampled, labels=None, highlight_indices=highlight_indices)
    pca_3d.visualize_projection(X_sampled, labels=None, highlight_indices=highlight_indices, plot_3d=True)
    # Show eigenfaces
    pca_3d.visualize_eigenfaces()

def main(cmu_pie_dir,selfie_dir,processed_selfie_dir):
    # Initialize image processor and PCA objects
    subjects = range(21, 46)  # Choose 25 subjects
    img_processor = ImageProcessor(cmu_pie_dir, selfie_dir,processed_selfie_dir)
    # Load training and test data
    X_train, X_test, y_train, y_test = img_processor.get_dataset(subjects)
    # Separate CMU PIE data and selfie data from combined dataset
    # Assuming last 7 samples in train and test sets are selfies based on prior loading pattern

    X_test_pie, X_test_selfie = X_test[:-3], X_test[-3:]
    y_test_pie, y_test_selfie = y_test[:-3], y_test[-3:]

    for n_components in [40, 80, 200]:
        pca = PCA(n_components=n_components)
        
        # Apply PCA to reduce dimensions
        pca.fit(X_train)
        X_train_reduced = pca.transform(X_train)
        X_test_pie_reduced = pca.transform(X_test_pie)
        X_test_selfie_reduced = pca.transform(X_test_selfie)
        
        # Nearest neighbor classification
        nn_classifier = NearestNeighborClassifier()
        ##A scikit-learn.KNeighborsClassifier with n_neighbors=1 is used to classify the test data which can improve the efficiency because of Vectorized computations and Efficient Distance Calculations
        # knn = KNeighborsClassifier(n_neighbors=1)
        # knn.fit(X_train_reduced, y_train)
        nn_classifier.fit(X_train_reduced, y_train)
        
        # Predict and calculate accuracy for CMU PIE test images
        y_pred_pie = nn_classifier.predict(X_test_pie_reduced)
        accuracy_pie = np.mean(y_pred_pie == y_test_pie)
        # Predict and calculate accuracy for selfie test images
        y_pred_selfie = nn_classifier.predict(X_test_selfie_reduced)
        accuracy_selfie = np.mean(y_pred_selfie == y_test_selfie)
        
        print(f"Accuracy on CMU PIE test images with {n_components} components: {accuracy_pie * 100:.2f}%")
        print(f"Accuracy on Selfie test images with {n_components} components: {accuracy_selfie * 100:.2f}%")


if __name__ == "__main__":
    # Set file paths for the datasets
    cmu_pie_dir = "D:/NUS/5907/CA2/PIE"
    selfie_dir = "D:/NUS/5907/CA2/Raw_Selfie"
    processed_selfie_dir = "D:/NUS/5907/CA2/Selfie"

    main_test(cmu_pie_dir, selfie_dir, processed_selfie_dir)
    main(cmu_pie_dir,selfie_dir,processed_selfie_dir)