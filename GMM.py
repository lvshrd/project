import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from preprocess import ImageProcessor
from PCA import PCA


def fit_gmm(X, n_components=3):
    """Fit a GMM model to the data."""
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    return gmm

def visualize_clusters(X, labels, title="GMM Clustering", xlabel="x1", ylabel="x2"):
    """Visualize clustering results with scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Component {label}")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

def visualize_cluster_images(X_original, labels, image_shape=(32, 32), images_per_cluster=10):
    """Visualize images that belong to each GMM component."""
    fig = plt.figure(figsize=(9, 3))
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        cluster_images = X_original[labels == label]
        
        for j in range(min(images_per_cluster, len(cluster_images))):
            ax = fig.add_subplot(len(unique_labels), images_per_cluster, i * images_per_cluster + j + 1)
            ax.imshow(cluster_images[j].reshape(image_shape), cmap="gray")
            ax.axis("off")
    
    plt.suptitle("Images grouped by GMM component")
    plt.show()

def main():
    # Set paths
    cmu_pie_dir = "D:/NUS/5907/CA2/PIE"
    selfie_dir = "D:/NUS/5907/CA2/Raw_Selfie"
    processed_selfie_dir = "D:/NUS/5907/CA2/Selfie"
    
    # Initialize image processor
    subjects = range(40, 65)  # Select 25 subjects
    img_processor = ImageProcessor(cmu_pie_dir, selfie_dir, processed_selfie_dir)
    X_train, _, _, _ = img_processor.get_dataset(subjects)

    # Train and visualize GMM on raw data
    # gmm = GMM(n_components=3)
    # gmm.fit(X_train)
    # labels = gmm.predict(X_train)
    gmm = fit_gmm(X_train, n_components=3)
    labels = gmm.predict(X_train)
    visualize_clusters(X_train, labels, title="GMM Clustering on Raw Data")
    visualize_cluster_images(X_train, labels )
    
    # Apply PCA and train GMM on reduced data
    for n_components in [200, 80]:
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_reduced = pca.transform(X_train)

        # gmm = GMM(n_components=3)
        # gmm.fit(X_train_reduced)
        # labels = gmm.predict(X_train_reduced)
        gmm = fit_gmm(X_train_reduced, n_components=3)
        labels = gmm.predict(X_train_reduced)
        visualize_clusters(X_train_reduced, labels, title=f"GMM Clustering with {n_components} PCA Components")
        visualize_cluster_images(X_train, labels )  
        # Use original images for visualization

if __name__ == "__main__":
    main()
