import numpy as np
from sklearn.decomposition import PCA

def preprocess(cube):
    """
    Preprocess the HSI cube.
    Steps:
    1. Normalize the cube (per-band standardization).
    2. Flatten the cube.
    3. Return the preprocessed cube.
    """

    H, W, B = cube.shape
    cube_reshaped = cube.reshape(-1, B)  
    

    mean = np.mean(cube_reshaped, axis=0)
    std = np.std(cube_reshaped, axis=0) + 1e-8  
    cube_normalized = (cube_reshaped - mean) / std
        
    return cube_normalized


def pca(cube, n_components=30):
    """
    Apply PCA to reduce the spectral dimensionality of the HSI cube.
    Steps:
    1. Reshape the cube to (H*W, B).
    2. Compute the covariance matrix and eigen decomposition.
    3. Project the data onto the top n_components eigenvectors.
    4. Reshape back to (H, W, n_components).
    """
    H, W, B = cube.shape
    cube_reshaped = cube.reshape(-1, B)  
    mean = np.mean(cube_reshaped, axis=0)
    cube_centered = cube_reshaped - mean
    cov_matrix = np.cov(cube_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors_reduced = eigenvectors[:, :n_components]
    cube_pca = np.dot(cube_centered, eigenvectors_reduced)
    
    return cube_pca.reshape(H, W, n_components)

def pcaV1(cube, n_components=30):
    """
    Apply PCA using sklearn's PCA implementation.
    Steps:
    1. Reshape the cube to (H*W, B).
    2. Fit PCA and transform the data.
    3. Reshape back to (H, W, n_components).
    """
    H, W, B = cube.shape
    cube_reshaped = cube.reshape(-1, B)  
    pca_model = PCA(n_components=n_components)
    cube_pca = pca_model.fit_transform(cube_reshaped)
    
    return cube_pca.reshape(H, W, n_components)