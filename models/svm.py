from sklearn.svm import SVC
from utils.dataset import dataloader
from utils.preprocessor import preprocess, pca
import numpy as np
from utils.metrics import classification_report
from utils.visualization import plot_confusion_matrix, plot_ground_truth, plot_classification_map
from utils.logger import RunLogger
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset = "ip"

    with RunLogger(dataset=dataset, model="svm") as logger:

        cube, gt = dataloader(dataset)
        cube_preprocessed = preprocess(cube)
        cube_pca = pca(cube_preprocessed.reshape(cube.shape[0], cube.shape[1], -1), n_components=30)

        H, W, B = cube_pca.shape
        X = cube_pca.reshape(-1, B)
        y = gt.flatten()

        # Train SVM
        logger.log("Training SVM (kernel=rbf, C=1)")
        svm_model = SVC(kernel='rbf', C=1, gamma='scale')
        svm_model.fit(X, y)
        logger.log("Training complete")

        # Predict and evaluate
        y_pred = svm_model.predict(X)
        report = classification_report(y, y_pred)
        logger.save_array(y_pred.reshape(H, W), "prediction_map.npy")

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_confusion_matrix(y, y_pred,
                              class_names=[str(i) for i in np.unique(y)],
                              save_path=logger.get_path("confusion_matrix.png"))

        # Save classification map
        plot_classification_map(y_pred.reshape(H, W), gt, dataset=dataset,
                                save_path=logger.get_path("classification_map.png"))

        # Save ground truth
        plot_ground_truth(gt, dataset=dataset,
                          save_path=logger.get_path("ground_truth.png"))

        logger.log(f"OA={report['overall_accuracy']:.4f}  AA={report['average_accuracy']:.4f}  Kappa={report['kappa']:.4f}")
