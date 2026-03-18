from sklearn.ensemble import RandomForestClassifier
from utils.dataset import dataloader
from utils.preprocessor import preprocess, pca
import numpy as np
from utils.metrics import classification_report
from utils.visualization import plot_confusion_matrix, plot_ground_truth, plot_classification_map
from utils.logger import RunLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dataset = "ip"

    with RunLogger(dataset=dataset, model="rf") as logger:

        cube, gt = dataloader(dataset)
        cube_preprocessed = preprocess(cube)
        cube_pca = pca(cube_preprocessed.reshape(cube.shape[0], cube.shape[1], -1), n_components=30)

        H, W, B = cube_pca.shape
        X = cube_pca.reshape(-1, B)
        y = gt.flatten()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train Random Forest
        logger.log("Training Random Forest (n_estimators=100)")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        logger.log("Training complete")

        # Predict and evaluate on test set
        y_pred = rf_model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Full prediction map (all pixels) for visualisation
        y_pred_full = rf_model.predict(X)
        logger.save_array(y_pred_full.reshape(H, W), "prediction_map.npy")

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_confusion_matrix(y_test, y_pred,
                              class_names=[str(i) for i in np.unique(y)],
                              save_path=logger.get_path("confusion_matrix.png"))

        # Save classification map
        plot_classification_map(y_pred_full.reshape(H, W), gt, dataset=dataset,
                                save_path=logger.get_path("classification_map.png"))

        # Save ground truth
        plot_ground_truth(gt, dataset=dataset,
                          save_path=logger.get_path("ground_truth.png"))

        logger.log(f"OA={report['overall_accuracy']:.4f}  AA={report['average_accuracy']:.4f}  Kappa={report['kappa']:.4f}")