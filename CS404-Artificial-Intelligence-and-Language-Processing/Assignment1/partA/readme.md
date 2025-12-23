# Part A

This part implements and trains CNN models (e.g. VGG and ResNet) on image classification datasets, and evaluates them with accuracy and per-class metrics.

---

## 1. Dataset preparation

For Part A **you do not need to manually download or unpack any dataset**.

* The datasets **MNIST** and **CIFAR-10** are created using `torchvision.datasets` inside the Jupyter notebook / scripts.
* When you run the code for the first time, `torchvision` will automatically download the data into the configured folder  `../dataset` relative to `partA`.

---

## 2. start environment

1. Activate the environment (if not already):

   ```bash
   conda activate assign1
   ```

2. Start Jupyter from the project root or from the `partA` folder using vscode:

   ```bash
   cd Assignment1/partA
   ```

3. In the PartA:

   * **Model definition** is in `model.py` (`VGG.SimpleVGG`, `ResNet`).
   * **Training logic** is in `train.py` (`train_model()`).
   * **Evaluation & visualisation** is in `test.py` (`evaluate_model()`).

   The typical workflow is:

   1. Run the import and dataset preparation cells.
   2. Call `train_model()` for a chosen model (VGG / ResNet) and hyperparameters
      (learning rate, batch size, number of epochs).
   3. After training finishes, call `evaluate_model()` with:

      * `model`,
      * `checkpoint_path` (best model `.pth` path saved by training),
      * `test_dataset`, to compute accuracy, per-class F1-score, confusion matrix, and visualisations.

---

## 4. Outputs and logs

* During training, logs and plots are saved under:

  ```text
  ./logs/<experiment_name>/
  ```

  including:

  * training / test loss curves,
  * test accuracy vs. epoch,
  * the best model checkpoint: `model_best.pth`.

* During evaluation, additional files are saved under:

  ```text
  ./logs/<experiment_name>/test_results/
  ```

  including:

  * `test_metrics.txt` – overall accuracy, macro F1, per-class metrics (from `sklearn`),
  * `confusion_matrix.png` – confusion matrix visualisation,
  * `correct_examples.png` – examples of correctly classified images,
  * `incorrect_examples.png` – examples of misclassified images.

---

## 5. How to run code
### CIFAR10_train
Run code blocks in CIFAR10_train.ipynb one by one

### MNIST_train
Run code blocks in MNIST_train.ipynb one by one


---

## 6. Hyperparameter comparison
### CIFAR10_train

### MNIST_train
