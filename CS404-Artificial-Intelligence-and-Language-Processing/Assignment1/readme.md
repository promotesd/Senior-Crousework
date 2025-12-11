
# 环境配置
``` 
cd Assignment1

conda create -n assign1 python=3.10
conda activate assign1
pip install -r requirement.txt
```


# Part A

This part implements and trains CNN models (e.g. VGG and ResNet) on image classification datasets, and evaluates them with accuracy and per-class metrics.

---

## 1. Dataset preparation

For Part A **you do not need to manually download or unpack any dataset**.

* The datasets (e.g. **MNIST**, and optionally **CIFAR-10**) are created using `torchvision.datasets` inside the Jupyter notebook / scripts.
* When you run the code for the first time, `torchvision` will automatically download the data into the configured folder (for example `../dataset` relative to `partA`).

---

## 2. How to run the code (Jupyter notebook)

1. Activate the environment (if not already):

   ```bash
   conda activate assign1
   ```

2. Start Jupyter from the project root or from the `partA` folder, for example:

   ```bash
   cd Assignment1
   jupyter notebook
   ```

3. In the Jupyter interface, open the Part A notebook, e.g.:

   * `partA/partA.ipynb`
     (use the actual notebook filename you saved)

4. In the notebook:

   * **Model definition** is in `model.py` (e.g. `VGG.SimpleVGG`, `ResNet`).
   * **Training logic** is in `train.py` (`train_model()`).
   * **Evaluation & visualisation** is in `test.py` (`evaluate_model()`).

   The typical workflow is:

   1. Run the import and dataset preparation cells.
   2. Call `train_model()` for a chosen model (e.g. VGG / ResNet) and hyperparameters
      (learning rate, batch size, number of epochs).
   3. After training finishes, call `evaluate_model()` with:

      * `model` (how to construct the model),
      * `checkpoint_path` (best model `.pth` saved by training),
      * `test_dataset`,
        to compute accuracy, per-class F1-score, confusion matrix, and visualisations.

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


# Part B

目录结构为：
Assignment1/
├── dataset/CrowdHuman/
│   ├── images/train, images/val
│   ├── labels/train, labels/val
│   ├── annotation_train.odgt, annotation_val.odgt
└── partB/odgt2yolo.py

cd \Assignment1\partB
python odgt2yolo.py ../dataset/CrowdHuman/annotation_train.odgt
python odgt2yolo.py --val ../dataset/CrowdHuman/annotation_val.odgt