# COVID / Pneumonia / Normal Chest X-ray Classifier

A simple CNN (Convolutional Neural Network) that looks at a chest X-ray
image and predicts whether it shows **COVID**, **Pneumonia**, or is
**Normal**. Includes a Gradio web interface to try the model interactively.

> ⚠️ **Disclaimer:** This is a learning/portfolio project, not a medical
> diagnostic tool. Predictions should never be used for real medical decisions.

---

## Dataset

This project uses the **COVID-19 Radiography Dataset**, which contains
chest X-ray images organized into 4 folders: `COVID`, `Lung_Opacity`,
`Normal`, and `Viral Pneumonia` (each with an `images/` and `masks/`
subfolder).

This project only uses 3 of those classes:

| Dataset folder     | Used as   |
|---------------------|-----------|
| `COVID`             | COVID     |
| `Normal`             | Normal    |
| `Viral Pneumonia`   | Pneumonia |

`Lung_Opacity` and the `masks/` folders are not used.

---

## Project files

| File                          | What it does                                                |
|-------------------------------|---------------------------------------------------------------|
| `covid_pneumonia_cnn.ipynb`   | Full pipeline: prepares the data, splits it into train/test, builds and trains the CNN, evaluates it, and saves the trained model. |
| `app.py`                      | Gradio web app — upload an X-ray, get a prediction.          |
| `pneumonia_covid_model.keras` | The trained model (created after you run the notebook).       |
| `class_names.txt`             | The list of class names, in the order the model predicts them (created after you run the notebook). |

---

## How it works, step by step

1. **Collect the images** — the notebook copies only the actual X-ray
   images (not the mask images) from the 3 class folders into a clean,
   simple folder structure: `clean_dataset/COVID/`, `clean_dataset/Normal/`,
   `clean_dataset/Pneumonia/`.

2. **Split into train/test** — 80% of each class's images go into
   `dataset_split/train/`, 20% go into `dataset_split/test/`. This is a
   real folder split, done once and reused after that.

3. **Build the CNN** — a simple model: 3 convolution + pooling blocks,
   followed by a couple of dense layers, ending in a 3-way softmax output
   (COVID / Normal / Pneumonia).

4. **Train** — the model trains on the training folder for a set number of
   epochs, while being checked against the test folder after each epoch.

5. **Evaluate** — after training, the notebook reports test accuracy and
   plots accuracy/loss curves so you can see how training went.

6. **Save the model** — the trained model is saved to
   `pneumonia_covid_model.keras`, and the class names are saved to
   `class_names.txt`.

7. **Gradio interface** — `app.py` loads the saved model and lets you
   upload any chest X-ray image through a simple web page. It shows the
   predicted class along with confidence scores for all 3 classes.

---

## How to run it

### 1. Install the required packages

```bash
pip install tensorflow matplotlib gradio
```

(If `pip` isn't recognized on your system, try `py -3.11 -m pip install ...`
instead.)

### 2. Set your dataset path

Open `covid_pneumonia_cnn.ipynb` and check the `SOURCE_DIR` variable in the
"Set the paths" section — make sure it points to your
`COVID-19_Radiography_Dataset` folder.

### 3. Run the notebook

Run all cells from top to bottom. This will:
- Copy and organize the images (first run only — takes a few minutes)
- Split into train/test folders (first run only)
- Train the CNN
- Save `pneumonia_covid_model.keras` and `class_names.txt`

### 4. Launch the Gradio app

```bash
python app.py
```

Open the local link it prints in your browser (usually
`http://127.0.0.1:7860`), upload a chest X-ray image, and see the prediction.

---

## Notes / possible improvements

- The CNN here is intentionally simple (no transfer learning, no heavy
  augmentation) to keep the code easy to follow. If you want higher
  accuracy for a stronger portfolio piece, a pretrained model like
  MobileNetV2 (fine-tuned on this dataset) usually performs noticeably
  better than a from-scratch CNN.
- The dataset can be imbalanced across classes — if accuracy on one class
  looks much worse than the others, that's often why.
- Training time depends on your CPU/GPU. If it's slow, reducing `EPOCHS`
  or `IMG_SIZE` in the notebook will speed things up (at some cost to
  accuracy).
