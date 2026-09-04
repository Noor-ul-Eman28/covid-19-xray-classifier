"""
Gradio interface for the COVID / Pneumonia / Normal chest X-ray classifier.

Run this AFTER running covid_pneumonia_cnn.ipynb, since it needs the two
files that notebook saves:
    pneumonia_covid_model.keras
    class_names.txt

Run with:  python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860)
"""

import tensorflow as tf
import gradio as gr

IMG_SIZE = (150, 150)   # must match IMG_SIZE used in training

# Load the trained model and the class names
model = tf.keras.models.load_model("pneumonia_covid_model.keras")

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]


def predict(image):
    if image is None:
        return None

    # Resize and normalize the uploaded image the same way we did in training
    img = tf.image.resize(image, IMG_SIZE)
    img = tf.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img, verbose=0)[0]

    # Return a dict of {class_name: confidence} — Gradio's Label shows this nicely
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a chest X-ray"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="COVID / Pneumonia / Normal Chest X-ray Classifier",
    description=(
        "Upload a chest X-ray image to see the model's prediction.\n\n"
        "⚠️ This is a student/portfolio project, not a medical diagnostic tool. "
        "Do not use it for real medical decisions."
    ),
)

if __name__ == "__main__":
    demo.launch()
