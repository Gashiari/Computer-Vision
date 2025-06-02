# Gradio App with ViT and Zero-Shot CLIP for Utility Pole Classification

import gradio as gr
from transformers import pipeline

# Hugging Face model repositories
vit_pipeline = pipeline("image-classification", model="your-username/vit-utility-poles")
clip_pipeline = pipeline(model="openai/clip-vit-base-patch32", task="zero-shot-image-classification")

# List of countries (all 96 classes from the training dataset)
labels = [
    "Albania", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium", "Bhutan", "Bolivia",
    "Bosnia", "Botswana", "Brazil", "Bulgaria", "Cambodia", "Canada", "Chile", "China", "Christmas", "Colombia",
    "Costa", "Curaçao", "Cyprus", "Czech", "Dominican", "Ecuador", "Estonia", "Eswatini", "Faroe", "France",
    "Germany", "Ghana", "Greece", "Greenland", "Guam", "Guatemala", "Hong", "Hungary", "Iceland", "India",
    "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lesotho", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malaysia", "Mali",
    "Malta", "Mexico", "Montenegro", "Netherlands", "New", "Nigeria", "North", "Norway", "Oman", "Panama", "Peru",
    "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto", "Romania", "Russia", "Rwanda", "San", "Senegal",
    "Serbia", "Singapore", "Slovakia", "Slovenia", "South", "Spain", "Sri", "Sweden", "Switzerland", "São", "Taiwan",
    "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United", "Uruguay"
]

# Classification function
def classify_pole(image):
    vit_results = vit_pipeline(image)
    vit_output = {res['label']: round(res['score'], 3) for res in vit_results[:5]}

    clip_results = clip_pipeline(image, candidate_labels=labels)
    clip_output = {res['label']: round(res['score'], 3) for res in clip_results[:5]}

    return {
        "ViT Fine-Tuned": vit_output,
        "CLIP Zero-Shot": clip_output
    }

# Example images (to be manually added)
example_images = [
    ["examples/Albania.jpg"],
    ["examples/Brazil.jpg"],
    ["examples/Hungary.jpg"]
]

# Build interface
iface = gr.Interface(
    fn=classify_pole,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Utility Pole Classification",
    description="Compare predictions between a fine-tuned ViT model and a zero-shot CLIP model.",
    examples=example_images
)

# Launch app
if __name__ == "__main__":
    iface.launch()