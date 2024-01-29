import PIL
import torch
import gradio as gr
from process import load_seg_model, get_palette, generate_mask
from gradio.inputs import Image as GrImage

device = 'cpu'

def initialize_and_load_models():
    checkpoint_path = 'model/cloth_segm.pth'
    net = load_seg_model(checkpoint_path, device=device)    
    return net

net = initialize_and_load_models()
palette = get_palette(4)

def run(input_image, styled_layer):
    cloth_seg = generate_mask(input_image, styled_layer, net=net, palette=palette, device=device)
    return cloth_seg

# Define input and output interfaces
input_image = gr.inputs.Image(label="Input Image", type="pil")
input_styled_layer = gr.inputs.Image(label="Styled Layer", type="pil")  # Add styled layer input

# Define the Gradio interface
cloth_seg_image = gr.outputs.Image(label="Cloth Segmentation", type="pil")

title = "Demo for Cloth Segmentation"
description = "An app for Cloth Segmentation"
inputs = [input_image, input_styled_layer]  # Include the styled layer input
outputs = [cloth_seg_image]

gr.Interface(fn=run, inputs=inputs, outputs=outputs, title=title, description=description).launch(share=True)
