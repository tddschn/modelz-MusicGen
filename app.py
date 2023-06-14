import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda"  # change this to "cpu" if you do not have a GPU

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def predict(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0])

demo = gr.Interface(fn=predict, inputs="text", outputs="text")

demo.launch(server_port=os.environ.get("GRADIO_SERVER_PORT", 8080))
