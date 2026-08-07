import gradio as gr
from server import app as fastapi_app

# Mount FastAPI app onto Gradio for Hugging Face Space launcher
demo = gr.Blocks()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
