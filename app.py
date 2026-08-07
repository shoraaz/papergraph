import os
import uvicorn
import gradio as gr
from server import app as fastapi_app

# Mount FastAPI app onto Gradio so the full PaperGraph frontend & API serve on Hugging Face
demo = gr.mount_gradio_app(fastapi_app, gr.Blocks(), path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
