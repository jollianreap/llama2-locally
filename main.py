from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from config import model_path, GPU
from llama import LLaMA
from logger import logger


# FastAPI app
app: FastAPI = FastAPI()
print(GPU)
model: LLaMA = LLaMA(model_path, GPU)
logger.info('Model is loaded')


@app.get("/status")
async def process_url():
    """Returns the state of model."""
    logger.info('Status is requested')

    return {
        'status': 'active',
        'model': 'llama-2-7b-chat.Q2_K.gguf',
        'link': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main'

    }


@app.get('/inference')
async def inference_model(prompt: str) -> HTMLResponse:
    try:
        result = model.inference(prompt)
        logger.info(f'Model LLaMA is used via prompt: {prompt} | answer: {result}')
        return HTMLResponse(
            f'<br>User:</br> {prompt}'
            f'<br>Model:</br> {result}'
        )

    except Exception as e:
        logger.error(f'Something went wrong: {e}')
        return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)