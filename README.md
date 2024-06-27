# LLaMA-2 Locally

This repository contains simple code to demonstrate the way LLaMA could be run on your computer. 
Here is the model I used: [LLama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

## Initialize project

```
python3 -v venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Start app
```
python3 main.py
```

## Configuration

Write your values to variables in .env:
- **model path**: the path to your LLaMA-2
- **gpu**: boolean variable (use or not to use)

Good Luck!
