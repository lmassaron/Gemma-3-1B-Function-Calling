uv venv --python 3.12

uv pip install -U transformers
uv pip install accelerate
uv pip install -U datasets
uv pip install -U peft
uv pip install -i https://pypi.org/simple/ bitsandbytes
uv pip install -U trl

uv pip install -U --no-cache-dir --force-reinstall vllm
uv pip install -U pillow
uv pip install absl-py
uv pip install protobuf
uv pip install requests
uv pip install sentencepiece
uv pip install setuptools
uv pip install --no-deps --upgrade "flash-attn>=2.6.3"
uv pip install ipykernel
uv pip install ipywidgets
uv pip install jupyter
uv pip install tensorboard
uv pip install huggingface-hub
uv pip install numpy
uv pip install pandas