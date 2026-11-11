import modal

from tests.train import *

image = (
    modal.Image.debian_slim()
    # Reads dependencies from local pyproject.toml using uv
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("cs336_basics", "tests")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("tests/fixtures", remote_path="/root/tests/fixtures")
    .add_local_file(
        local_path="data/TinyStoriesV2-GPT4-valid.bin", 
        remote_path="/root/data/TinyStoriesV2-GPT4-valid.bin"
    )
    .add_local_dir("data", remote_path="/root/data")
)

app = modal.App("llm-training", image=image)

tb_logs_volume = modal.Volume.from_name("tb_logs", create_if_missing=True)

@app.function(
    gpu="A100-80GB:1",          # Allocates an 8x A100 (80GB) node
    volumes={"/tb_logs": tb_logs_volume},
    timeout=60 * 60 * 1,        # Max run time (e.g., 1 hours)
)
def train():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")


    pipeline = TrainPipeline.create(data_base_dir='./data',
                                    run_id=2, 
                                    test=False, 
                                    device='cuda',
                                    tb_base_dir='/tb_logs')
    pipeline.before_training()
    pipeline.train(input_tokenized_file='./data/TinyStoriesV2-GPT4-valid.bin',
                   steps=1_000)
