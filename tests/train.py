
from tests.adapters import *
from tests.config import *
from tests.model import *
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

import numpy as np
import time
from pathlib import Path
import logging
import json
import dataclasses


class TrainPipeline:
    def __init__(self, 
                 config: TrainingConfig, 
                 model: MyLLM, 
                 optimizer: AdamWOptimizer,
                 run_dir: str, 
                 run_id: int,
                 device='cpu',
                 tb_dir=None):
        self.config = config
        assert self.config.model_config.vocab_size is not None
        self.model = model
        self.optimizer = optimizer

        self.run_dir = run_dir
        self.device = device
        self.run_id = run_id

        self.tb_dir = tb_dir

    @classmethod
    def create(cls, data_base_dir, run_id=None, test=False, device='cpu', tb_base_dir=None):
        tokenizer = get_tokenizer_from_vocab_merges_path(
            vocab_path=VOCAB_PATH,
            merges_path=MERGES_PATH,
            special_tokens=["<|endoftext|>"]
        )

        if not run_id:
            folder = Path(data_base_dir)
            # List of Path objects for all subdirectories
            run_ids = [int(p) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
            run_ids.append(0)
            run_id = max(run_ids) + 1
        run_dir = f"{data_base_dir}/run_{run_id}"

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        config_path = f"{run_dir}/config.json"
        if Path(config_path).is_file():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = TrainingConfig(
                model_config=TransformerConfig(**config_dict['model_config']),
                optimizer_config=OptimizerConfig(**config_dict.get('optimizer_config', {})),
                pipeline_config=PipelineConfig(**config_dict.get('pipeline_config', {}))
            )
        else:
            config = TrainingConfig(model_config=TransformerConfig(vocab_size=tokenizer.get_voc_size()))
            with open(config_path, "w") as f:
                json.dump(dataclasses.asdict(config), f, indent=2)
        
        if test:
            config.model_config.num_layers = 2
            config.pipeline_config.loss_log_freq = 1
        print(f"config: {config}")

        model = MyLLM(config.model_config).to(device)
        optimizer = AdamWOptimizer(model.parameters(),
                                   config.optimizer_config.betas, 
                                   config.optimizer_config.weight_decay,
                                    lr=config.optimizer_config.lr,
                                    eps=config.optimizer_config.eps)
    
        # trace
        if tb_base_dir:
            tb_dir = f"{tb_base_dir}/{run_dir}"
            os.makedirs(tb_dir, exist_ok=True)
        else:
            tb_dir = None

        return cls(config, model, optimizer, run_dir, run_id, device, tb_dir=tb_dir)
        
    def before_training(self):
        ## TODO; load weights
        self.model.init_params()

    def train(self, input_tokenized_file, steps=1_000):
        dataset = np.memmap(input_tokenized_file, dtype=np.uint16, mode='r')

        history = []
        last_saved_pos = 0

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=100, warmup=0, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tb_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(steps):
                if step == 0:
                    torch.cuda.memory._record_memory_history()

                with torch.profiler.record_function(f"## get_batch step {step} ##"):
                    batch, labels = run_get_batch(
                        dataset, 
                        self.config.pipeline_config.batch_size, 
                        self.config.pipeline_config.context_length, 
                        self.device
                    )

                if step == 3:
                    torch.cuda.memory._dump_snapshot(f"{self.tb_dir}/my_snapshot.pickle")

                loss = self.train_step(step, batch, labels)
                if step % self.config.pipeline_config.loss_log_freq == 0:
                    loss_val = loss.item()
                    logging.info(f'{step=}: loss={loss_val}')
                    history.append({'step': step, 'loss': loss_val})
                
                prof.step()

                if step > 0 and (step % self.config.pipeline_config.loss_log_freq == 0):
                    ck_file = self.checkpoint(step)

                    history.append({'step': step, 'ck_file': ck_file, 'loss': loss.item()})
                    with open(f"{self.run_dir}/loss.text", "a") as f:
                        while last_saved_pos < len(history):
                            print(json.dumps(history[last_saved_pos]), file=f)
                            last_saved_pos += 1

    def train_step(self, step, in_indices, labels):
        with torch.profiler.record_function(f"## foward step {step} ##"):
            lm_output = self.model(in_indices)
        
        self.model.zero_grad()

        with torch.profiler.record_function(f"## loss step {step} ##"):
            voc_size = lm_output.shape[-1]
            loss = run_cross_entropy(lm_output.reshape(-1, voc_size), labels.reshape(-1))
        # print(loss.shape)

        with torch.profiler.record_function(f"## backward step {step} ##"):
            loss.backward()

            run_gradient_clipping(self.model.parameters(),
                                max_l2_norm=self.config.pipeline_config.max_l2_norm)

        with torch.profiler.record_function(f"## optimizer step {step} ##"):
            self.optimizer.step()

        return loss
    
    def checkpoint(self, step):
        path_dir = f"{self.run_dir}/checkpoints"
        output_dir = Path(path_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ts = int(time.time())
        file_path = f"{path_dir}/{ts}.pt"
        run_save_checkpoint(self.model, self.optimizer, step, file_path)

        return file_path


if __name__ == "__main__":
    pipeline = TrainPipeline.create(data_base_dir='./data',
                                    run_id=1, 
                                    test=True,
                                    device='mps')
    pipeline.before_training()
    pipeline.train(input_tokenized_file='./data/TinyStoriesV2-GPT4-valid.bin', steps=3)
