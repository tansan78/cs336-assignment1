import torch

from tests.adapters import *
from tests.config import *
from tests.model import *
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH


def top_p_sampling(output: Float[Tensor, "batch_size seq_len voc_size"], temp=0.1, p=0.5):
    output = output[:, -1, :]
    softmax_output = torch.softmax(output / temp, dim=-1)

    sorted_output, sorted_indices = torch.sort(softmax_output, dim=-1, descending=True)
    cumsum_output = torch.cumsum(sorted_output, dim=-1)
    
    filtered_output = sorted_output.masked_fill(cumsum_output - sorted_output > p, 0.0)
    norm_filtered_output = filtered_output / torch.sum(filtered_output, dim=-1, keepdim=True)

    selected_sorted_indices = torch.multinomial(norm_filtered_output, num_samples=1)

    return torch.gather(sorted_indices, dim=1, index=selected_sorted_indices)


@torch.no_grad()
def generate(prompt, model_path, max_len=20, device='cpu'):
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )
    config = TransformerConfig()
    config.vocab_size = len(tokenizer._vocab)
    model = MyLLM(config).to(device)
    run_load_checkpoint(
        model_path, model, None
    )
    
    input = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)

    for i in range(max_len):
        # padded_input_tensor = torch.nn.functional.pad(input_tensor, config.max_seq_len, mode="constant", value=0)
        output = model(input_tensor)

        pred_token = top_p_sampling(output)
        input_tensor = torch.cat([input_tensor, pred_token], dim=-1)

    return tokenizer.decode(input_tensor.squeeze(0).tolist())
