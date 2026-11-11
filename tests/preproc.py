
from concurrent.futures import ProcessPoolExecutor
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH


import logging
import numpy as np


def tokenize_chunk(tokenizer, text_chunk):
    return tokenizer.encode(text_chunk)


def tokenize_file(text_file, output_path):
    logging.info('Start tokenize_file')
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )
    print(tokenizer.get_voc_size())
    logging.info('tokenizer initiated')
    
    chunk_size = 10_000
    process_num = 20
    ## TBD: do we need to append special_tokens=["<|endoftext|>"] at the of the file?
    with open(text_file, "r", encoding="utf-8") as in_f, \
        open(output_path, "wb") as out_f, ProcessPoolExecutor() as executor:

        i = 0
        futures = []
        while True:
            text_chunk = in_f.read(chunk_size)
            if not text_chunk:
                break
        
            futures.append(executor.submit(tokenize_chunk, tokenizer, text_chunk))
            
            if len(futures) >= process_num:
                for f in futures:
                    token_ids = f.result()
                    # Write directly to disk as binary bytes without caching in RAM
                    np.array(token_ids, dtype=np.uint16).tofile(out_f)

                futures = []

                i += 1
                logging.info(f'complete processed {i} batch with {chunk_size=} and {process_num=}')
    
        if len(futures) >= process_num:
            for f in futures:
                token_ids = f.result()
                # Write directly to disk as binary bytes without caching in RAM
                np.array(token_ids, dtype=np.uint16).tofile(out_f)
            i += 1
            logging.info(f'processed {i} batch with {text_chunk=} and {process_num=}')


    logging.info('completed tokenize_file')

if __name__ == "__main__":
    text_file = './data/TinyStoriesV2-GPT4-valid.txt'
    bin_file = './data/TinyStoriesV2-GPT4-valid.bin'
    tokenize_file(text_file, bin_file)