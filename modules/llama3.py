from typing import List, Optional

import fire

from llama import Dialog, Llama

def main(
        ckpt_dir: str, 
        tokenizer_path: str, 
        temperature: float = 0.6, 
        top_p: float = 0.9, 
        max_seq_len: int = 512, 
        max_batch_size: int = 4, 
        max_gen_len: Optional[int] = None
        ):
    # generator = Llama.build(
    #     ckpt_dir=ckpt_dir,
    #     tokenizer_path=tokenizer_path, 
    #     max_seq_len=max_seq_len, 
    #     max_batch_size=max_batch_size
    #     )

    print("hello")

if __name__ == "__main__":
    fire.Fire(main)