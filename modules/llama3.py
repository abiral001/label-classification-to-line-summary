from typing import List, Optional

from argparse import ArgumentParser

from .llama import Dialog, Llama

class Llama3:
    def __init__(self, ckpt_dir, tokenizer_path):
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path

    def generate(self, embedding_inps, label, length):
        self.embedding_inps = embedding_inps
        self.labels = label
        self.output_length = length
        return self.useLlama()

    def useLlama(self):
        generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path, 
            max_seq_len=4096,
            max_batch_size=4
        )
        dialogs: List[Dialog] = [
            [
                {
                    "role": "user", 
                    "content": self.embedding_inps
                },
                {
                    "role": "user",
                    "content": f"Write down the summary in {self.output_length} words or less of the above text using the following keywords: ({', '.join(self.labels)})"
                }
            ],
        ]
        results = generator.chat_completion(
            dialogs,
            max_gen_len=1024,
            temperature=0.6,
            top_p=0.9
        )
        final_output = []
        for _, result in zip(dialogs, results):
            final_output.append(f'{result["generation"]["content"]}')
        return "\n".join(final_output)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=None)
    args = parser.parse_args()