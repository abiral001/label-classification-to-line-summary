from typing import List
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