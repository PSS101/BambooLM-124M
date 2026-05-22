```markdown
    ██████╗  █████╗ ███╗   ███╗██████╗  ██████╗  ██████╗ ██╗     ███╗   ███╗       ██╗██████╗ ██╗  ██╗███╗   ███╗
    ██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔═══██╗██║     ████╗ ████║      ███║╚════██╗██║  ██║████╗ ████║
    ██████╔╝███████║██╔████╔██║██████╔╝██║   ██║██║   ██║██║     ██╔████╔██║█████╗╚██║ █████╔╝███████║██╔████╔██║
    ██╔══██╗██╔══██║██║╚██╔╝██║██╔══██╗██║   ██║██║   ██║██║     ██║╚██╔╝██║╚════╝ ██║██╔═══╝ ╚════██║██║╚██╔╝██║
    ██████╔╝██║  ██║██║ ╚═╝ ██║██████╔╝╚██████╔╝╚██████╔╝███████╗██║ ╚═╝ ██║       ██║███████╗     ██║██║ ╚═╝ ██║
    ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝       ╚═╝╚══════╝     ╚═╝╚═╝     ╚═╝
```
## BambooLM-124M Configuration

BambooLM-124M is a 124-million parameter model based on the GPT-2 Small architecture.

| Parameter | Value | Description |
|---|---|---|
| **Vocabulary Size** (`vocab_size`) | 50,257 | Standard GPT-2 byte-pair encoding (BPE) |
| **Context Length** (`block_size` / `T`) | 1,024 | Maximum sequence length |
| **Transformer Layers** (`n_layer`) | 12 | Number of transformer blocks |
| **Attention Heads** (`n_head`) | 12 | Number of query/key/value heads |
| **Embedding Size** (`n_embd`) | 768 | Dimensionality of hidden states |
| **Micro-Batch Size** (`B`) | 8 | Sequences per forward/backward pass |
| **Total Batch Size** | 524,288 | Total tokens per gradient update (~0.5M) |
