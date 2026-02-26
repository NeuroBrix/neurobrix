# Model Hub

The NeuroBrix Hub hosts pre-built `.nbx` containers ready for inference.

## Browse Models

```bash
neurobrix hub
```

Filter by category:

```bash
neurobrix hub --category IMAGE
neurobrix hub --category LLM
```

Search:

```bash
neurobrix hub --search sana
```

## Available Models

### Image Generation

| Model | Org | Description |
|-------|-----|-------------|
| Flex.1-alpha | ostris | Flexible image generation |
| FLUX.2-dev | FLUX | High-quality diffusion |
| Sana_1600M_4Kpx_BF16 | sana | 4K resolution, 1.6B params |
| Sana_1600M_1024px_MultiLing | sana | Multilingual, 1024px |
| PixArt-Sigma-XL-2-1024-MS | pixart | Multi-subject generation |
| PixArt-XL-2-1024-MS | pixart | XL generation |
| Janus-Pro-7B | janus | VQ autoregressive images |

### Language Models

| Model | Org | Description |
|-------|-----|-------------|
| deepseek-moe-16b-chat | deepseek | Mixture-of-experts chat |
| Qwen3-30B-A3B-Thinking-2507 | qwen | Reasoning model |

## Import a Model

```bash
neurobrix import <org>/<model>
```

Example:

```bash
neurobrix import sana/1600m-1024
```

The model is downloaded and extracted to `~/.neurobrix/cache/`.

## Manage Models

```bash
# List installed models
neurobrix list

# Remove a model
neurobrix remove 1600m-1024

# Clean all
neurobrix clean --all -y
```
