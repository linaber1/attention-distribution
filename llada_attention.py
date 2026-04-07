# ── LLaDA Attention Heatmap — Denoising Steps ────────────────────────────────
# Visualizes attention distribution per token per denoising step for LLaDA-8B-Instruct.
# Probes the lost-in-the-middle phenomenon in discrete diffusion.
#
# Usage:
#   python llada_attention_viz.py
#
# Outputs:
#   attn_per_step.png
#   lost_in_middle.png
#   attn_step{N}_layer{M}.png
#   layer_comparison_step{N}.png
#   attn_evolution.gif

# ── Cell 1: Imports & Configuration ──────────────────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = 'GSAI-ML/LLaDA-8B-Instruct'
MASK_ID    = 126336
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE      = torch.bfloat16

# Generation params (keep small for speed / VRAM)
GEN_LENGTH   = 64    # tokens to generate
STEPS        = 32    # denoising steps  (≤ GEN_LENGTH)
BLOCK_LENGTH = 64    # same as gen_length → full bidirectional attention

# Which layers to average for the summary heatmap (-1 = all)
LAYERS_TO_AVERAGE = -1   # or e.g. [16, 17, 18] for specific layers

# Prompt to probe
PROMPT = (
    "In a small village nestled between two mountains, there lived a young baker "
    "named Elara who discovered a recipe that could make people remember their "
    "happiest childhood memory. The recipe required seven rare ingredients, each "
    "hidden in a different corner of the world. "
    "What happened when Elara finally baked the bread?"
)

print(f"Device : {DEVICE}")
print(f"Prompt : {PROMPT[:80]}...")


# ── Cell 2: Load Model ────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.padding_side != 'left':
    tokenizer.padding_side = 'left'

print("Loading model (this may take a few minutes on first run)...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    # NOTE: output_attentions=True is NOT supported by LLaDA — removed.
    # Attention weights are captured via SDPA monkey-patch below.
).to(DEVICE).eval()

print(f"Model loaded — {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")


# ── Cell 3: SDPA Monkey-Patch for Attention Capture ──────────────────────────
# LLaDA raises ValueError for output_attentions=True, so we intercept attention
# weights by temporarily replacing F.scaled_dot_product_attention during each
# forward pass. This works regardless of Flash Attention or standard SDPA path.

_attention_store: list = []   # filled each forward call: [(call_idx, tensor)]

def patch_sdpa():
    """
    Replace F.scaled_dot_product_attention globally with a version that:
      1. Computes attention weights explicitly (no fused kernel).
      2. Stores them in _attention_store as (call_idx, [B, H, S, S]).
      3. Returns the correct output.
    Returns a restore function — call it after the forward pass.
    """
    _original_sdpa = F.scaled_dot_product_attention
    _call_counter = [0]   # mutable so the closure can mutate it

    def _hooked_sdpa(query, key, value, attn_mask=None,
                     dropout_p=0.0, is_causal=False, **kwargs):
        scale = query.shape[-1] ** -0.5
        attn_weight = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        if is_causal:
            s = query.shape[-2]
            causal_mask = torch.triu(
                torch.full((s, s), float('-inf'), device=query.device), diagonal=1
            )
            attn_weight = attn_weight + causal_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weight = attn_weight.masked_fill(~attn_mask, float('-inf'))
            else:
                attn_weight = attn_weight + attn_mask.float()
        attn_weight = torch.softmax(attn_weight, dim=-1)
        _attention_store.append((_call_counter[0], attn_weight.detach().cpu()))
        _call_counter[0] += 1
        out = torch.matmul(attn_weight, value.float()).to(query.dtype)
        return out

    F.scaled_dot_product_attention = _hooked_sdpa

    def _restore():
        F.scaled_dot_product_attention = _original_sdpa

    return _restore


# ── Cell 4: Instrumented Denoising Loop ──────────────────────────────────────
# Mirrors LLaDA's generate() but captures attention at every step.

def get_num_transfer_tokens(mask_index, steps):
    """Distribute unmasking evenly across steps (from original LLaDA code)."""
    mask_num = mask_index.sum(dim=-1, keepdim=True)   # [B, 1]
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(mask_index.shape[0], steps).clone()
    for i in range(mask_index.shape[0]):
        num_transfer_tokens[i, :remainder[i].item()] += 1
    return num_transfer_tokens  # [B, steps]


@torch.no_grad()
def generate_with_attention(
    model, tokenizer, prompt_text,
    steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
    temperature=0., remasking='low_confidence',
):
    """
    Returns
    -------
    generated_ids   : [seq_len]  final token ids
    step_attentions : list of length `steps`
                      each element is np.ndarray [n_layers, seq_len, seq_len]
                      (averaged over heads)
    prompt_len      : int  number of prompt tokens
    token_labels    : list[str]  decoded token strings
    step_sequences  : list of [seq_len] arrays showing token state at each step
    """
    # Tokenise prompt
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors='pt'
    ).to(DEVICE)
    prompt_len = prompt_ids.shape[1]
    total_len  = prompt_len + gen_length

    # Initialise: prompt + fully masked response
    x = torch.cat([
        prompt_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=DEVICE)
    ], dim=1)                                              # [1, total_len]
    attn_mask = torch.ones(1, total_len, device=DEVICE)

    # Block setup (single block for simplicity)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    step_attentions = []    # one entry per denoising step
    step_sequences  = []    # token state at each step

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end   = block_start + block_length
        block_mask  = (x[:, block_start:block_end] == MASK_ID)   # [1, block_len]
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)  # [1, steps]

        for i in range(steps_per_block):
            _attention_store.clear()
            mask_index = (x == MASK_ID)

            # ── Patch SDPA, run forward, restore ─────────────────────────────
            restore_sdpa = patch_sdpa()
            try:
                # output_attentions removed — captured via SDPA patch above
                out = model(x, attention_mask=attn_mask)
            finally:
                restore_sdpa()
            # ─────────────────────────────────────────────────────────────────

            logits = out.logits   # [1, total_len, vocab_size]

            # Aggregate: patch stores (call_idx, [B, H, S, S])
            # call_idx corresponds to layer order (one SDPA call per layer in LLaDA)
            if _attention_store:
                layers_sorted = sorted(_attention_store, key=lambda t: t[0])
                attn_stack = np.stack(
                    [t.mean(dim=1).squeeze(0).float().numpy() for _, t in layers_sorted]
                )   # [n_layers, seq, seq]
                step_attentions.append(attn_stack)

            step_sequences.append(x[0].cpu().numpy().copy())

            # Sample / unmask
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*x.shape)
            else:
                x0 = logits.argmax(dim=-1)

            # Confidence = softmax prob of predicted token
            confidence = logits.softmax(dim=-1).max(dim=-1).values  # [1, total_len]

            # Only unmask tokens that are still masked
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -1))

            # Select top-k confident masked tokens in this block to unmask
            n_unmask = int(num_transfer[0, i].item())
            block_conf = confidence[0, block_start:block_end]
            block_x0   = x0[0, block_start:block_end]
            block_cur  = x[0, block_start:block_end]

            if remasking == 'low_confidence':
                # Unmask the n_unmask tokens with highest confidence
                unmask_indices = block_conf.topk(n_unmask).indices
            else:  # random
                masked_pos = (block_cur == MASK_ID).nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(masked_pos))[:n_unmask]
                unmask_indices = masked_pos[perm]

            new_block = block_cur.clone()
            new_block[unmask_indices] = block_x0[unmask_indices]
            x[0, block_start:block_end] = new_block

    # Decode token labels for final sequence
    final_ids = x[0].cpu().tolist()
    token_labels = [tokenizer.decode([tid], skip_special_tokens=False) for tid in final_ids]
    # Shorten labels for display
    token_labels = [
        (lbl[:6] + '…' if len(lbl) > 7 else lbl).replace('\n', '↵').replace(' ', '·')
        for lbl in token_labels
    ]

    return x[0].cpu(), step_attentions, prompt_len, token_labels, step_sequences


print("Running generation...")
final_ids, step_attentions, prompt_len, token_labels, step_sequences = \
    generate_with_attention(model, tokenizer, PROMPT)

total_len = len(final_ids)
print(f"Done. Steps captured: {len(step_attentions)} | Sequence length: {total_len}")
print(f"Prompt tokens: {prompt_len} | Generated tokens: {total_len - prompt_len}")


# ── Cell 5: Attention Heatmap per Denoising Step ──────────────────────────────
# Shows attention FROM every generated token TO the full sequence.
# Rows   = denoising steps (0 = most masked, last = clean)
# Columns = source positions attended to
# Value  = mean attention weight (averaged across all layers, then over gen tokens)

def plot_step_heatmap(step_attentions, prompt_len, token_labels, layers='all'):
    """
    For each step: average the [n_layers, seq, seq] attention over
    - selected layers
    - all generated-token rows (the query dimension)
    to get a [seq] vector = "how much do generated tokens attend to each position?"
    """
    n_steps = len(step_attentions)
    seq_len = step_attentions[0].shape[-1]

    heatmap = np.zeros((n_steps, seq_len))

    for s, attn in enumerate(step_attentions):
        # attn: [n_layers, seq, seq]
        if layers == 'all':
            layer_mean = attn.mean(axis=0)        # [seq, seq]
        else:
            layer_mean = attn[layers].mean(axis=0)
        # Average attention from generated tokens (rows prompt_len:) to all positions
        gen_attn = layer_mean[prompt_len:, :]     # [gen_len, seq]
        heatmap[s] = gen_attn.mean(axis=0)        # [seq]

    # Normalise each row to [0,1] for visual clarity
    row_max = heatmap.max(axis=1, keepdims=True).clip(min=1e-9)
    heatmap_norm = heatmap / row_max

    fig, ax = plt.subplots(figsize=(min(seq_len * 0.25, 28), max(n_steps * 0.35, 6)))

    im = ax.imshow(heatmap_norm, aspect='auto', cmap='inferno',
                   vmin=0, vmax=1, interpolation='nearest')

    # Prompt / generation boundary
    ax.axvline(prompt_len - 0.5, color='cyan', linewidth=2, linestyle='--', label='Prompt | Generation')

    ax.set_xlabel('Token Position (source)', fontsize=12)
    ax.set_ylabel('Denoising Step  (0 = fully masked → last = clean)', fontsize=12)
    ax.set_title('Attention from Generated Tokens → Each Position, per Denoising Step\n'
                 '(row-normalised; brighter = more attended)', fontsize=13)

    # X-tick labels: show every N tokens to avoid clutter
    tick_every = max(1, seq_len // 30)
    xtick_pos = list(range(0, seq_len, tick_every))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([token_labels[i] for i in xtick_pos], rotation=90, fontsize=7)

    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([f'Step {i}' for i in range(n_steps)], fontsize=7)

    plt.colorbar(im, ax=ax, label='Normalised attention weight')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig('attn_per_step.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → attn_per_step.png")

plot_step_heatmap(step_attentions, prompt_len, token_labels)


# ── Cell 6: Lost-in-the-Middle Analysis ──────────────────────────────────────
# For each denoising step, plot the mean attention from generated tokens
# to ONLY the PROMPT positions — as a line chart over prompt positions.
# If there's a U-shaped curve (high at ends, dip in the middle), that's
# the lost-in-the-middle pattern.

def plot_lost_in_middle(step_attentions, prompt_len, token_labels,
                        highlight_steps=None):
    """
    highlight_steps: indices of steps to draw as coloured lines (default: all).
    """
    n_steps = len(step_attentions)
    if highlight_steps is None:
        highlight_steps = list(range(n_steps))

    cmap = plt.cm.plasma
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # ── Left: attention to prompt positions per step ──
    ax = axes[0]
    prompt_positions = np.arange(prompt_len)

    for s in highlight_steps:
        attn = step_attentions[s]             # [n_layers, seq, seq]
        layer_mean = attn.mean(axis=0)        # [seq, seq]
        gen_to_prompt = layer_mean[prompt_len:, :prompt_len]  # [gen, prompt]
        per_prompt_pos = gen_to_prompt.mean(axis=0)           # [prompt]
        colour = cmap(s / max(n_steps - 1, 1))
        ax.plot(prompt_positions, per_prompt_pos, color=colour,
                alpha=0.7, linewidth=1.2, label=f'Step {s}')

    ax.set_xlabel('Prompt Token Position', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Attention from Generated Tokens → Prompt Positions\n'
                 'per Denoising Step  (purple=early, yellow=late)', fontsize=12)
    ax.set_xticks(range(0, prompt_len, max(1, prompt_len // 20)))
    ax.set_xticklabels(
        [token_labels[i] for i in range(0, prompt_len, max(1, prompt_len // 20))],
        rotation=90, fontsize=7
    )

    # Shade first/last 20% of prompt
    edge = max(1, int(prompt_len * 0.2))
    ax.axvspan(0, edge, alpha=0.08, color='green', label='First 20%')
    ax.axvspan(prompt_len - edge, prompt_len, alpha=0.08, color='blue', label='Last 20%')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    # ── Right: attention aggregated into 10 bins over prompt ──
    ax2 = axes[1]
    n_bins = 10
    bin_edges = np.linspace(0, prompt_len, n_bins + 1, dtype=int)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) // 2 for i in range(n_bins)]

    # Average over all steps for a summary view
    all_per_prompt = []
    for s in range(n_steps):
        attn = step_attentions[s].mean(axis=0)
        gen_to_prompt = attn[prompt_len:, :prompt_len]
        all_per_prompt.append(gen_to_prompt.mean(axis=0))

    mean_attn = np.stack(all_per_prompt).mean(axis=0)  # [prompt_len]
    std_attn  = np.stack(all_per_prompt).std(axis=0)

    binned_mean = [mean_attn[bin_edges[i]:bin_edges[i+1]].mean() for i in range(n_bins)]
    binned_std  = [std_attn [bin_edges[i]:bin_edges[i+1]].mean() for i in range(n_bins)]
    bin_labels  = [f'{int(bin_edges[i]/prompt_len*100)}%' for i in range(n_bins)]

    bars = ax2.bar(range(n_bins), binned_mean, yerr=binned_std,
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_bins)),
                   capsize=4, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels(bin_labels, fontsize=10)
    ax2.set_xlabel('Prompt Region (% of prompt length)', fontsize=12)
    ax2.set_ylabel('Mean Attention Weight (avg over all steps)', fontsize=12)
    ax2.set_title('Lost-in-the-Middle Summary\n'
                  'A U-shape → middle positions are under-attended', fontsize=12)

    # Reference line: uniform attention
    ax2.axhline(np.mean(binned_mean), color='red', linestyle='--',
                linewidth=1.5, label='Uniform baseline')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('lost_in_middle.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → lost_in_middle.png")

# Show every step (use highlight_steps=[0,8,16,24,31] to reduce clutter)
plot_lost_in_middle(step_attentions, prompt_len, token_labels)


# ── Cell 7: Per-Layer / Per-Head Explorer ─────────────────────────────────────
# Pick any step and inspect individual layers or heads.

INSPECT_STEP  = len(step_attentions) // 2   # middle step
INSPECT_LAYER = 15                          # change to any layer index
MAX_HEADS_SHOWN = 8

def plot_layer_heads(step_attentions, step_idx, layer_idx, token_labels,
                     max_heads=MAX_HEADS_SHOWN):
    """Show per-head attention heatmap for a single step + layer."""
    attn_all_layers = step_attentions[step_idx]   # [n_layers, seq, seq]
    n_layers = attn_all_layers.shape[0]

    if layer_idx >= n_layers:
        print(f"Layer {layer_idx} out of range (model has {n_layers} captured layers).")
        return

    attn_layer = attn_all_layers[layer_idx]        # [seq, seq]  (head-averaged)
    seq_len = attn_layer.shape[0]

    fig, ax = plt.subplots(figsize=(min(seq_len * 0.22, 22), min(seq_len * 0.22, 22)))
    tick_every = max(1, seq_len // 30)
    tick_pos = list(range(0, seq_len, tick_every))

    im = ax.imshow(attn_layer, cmap='magma', aspect='auto', vmin=0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([token_labels[i] for i in tick_pos], rotation=90, fontsize=6)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([token_labels[i] for i in tick_pos], fontsize=6)
    ax.axvline(prompt_len - 0.5, color='cyan', linewidth=1.5, linestyle='--')
    ax.axhline(prompt_len - 0.5, color='cyan', linewidth=1.5, linestyle='--')
    ax.set_title(f'Attention Map — Step {step_idx}, Layer {layer_idx} (head-averaged)\n'
                 f'Cyan line separates prompt ({prompt_len} tok) from generation', fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'attn_step{step_idx}_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → attn_step{step_idx}_layer{layer_idx}.png")


def plot_layer_comparison(step_attentions, step_idx, token_labels, prompt_len,
                          n_cols=4):
    """Grid of all layers at a given step — attention FROM generated TO prompt."""
    attn_all = step_attentions[step_idx]   # [n_layers, seq, seq]
    n_layers = attn_all.shape[0]
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.5), squeeze=False)
    fig.suptitle(f'Per-Layer Attention (gen→prompt) at Step {step_idx}', fontsize=13)

    prompt_positions = np.arange(prompt_len)
    for li in range(n_layers):
        row, col = divmod(li, n_cols)
        ax = axes[row][col]
        gen_to_prompt = attn_all[li, prompt_len:, :prompt_len].mean(axis=0)
        ax.plot(prompt_positions, gen_to_prompt, linewidth=1.0)
        ax.set_title(f'Layer {li}', fontsize=8)
        ax.set_xlim(0, prompt_len - 1)
        ax.tick_params(labelsize=6)
        ax.axvspan(0, max(1, int(prompt_len*0.2)), alpha=0.1, color='green')
        ax.axvspan(prompt_len - max(1, int(prompt_len*0.2)), prompt_len,
                   alpha=0.1, color='blue')

    # Hide unused axes
    for li in range(n_layers, n_rows * n_cols):
        row, col = divmod(li, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'layer_comparison_step{step_idx}.png', dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved → layer_comparison_step{step_idx}.png")


# Full attention map for one step + layer
plot_layer_heads(step_attentions, INSPECT_STEP, INSPECT_LAYER, token_labels)

# Per-layer breakdown
plot_layer_comparison(step_attentions, INSPECT_STEP, token_labels, prompt_len)


# ── Cell 8: Animated GIF (optional) ──────────────────────────────────────────
# Creates an animated heatmap showing how attention evolves across denoising steps.
# Requires: pip install pillow

from PIL import Image
import io

def make_attention_gif(step_attentions, prompt_len, token_labels,
                       output_path='attn_evolution.gif', fps=3):
    frames = []
    n_steps = len(step_attentions)

    for s, attn in enumerate(step_attentions):
        layer_mean = attn.mean(axis=0)         # [seq, seq]
        gen_to_prompt = layer_mean[prompt_len:, :prompt_len].mean(axis=0)  # [prompt]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(prompt_len), gen_to_prompt,
               color=plt.cm.plasma(np.linspace(0, 1, prompt_len)))
        ax.set_ylim(0, gen_to_prompt.max() * 1.4 + 1e-9)
        ax.set_title(f'Denoising Step {s}/{n_steps-1} — Attention from Gen → Prompt',
                     fontsize=11)
        ax.set_xlabel('Prompt token position')
        ax.set_ylabel('Mean attention weight')

        edge = max(1, int(prompt_len * 0.2))
        ax.axvspan(0, edge, alpha=0.1, color='green')
        ax.axvspan(prompt_len - edge, prompt_len, alpha=0.1, color='blue')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"GIF saved → {output_path}")

make_attention_gif(step_attentions, prompt_len, token_labels)
