from contextlib import nullcontext
import mmap
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from model import Transformer, ModelArgs
import torch
import os
import numpy as np
import torch.nn.functional as F


def noise_input(x, eps=1e-3):
    b, l = x.shape
    t = torch.rand(b, device=x.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=x.device) < p_mask
    # Use the mask token ID from the multilingual tokenizer
    mask_token_id = 103  # [MASK] token ID for multilingual tokenizer
    noisy_batch = torch.where(masked_indices, mask_token_id, x)
    return noisy_batch, masked_indices, p_mask

_memmap_cache = {}

def load_memmap(split):
    """Load a persistent np.memmap and cache it globally."""
    global _memmap_cache
    if split not in _memmap_cache:
        file_path = os.path.join(data_dir, f"{split}.bin")
        _memmap_cache[split] = np.memmap(file_path, dtype=np.uint16, mode='r')
        try:
            # Apply madvise optimization (Linux/MacOS)
            _memmap_cache[split]._mmap.madvise(mmap.MADV_SEQUENTIAL)
        except AttributeError:
            pass  # Ignore if not supported on the OS
    return _memmap_cache[split]

def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for X, Y, masked_indices, p_mask in data_loader:
            X, Y = X.to(device), Y.to(device)
            masked_indices = masked_indices.to(device)
            p_mask = p_mask.to(device)
            
            logits = model(X, targets=(Y, masked_indices, p_mask))
            loss = model.last_loss
            
            # Count masked tokens for normalization
            num_masked = masked_indices.sum().item()
            total_loss += loss.item() * num_masked
            total_tokens += num_masked
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def get_batch(split, batch_size, block_size, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # if split == 'train':
    #     data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # else:
    #     data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    data = load_memmap(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    y = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    x = torch.clone(y)
    x, masked_indices, p_mask = noise_input(x)
   

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        masked_indices = masked_indices.pin_memory().to(device, non_blocking=True)
        p_mask = p_mask.pin_memory().to(device, non_blocking=True)
    
    else:
        x, y = x.to(device), y.to(device)
        p_mask = p_mask.to(device)

    return x, y, masked_indices, p_mask


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=103):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, l).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 103 for multilingual tokenizer.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    x[x==mask_id] = 0
    return x

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Tiny-LLaDiff on a specified dataset.")
    parser.add_argument('--dataset', choices=['shakespeare', 'enwik8', 'tamil'], default='shakespeare', 
                        help="Dataset to use for training. Options: 'shakespeare', 'enwik8'. Default: 'shakespeare'.")
    
    args = parser.parse_args()

    cwd = os.path.dirname(__file__)
    dataset = args.dataset
    data_dir = os.path.join(cwd, 'data', dataset)
    BLOCK_SIZE = 64  # Reduced from 256 to 64 for memory efficiency and faster training
    BATCH_SIZE = 32  # Increased from 16 to 32 since we reduced sequence length
    NUM_STEPS = 100_000
    VAL_STEPS = 50
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    
    args = ModelArgs(
        dim=384,
        n_layers=8,
        n_heads=8,
        vocab_size=250048,  # Updated for multilingual tokenizer
        multiple_of=256,
        max_seq_len=64,  # Reduced from 2048 to 64 for memory efficiency and faster training
        dropout=0.1
    )
    
    CHECKPOINT_NAME = f"{dataset}_dim={args.dim}_heads={args.n_heads}_layers={args.n_layers}_dropout={args.dropout:.1f}"
    checkpoint_dir = os.path.join(cwd, "checkpoints", CHECKPOINT_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = Transformer(args).to(device)
    # checkpoint_dir = os.path.join(cwd, "checkpoints", "enwik8_dim=256_heads=8_layers=8_dropout=0.1")
    # checkpoint_path = os.path.join(checkpoint_dir, "model_7500.pth")
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), weight_decay=0.1)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")
    
    # Validate data loading
    print("Validating data loading...")
    try:
        X_test, Y_test, masked_indices, p_mask = get_batch('train', 2, BLOCK_SIZE, device)
        print(f"Data shapes - X: {X_test.shape}, Y: {Y_test.shape}")
        print(f"Masked tokens: {masked_indices.sum().item()}")
        print("Data loading validation successful!")
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for step in tqdm(range(NUM_STEPS), desc='Training steps', total=NUM_STEPS):
        model.train()
        X, Y, masked_indices, p_mask = get_batch('train', BATCH_SIZE, BLOCK_SIZE, device)
        
        with ctx:
            logits = model(X, targets=(Y, masked_indices, p_mask))
            loss = model.last_loss
            scaler.scale(loss).backward()
        
        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()  # Step the scheduler
        train_losses.append(loss.item())
        
        if step % 70 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step} loss {loss.item():.3f} lr {current_lr:.2e}")
        
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for val_step in tqdm(range(VAL_STEPS), desc='Validation steps', total=VAL_STEPS):
                    X_val, Y_val, masked_indices, p_mask = get_batch('val', BATCH_SIZE, BLOCK_SIZE, device)
                    logits = model(X_val, targets=(Y_val, masked_indices, p_mask))
                    loss = model.last_loss
                    total_val_loss += loss.item()
                avg_val_loss = total_val_loss / VAL_STEPS
                val_losses.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model_args": args,
                    "step": step,
                    "best_val_loss": best_val_loss
                }, os.path.join(checkpoint_dir, f"model_{step}.pth"))
                print("*"*50)
                print(f"Checkpoint saved at step {step} with validation loss {avg_val_loss:.3f}")
                print("*"*50)

            # Calculate perplexity
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            print(f"Validation perplexity: {val_perplexity:.2f}")
            
            # Generate sample text
            try:
                prompt = "All:\nWhy are"
                tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                # Truncate prompt if it's too long for the reduced sequence length
                if len(tokenized_prompt) > BLOCK_SIZE // 2:
                    tokenized_prompt = tokenized_prompt[:BLOCK_SIZE // 2]
                tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long).to(device)
                output = generate(model, tokenized_prompt.unsqueeze(0), 
                               steps=32, gen_length=32, block_length=16, temperature=0.7)
                # The generate function returns a tensor, we need to get the first sequence
                generated_tokens = output[0] if output.dim() > 1 else output
                res = tokenizer.decode(generated_tokens.cpu().tolist())
                print("Generated text:")
                print(res)
            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Mean validation loss: {avg_val_loss:.3f}")
            
            # Plot training curves
            plt.figure(figsize=(10,5))
            plt.loglog(train_losses, label='Train Loss', alpha=0.7)
            plt.loglog(range(0, len(val_losses) * 1000, 1000), val_losses, label='Validation Loss', marker='o', linestyle='dashed')
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training and Validation Loss Over Time")
            plt.savefig(os.path.join(checkpoint_dir, "loss_plot.png"))
            plt.close()
