import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import numpy as np

# --- 1. Load Model ---
# Use a small model suitable for CPU
model_name = "distilgpt2"
print(f"Loading model: {model_name}...")
# Use float32 for clarity, avoids potential complexities of bfloat16 etc.
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

# --- 2. Isolate Weights ---
# Let's pick a specific weight matrix from one layer for demonstration
# Example: Query/Key/Value projection weights from the first attention block
target_layer = model.transformer.h[0].attn.c_attn
# Get the weights as a NumPy array for easier manipulation (optional)
weights = target_layer.weight.data.detach().clone()
print(f"Selected weight matrix shape: {weights.shape}") # Shape: [hidden_size, 3*hidden_size]

# --- 3. Select a Block ---
# Zero-shot quantization often works on blocks. Let's take a small block.
# Block size K (as mentioned in the paper, though values vary)
# Let's use K=64 for illustration
K = 64
# Take the first block from the first row of the weight matrix
# Ensure we don't go out of bounds
if weights.shape[1] >= K:
    weight_block_torch = weights[0, :K]
    print(f"Selected block of {K} weights from the first row.")
    # Example: tensor([-0.0112,  0.0381, ...])
    print(f"First few weights in block: {weight_block_torch[:5]}")
else:
    print(f"Weight matrix too small for block size {K}. Adjust K or choose a different layer.")
    exit()

# Detach and convert to numpy for easier manual simulation if needed
weight_block_np = weight_block_torch.numpy()


# (Keep the code from Step 4 above this)

# --- 4. Simplified Quantization Functions ---

def get_quantization_alphabet(bits=4):
    """
    Creates a simplified, symmetric quantization alphabet.
    For illustrative purposes, not exactly NF4 or FP4.
    """
    # Calculate the number of positive levels (excluding zero if bits is odd)
    num_positive_levels = 2**(bits - 1) - 1
    # Create positive levels symmetrically spaced between 0 and 1
    # Linspace includes endpoints, so we generate one extra point and remove 0 later
    positive_levels = np.linspace(0, 1, num_positive_levels + 1)

    # Combine positive, negative levels, and zero
    levels = np.concatenate([-positive_levels[:0:-1], positive_levels])
    levels.sort() # Ensure sorted order
    return torch.tensor(levels, dtype=torch.float32)

# Define our illustrative quantization alphabet (e.g., 4-bit equivalent)
# A 4-bit scheme has 2^4 = 16 levels.
A = get_quantization_alphabet(bits=4)
print(f"\nUsing Quantization Alphabet (bits=4, {len(A)} levels):")
# Format output for better readability
print(np.array2string(A.numpy(), formatter={'float_kind':lambda x: "%.3f" % x}))


def quantize_dequantize_block(block, alphabet):
    """
    Applies simplified quantization and dequantization based on absmax scaling.
    Returns the dequantized block and the scaling factor.
    """
    # Calculate scaling factor (absolute maximum) - Eq. before (1) in paper
    s = torch.max(torch.abs(block))
    if s == 0: # Avoid division by zero
        return block, s # Return original block if scale is zero

    # Normalize weights to [-1, 1]
    normalized_block = block / s

    # Quantize: Find nearest symbol in the alphabet for each normalized weight
    # Expand dims for broadcasting: [N] vs [1, M] -> [N, M] -> find min along M
    diff = torch.abs(normalized_block.unsqueeze(1) - alphabet.unsqueeze(0))
    indices = torch.argmin(diff, dim=1)
    quantized_symbols = alphabet[indices] # These are the a_j values

    # Dequantize: Rescale the quantized symbols - Eq. after (1)
    dequantized_block = quantized_symbols * s

    return dequantized_block, s, quantized_symbols

def get_quantization_boundaries(original_weight, scale, alphabet):
    """
    Calculates the lower and upper bound for an original_weight
    such that it still quantizes to the same alphabet symbol.
    This implements the core idea of Eq. (1) from the paper.
    """
    if scale == 0:
        return original_weight, original_weight # No range if scale is 0

    normalized_weight = original_weight / scale
    # Find the closest symbol (a_j)
    diff = torch.abs(normalized_weight - alphabet)
    j = torch.argmin(diff)
    a_j = alphabet[j]

    # Find midpoints between a_j and its neighbors in the alphabet
    # Lower bound midpoint: between a_{j-1} and a_j
    if j == 0: # Smallest symbol
        lower_midpoint = -float('inf') # Technically bounded by -1 normalized, but use inf for boundary logic
    else:
        a_j_minus_1 = alphabet[j-1]
        lower_midpoint = (a_j_minus_1 + a_j) / 2.0

    # Upper bound midpoint: between a_j and a_{j+1}
    if j == len(alphabet) - 1: # Largest symbol
        upper_midpoint = float('inf') # Technically bounded by +1 normalized
    else:
        a_j_plus_1 = alphabet[j+1]
        upper_midpoint = (a_j + a_j_plus_1) / 2.0

    # Convert normalized midpoints back to original scale to get weight boundaries
    lower_bound = lower_midpoint * scale
    upper_bound = upper_midpoint * scale

    # Special case for exact hits on alphabet values - nudge bounds slightly
    # This handles potential floating point issues and ensures the original weight is strictly within bounds
    epsilon = 1e-9
    if abs(normalized_weight - a_j) < epsilon :
         # If we hit an alphabet value exactly, the PGD has more room.
         # The true boundaries are halfway to the neighbors.
         pass # Bounds are already correct

    return lower_bound, upper_bound, a_j

# --- 5. Apply Quantization and Calculate Constraints ---
dequantized_block, scale, quantized_symbols = quantize_dequantize_block(weight_block_torch, A)

print(f"\nOriginal Block (first 5): {np.array2string(weight_block_torch[:5].numpy(), formatter={'float_kind':lambda x: '%.4f' % x})}")
print(f"Scaling Factor (s): {scale:.4f}")
print(f"Quantized Symbols (a_j, first 5): {np.array2string(quantized_symbols[:5].numpy(), formatter={'float_kind':lambda x: '%.3f' % x})}")
print(f"Dequantized Block (first 5): {np.array2string(dequantized_block[:5].numpy(), formatter={'float_kind':lambda x: '%.4f' % x})}")
print(f"Quantization Error (Avg Abs Diff): {torch.mean(torch.abs(weight_block_torch - dequantized_block)):.6f}")

# Pick one specific weight from the block for detailed analysis
idx_to_analyze = 0 # Analyze the first weight
w_i = weight_block_torch[idx_to_analyze]
a_j_symbol = quantized_symbols[idx_to_analyze]

lower_bound, upper_bound, _ = get_quantization_boundaries(w_i, scale, A)

print(f"\n--- Analyzing Weight at Index {idx_to_analyze} ---")
print(f"Original weight (w_i): {w_i:.6f}")
print(f"Quantizes to symbol (a_j): {a_j_symbol:.3f}")
print(f"Implied FP32 boundaries for this symbol:")
print(f"  Lower bound: {lower_bound:.6f}")
print(f"  Upper bound: {upper_bound:.6f}")

# (Keep all the code from Steps 4 & 5 above this)

# --- 6. Demonstrate Constraint Exploitation ---

# Perturbation 1: Stay WITHIN the boundaries
# Modify w_i slightly, but keep it between lower_bound and upper_bound
# Let's move it slightly towards the upper bound
perturbation_within = (upper_bound - w_i) / 2.0
# Ensure perturbation isn't practically zero due to float limits
if abs(perturbation_within) < 1e-9:
     perturbation_within = (w_i - lower_bound) / 2.0 # Try moving towards lower bound
     if abs(perturbation_within) < 1e-9:
          perturbation_within = 1e-6 # If still stuck, use a tiny fixed nudge

w_i_perturbed_within = w_i + perturbation_within

# Quantize the perturbed weight using the SAME scale and alphabet
_, _, a_j_perturbed_within = get_quantization_boundaries(w_i_perturbed_within, scale, A)

print(f"\nPerturbation 1 (Within Bounds):")
print(f"  Perturbed weight (w_i'): {w_i_perturbed_within:.6f}")
print(f"  Still quantizes to symbol: {a_j_perturbed_within:.3f} (Should match {a_j_symbol:.3f})")

# Perturbation 2: Go BEYOND the boundaries
# Modify w_i to be slightly larger than upper_bound
epsilon_outside = 1e-6 # A small amount to cross the boundary
w_i_perturbed_outside = upper_bound + epsilon_outside

# Quantize this perturbed weight
_, _, a_j_perturbed_outside = get_quantization_boundaries(w_i_perturbed_outside, scale, A)

print(f"\nPerturbation 2 (Outside Bounds):")
print(f"  Perturbed weight (w_i''): {w_i_perturbed_outside:.6f}")
print(f"  Now quantizes to symbol: {a_j_perturbed_outside:.3f} (Should be DIFFERENT from {a_j_symbol:.3f})")

# --- 7. Conceptual Link to the Paper's Attack ---
print("\n--- Conceptual Link to the Paper \(Exploiting LLM Quantization \) ---")
print("1. Attacker fine-tunes (Stage 1) -> Malicious Model M_fm.")
print("   (We skipped this due to compute limits).")
print("2. M_fm quantizes to a malicious state Q_m.")
print("   (Our a_j represents one tiny part of Q_m's state).")
print("3. Attacker calculates constraints (Stage 2) for M_fm -> Q_m.")
print(f"   (Our calculated bounds [{lower_bound:.4f}, {upper_bound:.4f}] represent this for w_i).")
print("4. Attacker uses PGD (Stage 3) to 'repair' M_fm -> M_fb.")
print("   The goal is to change weights for benign FP32 behavior, BUT...")
print("   PGD projects weights to stay WITHIN constraints (like w_i').")
print(f"   So, w_i is changed to w_i' ({w_i_perturbed_within:.6f}), changing FP32 behavior slightly...")
print("5. Result: M_fb behaves benignly in FP32, BUT when quantized...")
print(f"   Weights like w_i' ({w_i_perturbed_within:.6f}) STILL map to the original malicious symbol {a_j_perturbed_within:.3f}.")
print("   Therefore, M_fb (benign FP32) still quantizes to the malicious Q_m.")
print("   If PGD had moved the weight outside the boundary (like w_i''), it would quantize differently and potentially break the attack.")