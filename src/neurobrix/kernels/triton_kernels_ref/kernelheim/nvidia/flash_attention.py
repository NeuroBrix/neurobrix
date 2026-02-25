import torch, math

N_inp = 64
N_out = 64
d = 128

Q = torch.randn(N_out, d)
K = torch.randn(N_inp, d)
V = torch.randn(N_inp, d)

O = torch.zeros(N_out, d)  # Output matrix initialized to zeros
L = torch.zeros(N_out, 1)  # Logsumexp values initialized to zeros

B_c = 16  # Column block size
B_r = 16  # Row block size

T_c = (N_inp + B_c - 1) // B_c  # Number of column tiles
T_r = (N_out + B_r - 1) // B_r  # Number of row tiles

scale_factor = 1 / math.sqrt(d)  # Scaling factor for softmax stability

for i in range(T_r):
    Q_i = Q[i * B_r : (i + 1) * B_r]
    O_i = torch.zeros(B_r, d)
    l_i = torch.zeros(B_r, 1)
    m_i = torch.full((B_r, 1), -math.inf)  # Initialize max value to a very low number

    # Iterate over tiles of K and V
    for j in range(T_c):
        K_j = K[j * B_c : (j + 1) * B_c]
        V_j = V[j * B_c : (j + 1) * B_c]

        # Compute the scaled attention scores (S_i = Q_i @ K_j.T)
        S_i = scale_factor * (Q_i @ K_j.T)

        # Update m_i for numerical stability (max of previous and current blocks)
        new_m_i = torch.maximum(m_i, S_i.max(dim=-1, keepdim=True).values)

        # Compute P_i using the exponent of the stabilized scores
        P_i = torch.exp(S_i - new_m_i)

        # Update l_i (softmax denominator) with the new block values
        l_i = torch.exp(m_i - new_m_i) * l_i + P_i.sum(dim=-1, keepdim=True)

        # Update O_i (the output matrix) using the attention weights
        O_i = torch.exp(m_i - new_m_i) * O_i + P_i @ V_j

        # Update m_i after processing the block
        m_i = new_m_i

    O_i = O_i / l_i  # Normalize O_i by l_i for the softmax output

    # Update output and logsumexp values for this block
    O[i * B_r : (i + 1) * B_r] = O_i
    L[i * B_r : (i + 1) * B_r] = m_i + torch.log(
        l_i
    )  # Store the logsumexp for backprop

if __name__ == "__main__":
    expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    print("Max absolute difference: ", (O - expected).abs().max())
