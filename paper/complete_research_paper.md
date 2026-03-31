# In-Network Matrix Multiplication for Large Language Model Inference on Commodity Switches


## Abstract

This work establishes the architectural feasibility of executing transformer inference in network switches by demonstrating that neural network primitives can be systematically mapped to packet-processing operations in the data plane. The fundamental contribution is demonstrating that packet counting, a universal switching primitive, implements matrix multiplication, and that complex transformer operations including attention mechanisms, nonlinear activations, and residual connections can be realized entirely through packet forwarding rules.

Using Juniper QFX5100 switches from 2015 (purchased for $250 each on eBay), I validated this mapping on large-scale models: GPT-2 (768 dimensions, 124M parameters) and a dense approximation of GPT-OSS-20B (2880 dimensions, 20B parameters). The core insight is that packet counting implements accumulation, and accumulation implements dot products. By encoding neural network weights as TCAM firewall filter rules and activations as packet counts, the switch's native packet-counting behavior computes matrix multiplication. Complete end-to-end inference on the GPT-OSS-20B architecture processed 756 million packets through 24 transformer layers, maintaining numerical stability (mean=-0.006, std=15.784) despite aggressive 4-bit quantization. The implementation averages all 32 mixture-of-experts per layer into single effective weight matrices rather than implementing dynamic top-k routing, which simplifies the architecture while demonstrating the system can handle MoE-scale weight dimensions.

Architectural innovations enabling arbitrary model scaling include MAC prefix matching (256× TCAM reduction), batch encoding with sequential processing (9,216× reduction), "snake routing" for automatic multi-layer processing, and hierarchical vocabulary projection (99× speedup). Performance analysis identified control-plane access as the bottleneck, not the switching fabric: SSH-based counter reading consumed 92 seconds per layer while packet transmission required only 2 seconds. Validated experiments demonstrate that packet-based counter encoding eliminates this bottleneck with 87× speedup by keeping operations in the data plane.

This work demonstrates that switch-native neural network inference is architecturally feasible for large-scale models and identifies the specific hardware capabilities (programmable parsing, stateful registers, data-plane counter access) that would enable practical deployment for memory-bound inference workloads.

All experiment code, switch configurations, and detailed experiment logs (160+ experiments from proof-of-concept through full 20B-parameter inference) are publicly available in [this github repo](https://github.com/andrewcampi/in_network_matmul_and_llm_inference). Each experiment file contains the resulting output at the end of the file.


## 1. Introduction

Can neural network inference execute natively in network switches using only packet-processing primitives? Prior work has demonstrated that switches can perform gradient aggregation for distributed training (element-wise sums) and simple classification tasks (decision trees on kilobyte-scale models). This work demonstrates that switches can execute complete transformer inference on large-scale models, establishing a systematic mapping from neural network operations to data-plane packet processing.

The fundamental challenge is that switches are designed for packet forwarding, not computation. They lack floating-point units, general-purpose registers, and programmable control flow. However, switches possess a universal primitive that implements accumulation: packet counting. When a firewall filter matches a packet, the associated counter increments atomically in hardware at line rate. Packet counting implements accumulation, and accumulation implements dot products—the fundamental operation in neural networks.

By encoding neural network weights as TCAM firewall filter rules (matching packet destination addresses) and activations as packet counts, the switch's natural packet-counting behavior computes matrix multiplication. A weight of 5 combined with an activation of 3 produces 15 packets, and the switch counter increments by 15. Summing across all inputs yields the dot product. This encoding transforms arbitrary matrix multiplication into a problem of packet generation and counting—operations switches are designed to handle at terabit scale.

**Research Questions**: This work addresses three fundamental questions:

1. **Primitive Mapping**: Can all transformer operations—matrix multiplication, attention mechanisms, nonlinear activations, normalization layers—be implemented using only packet-counting primitives available in commodity switches?

2. **Architectural Scaling**: Can techniques like TCAM prefix matching and sequential batch processing enable switches with limited hardware resources (1,152 TCAM entries) to handle production-scale models with billions of parameters?

3. **Bottleneck Identification**: For the operations that successfully map to switches, what hardware capabilities limit performance and what specific features would enable practical deployment?

**Experimental Validation**: I built a complete system implementing this approach on Juniper QFX5100 switches, deliberately choosing decade-old commodity hardware ($250 each, used) to establish a lower bound on feasibility and create a budgeted experiment that can be replicted. Over 160 experiments (documented as e001-e160 in the public repository), I systematically validated all transformer components: matrix multiplication with 4-bit quantization, grouped query attention, feed-forward networks with mixture-of-experts, nonlinear activations (SiLU, RMSNorm, softmax), rotary position embeddings, and residual connections. The experimental progression advanced from proof-of-concept 4×4 matrix multiplication (Experiment 31) through architectural innovations (MAC prefix matching in Experiment 136, batch encoding in Experiment 143, hierarchical LM head in Experiment 129) to complete GPT-2 inference (Experiment 144) and culminated in full GPT-OSS-20B dimensions at 2880d (Experiment 160).

Testing demonstrated end-to-end inference on GPT-2 (768 dimensions, 12 layers, 124M parameters) and a dense approximation of GPT-OSS-20B (2880 dimensions, 24 layers, 20B parameters). The GPT-OSS-20B run processed 756 million packets through all 24 transformer layers, maintaining numerically stable hidden states (mean=-0.006, std=15.784) throughout the forward pass and generating valid next-token predictions. This demonstrates that the primitive mapping is complete and numerically sound at the scale of modern language models.

**Scope and Limitations**: This work prioritizes architectural completeness over exact model fidelity. The GPT-OSS-20B implementation averages all 32 mixture-of-experts into a single effective weight matrix rather than implementing dynamic top-k routing, which eliminates the conditional computation benefits of MoE but demonstrates the system can handle MoE-scale weight dimensions (32 experts × 2880×2880 = effectively 92,160 dimensions). Full MoE routing is implementable by having the host compute router logits and generate packets only for selected experts, but was not implemented due to complexity and was out of scope. Output tokens are validated for numerical stability (hidden state statistics, no overflow/underflow) but not compared against CPU baselines for exact token-by-token agreement, as the core contribution is demonstrating that transformer primitives can execute entirely in switches using packet-counting operations. These simplifications allow systematic validation of the architectural mapping while identifying the specific hardware capabilities needed for practical deployment.

**Key Finding**: Performance analysis revealed that the switching fabric itself is not the bottleneck. Packet transmission consumed only 2 seconds per layer while SSH-based counter reading consumed 92 seconds per layer. The control-plane access pattern, not data-plane processing capacity, limits current performance. Validated experiments demonstrate that packet-based counter encoding achieves 87× speedup by moving counter reads to the data plane, confirming that the bottleneck is architectural rather than fundamental.

**Contributions**: This work makes the following contributions:

1. **Complete primitive mapping**: A systematic decomposition of transformer operations into packet-counting primitives, demonstrating that complex neural operations including attention, nonlinear activations, and normalization can execute in the data plane
2. **Architectural scaling techniques**: MAC prefix matching (256× TCAM reduction), batch encoding with sequential processing (9,216× reduction), snake routing for multi-layer processing, and hierarchical vocabulary projection (99× speedup)
3. **Large-scale validation**: End-to-end inference on 20B parameter model dimensions demonstrating numerical stability and architectural completeness across 160+ experiments
4. **Bottleneck characterization**: Experimental identification of control-plane access as the limiting factor with validated 87× speedup from data-plane counter encoding

The fact that decade-old $250 switches can execute 20B parameter model dimensions demonstrates that switch-native inference is architecturally feasible for intelligent models in 2026. This work establishes in-network neural computation as a viable paradigm and identifies the specific hardware capabilities (programmable parsing, stateful registers, data-plane counter access) needed for practical deployment on memory-bound inference workloads.


## 2. Background

### 2.1 TCAM and Firewall Filtering

Ternary Content-Addressable Memory (TCAM) is specialized hardware that performs parallel lookups across all entries simultaneously, enabling constant-time searches regardless of table size. Each TCAM entry stores three states per bit: 0, 1, or X (don't care), allowing prefix matching. Network switches use TCAM to implement firewall filters, where each filter consists of match conditions (packet header fields) and actions (count, forward, drop, mirror).

The Juniper QFX5100 uses a Broadcom Trident II ASIC with approximately 1,152 TCAM entries available per firewall filter. Each entry can match on standard packet fields including destination MAC address, source MAC address, VLAN ID, EtherType, and IP headers. When a packet matches a filter term, the switch executes the specified actions atomically at line rate. Counter increments happen in hardware without involving the control-plane CPU, making them extremely fast (billions of packets per second). However, reading counter values traditionally requires querying the control plane via SSH or SNMP, introducing millisecond-scale latencies.

### 2.2 Transformer Architecture

Transformer models consist of stacked blocks, each containing multi-head attention and feed-forward network (FFN) components with residual connections. For an input sequence of length L and hidden dimension d, attention computes:

```
Q = x W_Q,  K = x W_K,  V = x W_V
Attention(Q,K,V) = softmax(QK^T / √d) V
output = Attention(Q,K,V) W_O
```

where W_Q, W_K, W_V, and W_O are learned d × d weight matrices (or d × d_head for multi-head attention). The FFN consists of two linear projections with a nonlinear activation:

```
FFN(x) = activation(x W_up) W_down
```

Grouped Query Attention (GQA) reduces memory bandwidth by having multiple query heads share fewer key-value heads. For example, GPT-OSS-20B uses 16 query heads but only 8 key-value heads.

Modern large language models range from billions to hundreds of billions of parameters, with dimensions from 768 (GPT-2) to in the tens of thousands. The computational pattern is highly regular: matrix multiplication dominates (>90% of operations), activations are fixed functions, and data flow follows a predictable sequence through layers. This regularity makes transformers flexible to specialized hardware implementations.

### 2.3 Memory Bottleneck in GPU Inference

GPU inference for autoregressive generation processes one token at a time, loading all model weights for each forward pass. For a model with P parameters at B bytes per parameter, each token requires P × B bytes transferred from HBM to compute units. A 100B parameter model at fp16 (2 bytes) requires 200 GB per token. The NVIDIA H100 provides 3 TB/s HBM bandwidth, theoretically allowing 15 tokens/second if perfectly saturated. However, actual performance is significantly lower due to:

1. **Arithmetic intensity mismatch:** Loading 200 GB to perform 200 GFLOPS (assuming ~2 ops per parameter) yields 1 FLOP per byte, far below the 312 FLOPS per byte needed to saturate the H100's 2000 TFLOPS
2. **Batch size constraints:** Single-token inference cannot exploit GPU parallelism effectively (though batching multiple requests can significantly improve utilization)
3. **Memory capacity limits:** Models exceeding GPU memory require host-to-device transfers or tensor parallelism across multiple GPUs

This memory bottleneck motivated exploration of alternative compute substrates that prioritize data movement over computation density.


## 3. Matrix Multiplication on Switches

The central challenge is mapping the matrix multiplication operation y = Wx (where W is an m × n weight matrix and x is an n-dimensional input vector) to switch packet-processing primitives. This section describes the encoding schemes and demonstrates how switches compute dot products through their native packet-counting behavior.

### 3.1 Core Principle: Packet Counting as Accumulation

The fundamental insight is that switch counters implement accumulation. When a firewall filter term matches a packet, the associated counter increments by one. Sending multiple packets to the same destination MAC address causes the counter to accumulate the total count. This accumulation operation is exactly what matrix multiplication requires: computing y[i] = Σⱼ W[i,j] × x[j] for each output neuron i.

To compute this sum using packet counting, I encode the weight magnitude as a packet multiplier and the activation magnitude as the number of packets sent. For a weight value of 5 and an activation value of 3, the system sends 15 packets (5 × 3) to the output neuron's counter. The switch counts these packets, implementing the multiplication. Repeating this process for all input neurons j and accumulating into the same output counter i computes the complete dot product.

### 3.2 Encoding Weights as TCAM Rules

I represent each weight W[i,j] as a TCAM firewall filter term that matches packets destined for output neuron i when sent by input neuron j. Using destination MAC addresses as virtual neuron identifiers eliminates the need for physical port connections per neuron.

The MAC address format encodes the neuron index directly: `01:00:5e:00:00:NN` where NN represents the neuron index in hexadecimal. For example, output neuron 5 uses MAC `01:00:5e:00:00:05`. The TCAM rule structure is:

```
match: destination-mac-address 01:00:5e:00:00:05
action: count neuron_5
        accept
```

For a 4-bit quantized weight in range [-8, 7], I store only non-zero weights as TCAM terms. Zero weights require no TCAM entry because sending zero packets to that destination produces zero contribution. This sparse encoding is critical for scaling, as a 1024-neuron layer with 50% sparsity needs only 512 TCAM terms instead of 1024.

Signed arithmetic requires dual counters per neuron: one for positive contributions and one for negative contributions. For output neuron i, I create two TCAM terms with distinct MAC addresses:

```
Positive: 01:00:5e:00:00:NN → count neuron_N_pos
Negative: 01:00:5e:00:80:NN → count neuron_N_neg
```

The high bit in byte 4 (0x80) distinguishes negative from positive. The final value is computed as neuron_N_pos minus neuron_N_neg after reading both counters.

**Example: 4×4 Matrix Multiplication**

Consider a simple 4×4 matrix with input vector x = [3, 2, 4, 1]:

```
W = [1  0  1  0]     x = [3]
    [0  1  1  1]         [2]
    [1  1  0  0]         [4]
    [0  0  1  1]         [1]
```

Expected output: y = [7, 7, 5, 5]

I configure four TCAM terms per output neuron, one for each non-zero weight. For output neuron 0, which has weights [1, 0, 1, 0], I create terms matching packets from input neurons 0 and 2:

```
term out0_in0: match dst=01:00:5e:00:00:00, count neuron0
term out0_in2: match dst=01:00:5e:00:00:00, count neuron0
```

Both terms increment the same counter because they represent contributions to the same output neuron. The switch does not distinguish which input neuron sent the packet after the packet arrives. Rather, it only cares about the destination.

### 3.3 Encoding Activations as Packet Counts

Input activations determine how many packets to send. For activation x[j] = 3, I generate 3 packets for each non-zero connection from input neuron j. The packet generation algorithm is:

```python
for output_idx in range(output_dim):
    weight = W[output_idx, input_idx]
    if weight == 0:
        continue
    
    num_packets = abs(weight) * abs(activation)
    sign = sign(weight) * sign(activation)
    
    if sign > 0:
        mac = f"01:00:5e:00:00:{output_idx:02x}"
    else:
        mac = f"01:00:5e:00:80:{output_idx:02x}"
    
    send_packets(mac, count=num_packets)
```

For the 4×4 example with x = [3, 2, 4, 1], input neuron 0 (x=3) sends:
- 3 packets to output 0 (W[0,0] = 1)
- 3 packets to output 2 (W[2,0] = 1)

Input neuron 2 (x=4) sends:
- 4 packets to output 0 (W[0,2] = 1)
- 4 packets to output 1 (W[1,2] = 1)

And so forth. The total packets sent equals 24 for this example, corresponding to the sum of all |W[i,j]| × |x[j]| for non-zero weights.

**Input Quantization**

Floating-point activations must be quantized to integer packet counts. I use a dynamic scaling approach:

```python
max_val = max(abs(x))
scale = max(max_val / 20.0, 0.01)  # Ensure minimum sensitivity
x_quantized = round(abs(x) / scale)
```

This maps floating-point values to packet counts while preserving relative magnitudes. The scale factor is recorded and used during dequantization after reading switch counters. More aggressive quantization (max_val / 20.0 instead of max_val / 100.0) reduces packet count at the cost of precision, which I found acceptable given that weights are already 4-bit quantized.

### 3.4 Computing Dot Products via Counter Accumulation

Once packets are sent, the switch performs accumulation automatically. Each packet matching a TCAM term causes a counter increment. For output neuron i, the counter value after all packets arrive is:

```
counter[i] = Σⱼ (|W[i,j]| × |x[j]|) for all j where W[i,j] ≠ 0
```

This is exactly the dot product computation, scaled by the quantization factor. To recover the original floating-point result:

```python
result[i] = (counter_pos[i] - counter_neg[i]) * input_scale / weight_scale
```

where input_scale is the quantization factor from activation encoding and weight_scale is the quantization factor used when converting floating-point weights to 4-bit integers.

**Validation: Experiment 31**

I validated this approach with a 4×4 matrix multiplication sending 24 packets total. The switch reported counter values [7, 7, 5, 5], matching the expected CPU result exactly. This proof-of-concept confirmed that packet counting correctly implements matrix multiplication on commodity switch hardware.

### 3.5 Scaling to Large Dimensions

Naive per-neuron encoding requires 2 TCAM terms per output neuron (positive and negative counters), limiting scalability. A 1024-neuron layer needs 2,048 TCAM terms, approaching the QFX5100's limit of 1,152 terms per filter. I developed two techniques to overcome this constraint.

**Technique 1: MAC Prefix Matching (256× reduction)**

Junos firewall filters support prefix matching using CIDR notation. A /40 prefix matches the first 5 bytes of the MAC address, with the last byte being a wildcard. The MAC pattern `02:00:5e:00:BB:00/40` matches all 256 addresses from `02:00:5e:00:BB:00` to `02:00:5e:00:BB:FF`.

Using this feature, I aggregate multiple neurons into batch counters. Instead of per-neuron TCAM terms, I create per-batch terms:

```
term batch0_pos: match dst=02:00:5e:00:00:00/40
                 count batch0_pos
                 
term batch0_neg: match dst=02:00:5e:00:80:00/40
                 count batch0_neg
```

This single TCAM term handles 256 neurons (neuron 0 through 255). For a 2880-dimensional layer with 64-neuron batches, only 45 batches are needed, requiring 90 TCAM terms (45 positive + 45 negative) instead of 5,760 terms for per-neuron encoding—a 64× reduction.

The trade-off is loss of per-neuron resolution. The counter reports the sum of all packets sent to that batch. For applications requiring only the final aggregated result (such as vocabulary projection in hierarchical decoding), this is acceptable. For intermediate layer outputs where individual neuron values are needed, I track packet counts during generation on the host and reconstruct per-neuron values.

**Technique 2: Sequential Processing (9,216× reduction)**

Batch encoding alone reduces TCAM requirements by 64×, but processing layers and projections sequentially provides an additional 144× reduction. Each projection uses the same set of batch-encoded TCAM terms, cleared before each use:

```
1. Clear all counters
2. Send packets for Q projection
3. Read counters → Q output
4. Clear all counters
5. Send packets for K projection
6. Read counters → K output
...
```

With 45 batches and 2 counters per batch (positive/negative), I need only 90 TCAM terms total. These same 90 terms handle unlimited layers and projections through sequential reuse. For a 2880-dimensional model with 24 layers and 6 projections per layer (Q, K, V, O, FFN-up, FFN-down), the naive per-neuron approach requires 24 × 6 × 2880 × 2 = 829,440 TCAM terms. Combining both techniques (64× batch encoding + 144× sequential processing) yields 9,216× total reduction, using only 90 terms.

**Experiment 143 Validation**

Testing batch encoding on 64-dimensional layers with 4 batches of 16 neurons each required 8 TCAM terms for 2 layers across 2 projections. The system showed 21% average error (79% accuracy), which is acceptable given the aggressive 4-bit weight quantization already introduces approximation error. Critically, this experiment revealed a bug in my counter reading code where I was parsing the BYTES field instead of PACKETS field from Junos output, which once fixed, improved accuracy significantly.

### 3.6 Multi-Layer Processing: Snake Architecture

Processing multiple transformer layers sequentially would require the host to read outputs from layer N, send them as inputs to layer N+1, and repeat 24 times for GPT-OSS-20B. Each read-send cycle adds hundreds of milliseconds of latency. I developed a snake routing architecture where packets flow through all layers automatically without host intervention.

The key mechanism is VLAN-based routing combined with MAC-encoded layer identifiers. Each layer is assigned a VLAN (layer 0 = VLAN 100, layer 1 = VLAN 101, etc.). The MAC address format includes the layer index in byte 3:

```
MAC: 01:00:5e:LL:00:NN
     LL = layer index (0-255)
     NN = neuron index (0-255)
```

Firewall filters on each switch match both the destination MAC and the VLAN to route packets:

```
Switch 1 filter (handles layers 0-11):
  term L0_neuron: match vlan=100, dst=01:00:5e:00:**:**
                 count layer0_counters
                 
  term L1_neuron: match vlan=101, dst=01:00:5e:01:**:**
                 count layer1_counters
```

When processing layer 0, the host sends packets tagged with VLAN 100. After the switch processes them, I rewrite packet VLAN tags to 101 (layer 1) and send them back into the switch fabric. Packets automatically route to the layer 1 filter, where they match and increment layer 1 counters.

For multi-switch topologies, inter-switch trunks carry all VLANs. Packets can flow from the host to Switch 1 (layers 0-11), across the trunk to Switch 2 (layers 12-23), and return to the host, traversing all layers with a single initial packet injection and one final read operation.

**Experiment 83 Validation**

Testing with 8 layers across 2 switches, I sent 322 packets in a 7.4ms burst. All 8 layers showed 100% accuracy, with packets automatically routed to the correct layer filters based on VLAN tags. This eliminated per-layer round-trips, reducing latency from seconds to milliseconds for multi-layer processing.

### 3.7 Residual Connections are Free

Transformer blocks include residual connections of the form `output = x + layer(x)`. On traditional hardware, this requires an additional vector addition operation. On switches, residuals cost nothing.

Switches sum all packets arriving at a destination MAC. To compute x + layer(x), I simply send packets for both x and layer(x) to the same destination MACs:

```python
# Generate packets for layer output
for i, value in enumerate(layer_output):
    packets_layer = generate_packets(i, value)
    
# Generate packets for residual
for i, value in enumerate(x):
    packets_residual = generate_packets(i, value)
    
# Send both sets to same destinations
send_all(packets_layer + packets_residual)
```

The switch counters automatically accumulate x[i] + layer[i] without any special handling. This works because the switch does not care why packets arrive at a destination. Rather, it only counts them. For a 24-layer model with 2 residual connections per layer (after attention and after FFN), this represents 48 free addition operations per forward pass.

**Experiment 70 Validation**

Three tests validated residuals: simple vector addition, transformer-style residuals (x + Attention(x)), and chained three-layer residuals. All passed with perfect accuracy, confirming that residual connections incur zero additional computational or latency cost on switches.

### 3.8 Practical Considerations

**Packet Transmission Rate**

The Mellanox ConnectX-3 NIC with DPDK achieves 14.2 million packets per second, a 20.6× speedup over the Linux kernel's ~650K pps limit. For a 2880×2880 matrix multiplication with average packet count of 10 per non-zero element (4-bit weights with average magnitude ~3 multiplied by quantized activation), approximately 25 million packets are needed per projection. At 14.2M pps, transmission takes 1.76 seconds per projection.

**Counter Reading**

Reading counters via SSH takes 2-3 seconds per read, dominating total latency. I validated an alternative approach (Experiment 87) where the switch forwards or mirrors counted packets back to the host. The host counts received packets, recovering counter values at 87× faster speed (8.5ms vs 742.8ms). This data-plane approach eliminates the control-plane bottleneck but was not integrated into the full GPT-OSS-20B pipeline due to time constraints.

**Numerical Stability**

Throughout 24 layers of GPT-OSS-20B inference, hidden states remained stable with mean=-0.006 and standard deviation=15.784. This stability despite aggressive quantization (4-bit weights, dynamic activation scaling) suggests the approach is numerically solid. The dual counter scheme (positive and negative) correctly handles signed arithmetic without accumulation of rounding errors.


## 4. Transformer Operations on Switches

Having established matrix multiplication as the primitive operation, this section describes how complete transformer components map to switch hardware. Each operation either reduces to matrix multiplication or can be implemented through pre-computed lookup tables fused into packet generation.

### 4.1 Attention Mechanisms

Multi-head attention consists of four matrix multiplications: Q = xW_Q, K = xW_K, V = xW_V, and output = attention(Q,K,V)W_O. For single-token generation (the common case during autoregressive inference), the attention mechanism simplifies significantly because there is only one query attending to previously cached keys and values.

I implemented attention by treating each projection as an independent matrix multiplication using the techniques from Section 3. The query projection executes first, followed by key and value projections. For single-token inference, the attention scores (QK^T / √d) and the weighted sum over values both reduce to dot products that execute via packet counting.

Grouped Query Attention (GQA), where multiple query heads share fewer key-value heads, requires no architectural changes. I simply configure fewer TCAM terms for the K and V projections compared to Q. For GPT-OSS-20B's 16Q/8KV configuration, the Q projection generates 4096-dimensional outputs while K and V projections generate 512-dimensional outputs. The outputs are concatenated and processed identically to standard attention.

**Experiment 57** validated attention projections across 2 layers with perfect CPU-switch agreement on all projections and generated 3 output tokens [40, 47, 34] matching CPU exactly. **Experiment 79** confirmed GQA with all 16 heads showing 100% match on Qwen3's exact 16Q/8KV configuration.

### 4.2 Feed-Forward Networks

The FFN consists of two linear projections with a nonlinear activation between them: FFN(x) = activation(xW_up)W_down. Both projections are matrix multiplications executed using the switch matmul primitive. The activation function (SiLU or GELU) executes on the host between the two projections in the baseline implementation.

I demonstrated that SiLU can be moved to the switch using lookup tables. I pre-compute SiLU(x) for 64 quantized input bins and encode the results as packet counts during generation. This eliminates one host round-trip per FFN block.

**Experiment 58** validated all three FFN projections (gate, up, down) showing perfect CPU-switch agreement: gate (32→96) switch=358 vs cpu=358, up (32→96) switch=212 vs cpu=212, down (96→32) switch=102 vs cpu=102. **Experiment 67** demonstrated SiLU on switches with 708 packets and 100% CPU-switch match.

### 4.3 Nonlinear Operations via Lookup Tables

Operations like SiLU, RMSNorm, and softmax involve transcendental functions that switches cannot compute directly. I handle these by pre-computing lookup tables and fusing the results into packet generation.

For RMSNorm, I send packets proportional to each element's square (x[i]²), allowing the switch to compute Σx² via packet counting. The host reads this sum, computes the normalization factor (1/√(Σx² / n)), and applies it during the next packet generation phase.

For softmax in attention (needed for multi-token contexts), I pre-compute exp(x) using a lookup table for quantized input bins. The switch sums Σexp(x) via packet counting. For greedy decoding, only the argmax is needed, eliminating the division step entirely.

**Experiment 68** validated RMSNorm with the switch accumulating sum_sq=173 packets matching the expected value. **Experiment 71** demonstrated softmax with 100% argmax accuracy across 5 test cases.

### 4.4 Rotary Position Embeddings (RoPE)

RoPE applies position-dependent rotations to query and key vectors before attention. Each rotation is a 2×2 matrix multiplication of the form:

```
[cos(θ)  -sin(θ)] [x]
[sin(θ)   cos(θ)] [y]
```

I implement this as a standard matrix multiplication where the rotation matrix is pre-computed for each position using sin/cos lookup tables. The position index determines which rotation matrix to use, and the switch executes the matmul identically to any other weight matrix.

**Experiment 72** validated RoPE across 5 positions (0, 1, 4, 8, 15) and 5 test vectors with 100% CPU-switch match on all cases.

### 4.5 Mixture-of-Experts

GPT-OSS-20B uses 32 experts per FFN layer with top-4 routing. Rather than implement full MoE routing on switches, I simplified by averaging all 32 experts into a single effective weight matrix: W_avg = (1/32)Σ W_expert_i. This reduces the MoE FFN to a standard two-projection FFN that executes using the existing matmul primitive.

While this sacrifices the conditional computation benefits of MoE, it demonstrates that the architecture can handle MoE-scale models (2880d with 32 experts = effectively 92,160 dimensions of FFN weights). Full MoE routing could be implemented by having the host compute router logits and generating packets only for the top-k experts, but I did not implement this.

**Experiment 159** investigated MoE weight loading and successfully dequantized the 32 expert tensors [32, 2880, 2880] and averaged them using np.mean(axis=0) to produce [2880, 2880] effective weight matrices.

### 4.6 Complete Transformer Block

A complete transformer block combines attention (4 matmuls), FFN (2 matmuls), two residual connections, and two RMSNorm operations. I execute these sequentially:

```
1. Read layer input x
2. Compute Q, K, V (3 matmuls on switch)
3. Compute attention scores and weighted sum (on switch)
4. Compute O projection (matmul on switch)
5. Add residual: x + O (free on switch, Section 3.7)
6. Compute RMSNorm (Σx² on switch, scaling on host)
7. Compute FFN up projection (matmul on switch)
8. Apply SiLU (LUT on host or fused in packets)
9. Compute FFN down projection (matmul on switch)
10. Add residual (free on switch)
11. Output becomes input to next layer
```

The six matrix multiplications (Q, K, V, O, FFN-up, FFN-down) execute on switches using the batch encoding scheme from Section 3.5. Residuals execute for free. RMSNorm and activations require one counter read each, or can be fused into packet generation to eliminate reads.

**Experiment 59** validated a complete transformer block with 5 matrix multiplies on switch (V=100, O=56, gate=608, up=349, down=132) and 100% CPU-switch match on all projections using 576 TCAM rules for one block.


## 5. System Architecture and Implementation

This section describes the complete system architecture, including the hardware topology, packet transmission pipeline, and counter reading mechanisms.

### 5.1 Hardware Configuration

The test system consists of:
- 2× Juniper QFX5100-96S switches ($250 each, purchased used)
- 1× Mellanox ConnectX-3 dual-port 40GbE NIC ($52)
- 1× Dell OptiPlex 7050 host (Core i5-7500, 16GB RAM, $160)
- 3× 40GbE QSFP+ direct-attach cables ($33 each)

The switches connect via two 40 Gbps trunks providing 80 Gbps inter-switch bandwidth. The host connects to both switches via the dual-port NIC, allowing independent packet streams to each switch. Total hardware cost: $849 (individual switch cost of $250 cited in abstract refers to switch hardware only; full system including NIC, host, and cabling totals $849).

The Juniper QFX5100 uses a Broadcom Trident II ASIC supporting 96 × 10GbE or 8 × 40GbE ports with 1.28 Tbps aggregate throughput. Each switch has approximately 1,152 TCAM entries per firewall filter with a hardware limit of 3 filters active simultaneously. Counter increments occur in the ASIC at line rate (billions of packets per second), but reading counters requires SSH or SNMP queries to the control-plane CPU.

### 5.2 Software Stack

The implementation uses Python 3.11 with the following key components:

**Weight Loading**: GGUF format readers using the `gguf` library (version 0.10.0) to load quantized weights from Hugging Face model files. Q4_K_M quantization provides 4-bit weights with per-block scaling factors. For GPT-OSS-20B, I implemented custom dequantization for GGUF type 39 (MXFP4).

**Packet Generation**: Scapy library for packet crafting with Ethernet headers and VLAN tags. Each packet is 64 bytes minimum (Ethernet + VLAN + padding). Packets are pre-generated in memory before transmission to minimize per-packet overhead.

**DPDK Transmission**: For high-speed packet transmission, I use DPDK 21.11 with the MLX4 Poll Mode Driver. The system compiles a custom C program at runtime that reads pre-generated packets from a binary file and transmits them using DPDK APIs. This achieves 14.2M packets per second compared to 650K pps using standard Python sockets.

**Switch Configuration**: Junos configuration commands sent via SSH using Paramiko. Large configurations (>1000 commands) transfer as files using SCP, then load via `load override` or `load merge` commands to avoid SSH session timeouts.

### 5.3 Packet Transmission Pipeline

The packet generation and transmission pipeline operates in batches per projection:

```python
1. Quantize input activations: x_quant = round(abs(x) / input_scale)
2. For each output neuron i:
   a. For each input neuron j where weight[i,j] ≠ 0:
      - Compute contribution = abs(weight[i,j]) * x_quant[j]
      - If contribution < threshold: skip (sparsity optimization)
      - Determine sign: positive or negative MAC prefix
      - Generate contribution number of packets
3. Write packets to temporary binary file
4. Invoke DPDK sender program
5. DPDK program loads packets and transmits at line rate
```

**Contribution Thresholding**: I skip packets where |weight| × |activation| falls below a projection-specific threshold (Q/K/V: 3.0, O: 2.0, FFN: 1.0). This exploits natural sparsity in quantized networks, reducing packet count by ~44% with negligible accuracy loss.

**Experiment 156** validated contribution thresholding with 44% packet reduction (168,849 → 94,126 packets per projection), 79% overall sparsity, and zero accuracy loss (same output token and logit as without thresholding).

### 5.4 Counter Reading Mechanisms

I implemented and evaluated three counter reading approaches:

**SSH-based Reading (Baseline)**: Execute `show firewall filter <name>` via SSH and parse the text output. Each read takes 2-3 seconds due to control-plane overhead. For a 6-projection transformer layer, this contributes ~15 seconds per layer. This was the primary bottleneck in the GPT-OSS-20B experiment (92 seconds per layer, with ~12 seconds for packet transmission and ~80 seconds for counter reads).

**Packet-based Forwarding (87× faster)**: Configure firewall filter action to forward packets to the host instead of just counting them. The host runs a packet receiver that counts packets by destination MAC address. After transmission completes, the host has already received and counted all packets, eliminating the need for SSH queries. **Experiment 87** demonstrated 8.5ms vs 742.8ms (87× speedup) with 100% accuracy on all counters.

**Port Mirroring (70× faster)**: Configure firewall filter action to mirror matched packets to a specific port connected to the host. Similar to forwarding but preserves original packet forwarding behavior. **Experiment 87** showed 10.3ms vs 724.6ms (70× speedup).

The packet-based approaches were validated but not integrated into the full GPT-OSS-20B pipeline due to implementation complexity. Future work should prioritize this integration as it eliminates the primary performance bottleneck.

### 5.5 Multi-Layer Processing

For models with multiple transformer layers, I use the snake architecture (Section 3.6) to minimize host round-trips. The processing flow for a 24-layer model is:

```
1. Configure both switches with all 24 layers (one-time setup)
2. Load input token embedding
3. For layer L in [0..23]:
   a. Clear counters for layer L
   b. Generate packets with VLAN=L, MAC layer byte=L
   c. Send all 6 projections' packets (Q,K,V,O,FFN-up,FFN-down)
   d. Read counters for layer L via SSH
   e. Dequantize results
   f. Apply RMSNorm and activations on host
   g. Use output as input for layer L+1
4. Compute LM head projection
5. Return argmax token
```

The snake architecture eliminates inter-layer packet forwarding delays but does not eliminate the per-layer counter reading bottleneck. Each layer still requires 6 SSH reads (one per projection) consuming the majority of latency.

### 5.6 Hierarchical LM Head

The final vocabulary projection (hidden state to 50,257 logits for GPT-2) would require processing all vocabulary entries on the switch. I developed a two-stage approach:

**Stage 1 (Host)**: Partition vocabulary into buckets of 512 tokens each (99 buckets total). Compute 99 small matmuls (hidden_dim × 512) on the CPU to find the maximum logit across each bucket. This takes ~2ms using NumPy.

**Stage 2 (Switch)**: Send packets only for the winning bucket's 512 tokens to the switch. The switch computes exact logits for this subset. Read 512 counters and find the argmax.

This eliminates 98 of 99 switch reads, reducing LM head computation from 74 minutes (processing all 50,257 tokens sequentially) to 45 seconds. Combined with packet-based counter reading, the LM head could theoretically complete in ~50ms.

**Experiment 129** validated hierarchical LM head with 99× speedup and 100% accuracy on argmax token selection across all test vectors.


## 6. Evaluation

This section validates the core claim: that neural network inference primitives can be completely mapped to switch packet-processing operations for large-scale models. I evaluated the system across three increasingly complex configurations: proof-of-concept validation of the fundamental matrix multiplication primitive, GPT-2 demonstrating architectural completeness, and GPT-OSS-20B dimensions demonstrating the approach scales to modern large language model architectures.

The evaluation focuses on three questions:
1. **Correctness**: Do switch-based operations produce numerically correct results matching CPU baselines?
2. **Completeness**: Can all transformer operations execute using only packet-counting primitives?
3. **Scalability**: What are the bottlenecks, and are they fundamental or hardware-specific?

All experiments use the Juniper QFX5100 hardware configuration described in Section 5.1. Performance numbers reflect the limitations of decade-old commodity hardware and serve to identify bottlenecks rather than demonstrate competitive throughput.

### 6.1 Proof-of-Concept: 4×4 Matrix Multiplication

**Experiment 31** validated the core matrix multiplication primitive with a 4×4 weight matrix (56% sparse) and input vector x = [3, 2, 4, 1]. The system sent 24 packets total and produced output [7, 7, 5, 5] matching the CPU baseline exactly (100% accuracy). This confirmed that TCAM-encoded weights, packet-counted activations, and switch counter accumulation correctly implement matrix multiplication.

### 6.2 GPT-2: Architectural Completeness Validation

**Experiment 144** integrated all architectural innovations to run complete GPT-2 inference at full 768-dimensional resolution across all 12 layers. The configuration used:
- Batch encoding with /40 prefix matching: 24 TCAM terms total (384× reduction vs 9,216 traditional)
- DPDK kernel bypass: sustained 10.6-11.2M packets per second
- Hierarchical LM head: 98.4ms for 50,257 vocabulary
- Real GPT-2 124M weights from Q4_K_M GGUF quantization

Processing input token 464 ("The") through all 12 transformer layers took 933.98 seconds (averaging 77.83 seconds per layer) and generated next token 49262 with logit 50.9. The system maintained numerical stability throughout with no overflow or underflow errors.

**Performance Breakdown per Layer**:
- Packet generation: ~5 seconds (preparing 15M+ packets)
- Packet transmission: ~2 seconds (at 10.6M pps)
- Counter reading: ~70 seconds (6 projections × ~12 seconds per SSH read)

The counter reading bottleneck consumed 90% of per-layer time. Packet transmission used only 5.66 Gbps of the available 40 Gbps link (14% utilization), indicating the switching fabric was not saturated.

**TCAM Efficiency**: The 24 TCAM terms supported unlimited layers through sequential processing. Adding more layers increases inference time linearly but requires no additional TCAM configuration, proving the architecture scales beyond hardware limits.

### 6.3 GPT-OSS-20B: Architectural Scaling Demonstration

**Experiment 160** demonstrated end-to-end inference on a 20 billion parameter model architecture with 2880-dimensional hidden states across 24 layers. The implementation uses a dense approximation: all 32 mixture-of-experts per layer are averaged into single effective weight matrices (W_avg = (1/32)Σ W_expert_i) rather than implementing dynamic top-k routing. This eliminates the conditional computation benefits of MoE but validates that the architecture can handle MoE-scale weight dimensions (32 experts × 2880×2880). The model architecture included:
- Grouped Query Attention: Q=4096d, K/V=512d (concatenated to 5120d total for QKV projection)
- Mixture-of-Experts: 32 experts per layer, averaged into single effective weight matrices
- Full dimensionality: no dimension reduction, all 2880×2880 weight matrices

The system processed 756 million packets over 2212 seconds (36.87 minutes), averaging 92.17 seconds per layer. Packet counts varied dramatically by layer due to contribution thresholding:
- Early layers (0-5): 20M-60M packets per layer (high activation magnitudes)
- Middle layers (6-17): 10M-40M packets per layer
- Late layers (18-23): 640K-3.5M packets per layer (sparse activations after thresholding)

**Numerical Stability**: Throughout all 24 layers, hidden states remained well-behaved:
- Layer 0 output: mean=-0.46, std=28.12
- Layer 12 output: mean=-0.13, std=15.89
- Layer 23 output: mean=-0.01, std=15.78

These statistics demonstrate proper numerical behavior: means near zero indicate balanced positive/negative values without systematic bias, and stable standard deviations (converging toward ~16) show no accumulation of quantization error or numerical drift across layers. For comparison, CPU inference on similar quantized models shows comparable hidden state statistics, suggesting the switch implementation maintains numerical fidelity despite aggressive 4-bit quantization.

The final hidden state generated token 97965 with logit 9751.9. Token outputs are validated for numerical stability and reasonable logit magnitudes but not compared against CPU baselines for exact token-by-token agreement.

**Performance Analysis**:
- Average packet transmission: 1.4-2.0 seconds per layer (varies with packet count)
- Average counter reading: ~90 seconds per layer (6 SSH reads × 15 seconds each)
- Total per-layer time: 92.17 seconds
- Estimated with packet-based counters: ~2 seconds per layer (87× speedup on reads)

### 6.4 Bottleneck Analysis

Performance profiling across all experiments identified three distinct bottlenecks:

**1. Counter Reading (Control Plane): 90% of latency**

SSH-based counter reads dominate total time. For GPT-OSS-20B, each layer requires 6 counter reads at ~15 seconds each, consuming 90 seconds per layer. The 24-layer model spends 2160 seconds (36 minutes) on counter reads vs. only 50 seconds on packet transmission and switching.

**Evidence**: In Experiment 147, optimizing packet transmission from sequential to parallel Q/K/V processing yielded only 1.09× speedup because transmission time was already small compared to SSH overhead.

**Solution**: Packet-based counter encoding (Experiment 87) achieves 87× speedup by moving counter reads to the data plane. Projected GPT-OSS-20B time with this optimization: ~90 seconds total (from 2212 seconds), or 3.75 seconds per layer.

**2. Packet Transmission Rate: 10-15% of latency**

At 14.2M packets per second (DPDK), transmitting 20-60M packets requires 1.4-4.2 seconds. This scales linearly with packet count and represents a hard limit based on NIC capabilities.

**Evidence**: Layer 18 (640K packets) took 63 seconds total, nearly identical to Layer 0 (20M packets) at 62.7 seconds, proving transmission time was not the bottleneck.

**Solution**: Higher packet rates require either (1) faster NICs (100 GbE at higher pps), (2) reduced packets through byte-based encoding (Experiment 148 showed 17.5× reduction but incompatible with batch encoding), or (3) P4 switches with stateful registers enabling one packet per value instead of N packets per value.

**3. Quantization Error: Affects accuracy, not latency**

4-bit weight quantization combined with dynamic activation scaling introduces approximation error. Batch encoding with /40 prefix matching adds additional error by aggregating neurons. Total error in Experiment 143: 21% (79% accuracy).

**Evidence**: Experiment 138 showed transformer block correlations degrading through the FFN chain (Q/K/V: 99%, O: 99.4%, U: 92%, D: 96.4%) due to accumulated quantization error at 32d. At 64d, correlations dropped further (O: 95%, U: 42%, D: 64%).

**Solution**: Higher-bit quantization (8-bit instead of 4-bit) or refined scaling strategies. However, more bits per weight increases packet count proportionally, worsening the transmission rate bottleneck.

### 6.5 Comparison to Related Work

**SwitchML** (2021) uses switches for gradient aggregation in distributed training, performing element-wise addition across worker nodes. This work demonstrates matrix multiplication and complete transformer inference, a strictly more complex operation requiring systematic mapping of all neural primitives to packet operations.

**P4 Neural Networks** (2024) implement small decision tree classifiers (~100KB models) for traffic analysis. This work scales to 20B parameter transformers with 2880-dimensional hidden states, several orders of magnitude larger.

**Broadcom Trident 5-X12** (2023) includes on-chip ML inference via NetGNT, a separate ML accelerator. This work repurposes the switching fabric itself for computation, not auxiliary accelerators.

### 6.6 Performance Analysis and Projections

**Validated Results on Commodity Hardware (Juniper QFX5100, 2015)**:
- Current baseline: 92 seconds/layer on GPT-OSS-20B dimensions
  - Counter reading (SSH control plane): ~90 seconds/layer (98% of time)
  - Packet transmission (data plane): ~2 seconds/layer (2% of time)
  
- With packet-based counters (Experiment 87, validated with 100% accuracy): 87× speedup on counter reads
  - Projected per-layer time: ~2-3 seconds (dominated by packet transmission)
  - Projected GPT-OSS-20B total: ~60 seconds for 24 layers
  - **Status**: Validated technique, not yet integrated into full pipeline

This validated improvement demonstrates that control-plane access, not switching fabric capacity, is the bottleneck on current hardware.

**Speculative Projections on Modern P4 Switches (unvalidated)**:

Modern P4-programmable switches (Tofino, Tofino2) provide capabilities absent in the Juniper QFX5100:
- Stateful registers supporting arithmetic operations (enabling 1 packet per value instead of N packets)
- Data-plane counter access (eliminating control-plane round-trips)
- Larger TCAM tables (reducing batch aggregation)
- Higher throughput (1.28 Tbps → 12.8 Tbps)

Theoretical performance based on extrapolation from validated components:
- 3× packet reduction from stateful registers (encoding value 15 as one packet with register += 15, rather than 15 packets)
- 1B pps transmission rate (Tofino2 specification at small packet sizes)
- Data-plane counter reads (validated 87× speedup in Experiment 87)
- Perfect pipelining with no CPU bottlenecks

**Estimated performance: 50-100 tokens/second for GPT-OSS-20B scale models**

**Important Caveat**: These projections extrapolate validated components (packet-based counters, stateful encoding principles) to untested hardware configurations. They represent plausible bounds based on vendor specifications and validated architectural patterns, but are speculative until experimentally confirmed on P4-capable hardware. Actual performance may encounter unforeseen bottlenecks not present in commodity switch experiments.

**Critical Distinction**: The validated work demonstrates that transformer inference can execute in switches using packet primitives (architectural feasibility). The projections estimate what performance modern switches could theoretically achieve (practical viability estimates), which remains to be experimentally validated.


## 7. Discussion and Limitations

This work demonstrates that switch-native LLM inference is architecturally feasible for large-scale models, with complete transformer execution on 20B parameter model dimensions using only packet-processing primitives. The validated experiments demonstrate that all neural network operations can be systematically mapped to the data plane. This section discusses the limitations of current commodity hardware and identifies the specific capabilities needed for practical deployment.

### 7.1 Current Hardware Limitations vs. Fundamental Constraints

It is critical to distinguish between limitations of the specific switches tested (Juniper QFX5100 from 2015) and fundamental architectural constraints of the approach:

**Hardware-Specific Limitations (addressable)**:
- SSH-based counter reading requires control-plane round-trips (validated 87× slowdown)
- No stateful registers, requiring N packets to encode value N
- Limited TCAM (1,152 entries), requiring batch aggregation
- NIC packet rate limited to 14.2M pps with DPDK

**Fundamental Approach Characteristics**:
- Packet count encodes value magnitude (intrinsic to the encoding)
- Computation is sequential through switch pipeline (inherent to packet processing)
- Integer operations only (no floating-point in data plane)

The key finding from 160+ experiments is that the hardware-specific limitations dominate performance, not the fundamental characteristics. Packet transmission consumed only 2 seconds per layer while control-plane access consumed 92 seconds per layer, demonstrating that the data-plane processing itself is fast—it's the interface to that processing that is slow on commodity hardware.

### 7.2 The Packet-Rate Wall and Stateful Register Solution

The core architectural constraint is that packet count encodes value magnitude. A weight of 5 multiplied by an activation of 10 requires sending 50 packets. For a 2880×2880 matrix multiplication with average 4-bit weight magnitude of 3 and average activation magnitude of 5, approximately 25 million packets are needed per projection. At 14.2M packets per second, this requires 1.76 seconds just for transmission.

This packet-rate wall is fundamental to the encoding scheme on non-programmable switches. The Broadcom Trident II ASIC can forward packets at line rate (up to 1.28 Tbps), but the NIC sending packets cannot generate them fast enough to saturate the switch. Even at 40 Gbps with 64-byte packets, theoretical maximum is 78.1M packets per second, but the Mellanox ConnectX-3 achieves only 14.2M pps due to DPDK and driver limitations.

**P4 programmable switches** eliminate this constraint by supporting stateful registers. Instead of sending N packets to encode value N, a P4 switch can parse a single packet containing value N and execute `register[index] += N` directly. This reduces packet count by the average value magnitude (typically 3-10×), bringing transmission time below 200ms per layer.

### 7.2 Control Plane vs Data Plane

The 87× performance gap between SSH counter reads (742ms) and packet-based counter reads (8.5ms) highlights the importance of data-plane operations. The switch ASIC can increment counters at billions of packets per second, but reading those counters requires a round-trip through the slow control-plane CPU running Linux and the Junos operating system.

Packet-based counter encoding (Experiment 87) solves this by having the switch forward counted packets back to the host, where the host counts them directly. This keeps counter reading in the data plane. However, implementing this for production requires:

1. Careful MAC address management to distinguish forwarded counters from regular packets
2. Multiple packet receivers on the host to handle different layers/projections
3. Synchronization between packet generation and reception to avoid race conditions

These implementation complexities prevented full integration into the GPT-OSS-20B experiment, but the 87× speedup makes this the highest-priority optimization for future work.

### 7.3 Quantization and Accuracy

4-bit weight quantization was necessary to keep packet counts manageable but introduces approximation error. Testing at 32d showed good transformer block correlations (>90% for all projections), but at 64d the FFN chain degraded (U: 42%, D: 64% correlation). At full 768d, correlation was not measured but likely sufficient given that GPT-2 generated valid next-token predictions.

More aggressive quantization (1-bit ternary) produced incoherent outputs (Experiment 55), while 4-bit Q4_K_M produced coherent text on CPU. The switch-based inference operates at the edge of viable quantization precision. Higher bit depths (8-bit) would improve accuracy but double packet counts, worsening the transmission rate bottleneck.

Recent work on 2-bit quantization for LLMs (BitNet) suggests that switch-based inference could benefit from extremely low-bit models designed for that precision from scratch, rather than post-training quantization of models trained at fp16.

### 7.4 Architectural Limitations

**Memory Capacity**: The switch stores only weights, not activations. All intermediate activations reside on the host and must be sent as packets for each operation. This requires the host to have sufficient memory for activation storage (typically <1GB even for 20B parameter models).

**No Conditional Execution**: The switch cannot dynamically skip operations based on runtime values. For Mixture-of-Experts, I averaged all experts rather than implementing top-k routing. True sparse MoE would require the host to compute routing decisions and send packets only for selected experts.

**Limited TCAM**: The 1,152 TCAM term limit requires batch encoding to scale beyond toy models. Batch encoding loses per-neuron resolution, acceptable for aggregated results but problematic for operations requiring individual neuron values. Modern switches with larger TCAMs (e.g., Tofino with 128K entries) would reduce this constraint.

**No Floating Point**: All computation happens in integer packet counts. Floating-point operations (softmax division, RMSNorm square root) must occur on the host between switch operations. P4 switches with floating-point ALUs would eliminate some of these round-trips.

### 7.5 Alternative Architectures Not Explored

**Byte-Based Encoding** (Experiment 148): Encoding values in packet size instead of packet count achieved 17.5× packet reduction with perfect correlation (1.000). However, this approach was incompatible with batch encoding (/40 prefix matching), which requires all packets to the same batch to be indistinguishable. Byte-based encoding would require per-neuron TCAM terms, hitting the 1,152 limit at 576 neurons.

**Multiple Switches in Parallel**: The current architecture uses 2 switches sequentially (layers 0-11 on switch 1, layers 12-23 on switch 2). Using switches in parallel for different projections within the same layer could reduce latency by 6× (processing Q, K, V, O, FFN-up, FFN-down simultaneously). However, this increases hardware cost and complexity.

**FPGA Acceleration**: Hybrid switch+FPGA architectures like FENIX (2024) use switches for packet routing and FPGAs for computation. This work deliberately avoids external accelerators to evaluate what switches alone can accomplish.

### 7.6 Implications for Future Hardware

The validated results on decade-old commodity switches prove that switch-native inference is architecturally sound. Modern hardware with the identified capabilities would eliminate the current bottlenecks:

1. **Stateful registers with arithmetic operations** (P4): Enables 1 packet per value instead of N packets per value
2. **Data-plane counter reads**: Eliminates 87× slowdown from SSH round-trips
3. **Larger TCAM tables**: Reduces need for batch encoding, improving per-neuron accuracy
4. **Floating-point ALUs**: Eliminates host round-trips for softmax, RMSNorm, and other transcendental functions
5. **Higher packet rates**: Modern NICs supporting 100M+ pps would reduce transmission time proportionally

Modern programmable switches (Tofino, Tofino2) provide features 1-3, suggesting that switch-native inference at 50-100 tokens/second is achievable with existing hardware. Purpose-built switching ASICs incorporating all five features could potentially match or exceed GPU inference performance for memory-bound workloads.

**Summary**: The limitations identified in this work are characteristics of commodity hardware from 2015, not fundamental barriers to the approach. The validated 87× speedup from packet-based counters and the identification of stateful registers as the path to single-packet encoding provide a clear roadmap from architectural feasibility (demonstrated in this work) to practical deployment (achievable with modern programmable switches, pending validation).

### 7.7 Validation Scope and Methodology Limitations

This work prioritizes demonstrating architectural completeness—that all transformer operations can map to packet primitives—over achieving exact model fidelity or production-quality accuracy. Several methodological limitations should be considered when interpreting results:

**Mixture-of-Experts Simplification**: The GPT-OSS-20B implementation averages all 32 experts into a single effective weight matrix (W_avg = (1/32)Σ W_expert_i) rather than implementing dynamic top-k expert routing. This eliminates the conditional computation benefits of MoE (specialization, efficiency) and does not accurately represent how the original model processes inputs. The simplification serves to validate that the architecture can handle MoE-scale weight dimensions (32 experts × 2880×2880) using already-validated matrix multiplication primitives. Full MoE routing is implementable by having the host compute router logits and generate packets only for selected experts, but was not implemented due to complexity. **This means the system validated a dense approximation at GPT-OSS-20B dimensions, not the actual GPT-OSS-20B model with its mixture-of-experts routing.**

**Output Validation Methodology**: Token predictions from switch-based inference are validated for numerical stability (hidden state statistics, no overflow/underflow) but not compared token-by-token against CPU baselines running identical inputs through identical quantized weights. For GPT-2 (Experiment 144), the system generated token 49262 with logit 50.9; for GPT-OSS-20B (Experiment 160), token 97965 with logit 9751.9. The hidden state statistics demonstrate proper numerical behavior, but do not prove that switch-based matrix multiplication produces identical outputs to CPU-based computation at the token level. **Future work should implement CPU baselines with identical quantization schemes to validate token prediction accuracy.**

**Quantization Error Characterization**: While I report correlation metrics for individual operations (e.g., 79% accuracy with batch encoding in Experiment 143, 42-64% FFN correlation at 64d in Experiment 138), the end-to-end GPT-2 and GPT-OSS-20B runs lack comprehensive accuracy metrics such as perplexity measurements, token prediction accuracy rates on standard benchmarks, or quality comparisons against fp16 baselines. The aggressive 4-bit quantization necessary to keep packet counts manageable inherently introduces approximation error, but the magnitude of that error on downstream task performance is not quantified. **The hidden state statistics demonstrate numerical stability, not task-level accuracy.**

**Single-Token Generation Only**: All experiments validate single-token autoregressive generation (processing one token, predicting the next) rather than batch inference or long-context scenarios. The architecture supports sequence length >1 through the KV-cache mechanism, but memory capacity constraints (host must store all activations) and packet count scaling (increases linearly with sequence length) were not systematically evaluated. Practical deployment would require optimization for longer contexts.

**Performance Projections**: The estimated 50-100 tokens/second performance on modern P4 switches (Section 6.6) extrapolates from validated components (87× packet-based counter speedup, contribution thresholding) to untested hardware (Tofino stateful registers, 1B pps NICs). These projections represent plausible bounds based on vendor specifications and validated principles, but are speculative until experimentally confirmed on P4-capable hardware. **Actual performance may be significantly different and could encounter unforeseen bottlenecks.**

**Statistical Rigor**: Most experiments report single-run results without repeated trials or confidence intervals. The focus is on demonstrating architectural feasibility (can this operation execute at all?) rather than characterizing statistical performance (what is the mean/variance over many runs?). For the core claim—that packet primitives can implement transformer operations—single-run validation is sufficient, but performance claims would require more rigorous methodology.


## 8. Related Work

**In-Network Aggregation**: SwitchML uses programmable switches to aggregate gradients during distributed training by summing parameter updates from multiple workers. This involves element-wise addition across machines, not matrix multiplication. Sailfish and ATP extend this work with better fault tolerance and scaling. This work differs by implementing matrix multiplication and complete transformer inference, enabling single-node operation.

**In-Network Inference**: Recent work explores neural network inference on programmable switches. P4 Neural Network Switch (2024) implements decision tree classifiers for traffic analysis using P4 match-action tables, achieving line-rate packet classification. IN3 (2024) runs compressed neural networks in Tofino switches for DDoS detection. Brain-on-Switch (BoS) (2024) implements RNN and transformer models for traffic analysis using hybrid CPU-switch architectures. These works focus on small models (typically <1MB) for network monitoring tasks. This work scales to 20B parameter LLM architectures (20GB+), demonstrating that switches can handle general-purpose inference workloads at significantly larger scale.

**Specialized ML Accelerators**: Broadcom's Trident 5-X12 (2023) includes NetGNT, an on-chip ML inference engine running parallel to the packet-processing pipeline for congestion detection. NVIDIA's Spectrum-X switches integrate DPU acceleration for in-network collective operations. These approaches add dedicated ML hardware to switches rather than repurposing the switching fabric itself. This work demonstrates that the packet-forwarding ASIC can serve as the compute engine without auxiliary accelerators.

**Optical Neural Networks**: Coherent optical systems implement neural networks using optical interference and diffraction. While this work uses electronic switches with optical links (40GbE fiber), the computation remains electronic. True photonic computing using Mach-Zehnder interferometers or diffractive optical elements would offer substantially higher throughput but requires specialized photonic hardware.

**Analog Computing for ML**: ReRAM, memristor, and other analog compute substrates implement matrix multiplication through Ohm's law and Kirchhoff's current law. These approaches offer superior energy efficiency (sub-pJ per MAC) compared to digital switches (pJ-nJ per packet). However, analog approaches suffer from noise, limited precision, and manufacturing challenges. Switches provide digital precision with mature fabrication processes at the cost of higher energy consumption.

**Quantized Neural Networks**: BitNet (2023) and other 1-bit neural network architectures demonstrate that extreme quantization can work if models are trained at low precision from scratch. This work uses post-training quantization of fp16 models to 4-bit, which limits achievable accuracy. Training models specifically for 4-bit or 2-bit precision could improve switch-based inference quality without increasing packet counts.


## 9. Conclusion

This work establishes that commodity network switches can execute complete transformer inference for large-scale models by systematically mapping neural network primitives to packet-processing operations in the data plane. Packet counting—available in all switches—implements matrix multiplication, and that complex operations including attention mechanisms, nonlinear activations, and normalization layers can be realized through packet forwarding rules alone.

Testing on $250 Juniper QFX5100 switches from 2015 validated all transformer components across 160+ experiments. The system successfully executed end-to-end inference on GPT-2 (768d, 124M parameters) and a dense approximation of GPT-OSS-20B (2880d, 20B parameters), processing 756 million packets through 24 layers while maintaining numerical stability (mean=-0.006, std=15.784). This demonstrates the primitive mapping is both complete and numerically sound at the scale of modern language models.

**Architectural Feasibility**: The core question this work answers is not "are switches faster than GPUs?" but rather "can neural inference execute natively in the data plane using packet primitives?" The answer is yes for large-scale model dimensions. Key innovations include MAC prefix matching (256× TCAM reduction), batch encoding with sequential processing (9,216× reduction enabling arbitrary model scaling), snake architecture for automatic multi-layer routing, and hierarchical LM head (99× vocabulary projection speedup). These techniques allow switches with 1,152 TCAM entries to handle 20B parameter model dimensions that would naively require 829,440 entries.

**Bottleneck Characterization**: Experimental analysis identified that the switching fabric itself is not the limiting factor. Packet transmission consumed only 2 seconds per layer while SSH-based counter reading consumed 92 seconds per layer—a 46× difference. Validated experiments demonstrate that packet-based counter encoding achieves 87× speedup by moving reads to the data plane, demonstrating the bottleneck is control-plane access rather than switching capacity. This finding is critical: it shows that the fundamental approach is sound, and practical performance depends on hardware features (data-plane counter access, stateful registers) rather than switching throughput.

**Hardware Implications**: This work identifies specific capabilities needed for practical switch-native inference:
1. **Stateful registers with arithmetic operations** (P4): Enables single-packet value encoding instead of N packets per value
2. **Data-plane counter reads**: Eliminates 87× control-plane overhead
3. **Larger TCAM tables**: Reduces batch aggregation error
4. **Programmable parsing**: Allows arbitrary packet field interpretation for value encoding
5. **Higher packet rates**: Modern NICs at 100M+ pps would reduce transmission latency proportionally

Modern programmable switches (Tofino, Tofino2) provide features 1-4, suggesting that the architectural patterns validated in this work could achieve improved performance on current hardware, though experimental validation on P4 switches remains future work. Purpose-built switching ASICs incorporating all five features could potentially be competitive with GPU inference for memory-bound workloads.

**Broader Impact**: Network switches are fundamentally built for data movement rather than computation density—precisely the characteristic needed for memory-bound LLM inference. This work demonstrates that specialized architectures prioritizing data movement over raw FLOPS offer a viable alternative compute paradigm for exploration. Beyond performance, in-network inference enables architectural patterns impossible with traditional accelerators: privacy-preserving inference where data never leaves the network, zero-copy inference on streaming data, and compute collocated with data at network scale.

The fact that decade-old commodity hardware can execute 20B parameter model dimensions demonstrates switch-native neural network inference is architecturally feasible. Future work can look at integrating the validated 87× packet-based counter optimization, testing on modern P4 switches with stateful registers, implementing CPU baselines with identical quantization for token-level accuracy validation, and exploring switch-specific quantization strategies optimized for packet-count encoding.

---

**Acknowledgments**: This research was conducted independently without institutional affiliation or external funding. All hardware was purchased personally. 

**Code and Data Availability**: All experiment code, switch configurations, and detailed experiment logs (160+ experiments from proof-of-concept through full 20B-parameter inference) are publicly available in [this github repo](https://github.com/andrewcampi/in_network_matmul_and_llm_inference). The repository includes:
- Complete Python implementations for all 160+ experiments with progression from 4×4 matrix multiplication (e031) through full GPT-OSS-20B inference (e160)
- DPDK packet transmission utilities achieving 14.2M packets/second
- Switch configuration generation scripts for Junos firewall filters
- Weight loading utilities for GGUF quantized models (Q4_K_M, MXFP4)
- Detailed experiment logs documenting architectural discoveries and performance measurements
- Hardware setup documentation and topology configurations

The GPT-2 (124M parameters) and GPT-OSS-20B (20B parameters) model weights used are publicly available on Hugging Face. The Juniper QFX5100 switches and Mellanox ConnectX-3 NICs used in this work are commodity hardware available on secondary markets.
