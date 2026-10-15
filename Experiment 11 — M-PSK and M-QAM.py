# ==========================================
# Experiment 11: M-PSK and M-QAM Simulation
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------
# 1. Helper Functions for Gray Code & Mapping
# ------------------------------------------
def binary_to_gray(b):
    return b ^ (b >> 1)

def int_to_gray_bits(i, num_bits):
    gray_val = binary_to_gray(i)
    return np.array([int(x) for x in f"{gray_val:0{num_bits}b}"], dtype=int)

# ------------------------------------------
# 2. Constellation Generation & Normalization
# ------------------------------------------
def get_psk_constellation(M):
    k = int(np.log2(M))
    symbols = []
    mapping = {}
    for i in range(M):
        phase = 2 * np.pi * i / M
        s = np.exp(1j * phase)
        symbols.append(s)
        bits = int_to_gray_bits(i, k)
        mapping[tuple(bits)] = s
    return np.array(symbols), mapping

def get_qam_constellation(M):
    k = int(np.log2(M))
    sqrt_M = int(np.sqrt(M))
    # PAM levels using Gray-like spacing
    pam_levels = np.arange(-(sqrt_M - 1), sqrt_M, 2, dtype=float)
    symbols = []
    mapping = {}
    
    idx = 0
    for real in pam_levels:
        for imag in pam_levels:
            s = real + 1j * imag
            symbols.append(s)
            bits_real = int_to_gray_bits(int((real + sqrt_M - 1) / 2), k // 2)
            bits_imag = int_to_gray_bits(int((imag + sqrt_M - 1) / 2), k // 2)
            bits = np.concatenate([bits_real, bits_imag])
            mapping[tuple(bits)] = s
            idx += 1
    return np.array(symbols), mapping

# Define modulations dictionary
modulations = {
    'QPSK': {'M': 4, 'type': 'PSK'},
    '8-PSK': {'M': 8, 'type': 'PSK'},
    '16-QAM': {'M': 16, 'type': 'QAM'},
    '64-QAM': {'M': 64, 'type': 'QAM'}
}

print("--- MANDATORY VALIDATION: ENERGY NORMALIZATION ---")
constellations = {}
for name, params in modulations.items():
    M = params['M']
    if params['type'] == 'PSK':
        syms, mapping = get_psk_constellation(M)
    else:
        syms, mapping = get_qam_constellation(M)
        
    # Before normalization energy
    energy_before = np.mean(np.abs(syms)**2)
    
    # Normalize average symbol energy to 1
    syms_normalized = syms / np.sqrt(energy_before)
    energy_after = np.mean(np.abs(syms_normalized)**2)
    
    # Update mapping with normalized symbols
    norm_mapping = {k: v / np.sqrt(energy_before) for k, v in mapping.items()}
    
    constellations[name] = {
        'symbols': syms_normalized,
        'mapping': norm_mapping,
        'M': M,
        'k': int(np.log2(M))
    }
    
    print(f"Modulation: {name:<8} | Pre-normalization Energy: {energy_before:.4f} | Post-normalization Energy: {energy_after:.4f}")

# ------------------------------------------
# 3. Simulation Engine (AWGN & Min-Distance)
# ------------------------------------------
def minimum_distance_detector(received_signal, constellation_symbols):
    # Minimum Euclidean distance detection
    distances = np.abs(received_signal[:, None] - constellation_symbols[None, :])
    min_indices = np.argmin(distances, axis=1)
    return constellation_symbols[min_indices]

# Test SNR values in dB
snr_dbs = [0, 5, 10, 20]
num_symbols = 10000

results = {}

for name, data in constellations.items():
    M = data['M']
    k = data['k']
    syms = data['symbols']
    
    ber_list = []
    ser_list = []
    
    for snr_db in snr_dbs:
        # Generate random symbol indices
        tx_indices = np.random.randint(0, M, num_symbols)
        tx_symbols = syms[tx_indices]
        
        # Calculate noise variance based on Eb/N0 or SNR (using Es/N0 here)
        snr_linear = 10**(snr_db / 10.0)
        # Average symbol energy is 1, noise variance per complex dimension = 1 / (2 * snr_linear)
        noise_var = 1.0 / (2.0 * snr_linear)
        noise = np.sqrt(noise_var) * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        
        # Received signal through AWGN channel
        rx_symbols = tx_symbols + noise
        
        # Detection
        detected_symbols = minimum_distance_detector(rx_symbols, syms)
        
        # Find indices of detected symbols
        rx_indices = np.array([np.where(np.isclose(syms, ds))[0][0] for ds in detected_symbols])
        
        # Calculate SER
        ser = np.mean(tx_indices != rx_indices)
        ser_list.append(ser)
        
        # Approximate BER conversion for Gray coding (BER ≈ SER / k)
        ber = ser / k
        ber_list.append(ber)
        
    results[name] = {'ber': ber_list, 'ser': ser_list}

# ------------------------------------------
# 4. Required Visualizations
# ------------------------------------------

# A. Ideal and Received Constellations (at 10 dB)
plt.figure(figsize=(14, 10))
for idx, (name, data) in enumerate(constellations.items(), 1):
    plt.subplot(2, 2, idx)
    syms = data['symbols']
    
    # Simulate transmission at 10 dB for visualization
    snr_linear = 10**(10 / 10.0)
    noise = np.sqrt(1.0 / (2.0 * snr_linear)) * (np.random.randn(1000) + 1j * np.random.randn(1000))
    tx_sample = np.random.choice(syms, 1000)
    rx_sample = tx_sample + noise
    
    plt.scatter(rx_sample.real, rx_sample.imag, alpha=0.5, color='orange', label='Received (10 dB)')
    plt.scatter(syms.real, syms.imag, color='blue', s=50, marker='x', label='Ideal Constellation')
    plt.title(f"{name} Constellation")
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

# B. Decision Regions Example (16-QAM)
plt.figure(figsize=(7, 6))
grid_size = 200
x = np.linspace(-2, 2, grid_size)
y = np.linspace(-2, 2, grid_size)
xx, yy = np.meshgrid(x, y)
grid_points = xx + 1j * yy
flat_grid = grid_points.flatten()

# Detect for each grid point
detected_grid = minimum_distance_detector(flat_grid, constellations['16-QAM']['symbols'])
region_map = np.array([np.where(np.isclose(constellations['16-QAM']['symbols'], dg))[0][0] for dg in detected_grid])
region_map = region_map.reshape(grid_size, grid_size)

plt.imshow(region_map, extent=[-2, 2, -2, 2], origin='lower', alpha=0.3, cmap='jet')
plt.scatter(constellations['16-QAM']['symbols'].real, constellations['16-QAM']['symbols'].imag, color='black', marker='x', s=100, label='16-QAM Symbols')
plt.title("Decision Regions for 16-QAM")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()

# C. BER/SER Performance Curves
plt.figure(figsize=(10, 5))
for name, res in results.items():
    plt.plot(snr_dbs, res['ber'], marker='o', label=f"{name} (BER)")
plt.yscale('log')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR for M-PSK and M-QAM")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# D. BER versus Spectral Efficiency (at 10 dB)
spectral_efficiencies = {name: data['k'] for name, data in constellations.items()} # bits/s/Hz (assuming Nyquist bandwidth)
ber_at_10db = {name: results[name]['ber'][snr_dbs.index(10)] for name in results.items()}

plt.figure(figsize=(8, 5))
names = list(spectral_efficiencies.keys())
se_vals = list(spectral_efficiencies.values())
ber_vals = [ber_at_10db[n] for n in names]

plt.bar(names, ber_vals, color=['skyblue', 'salmon', 'lightgreen', 'plum'])
plt.yscale('log')
plt.ylabel("BER at 10 dB (Log Scale)")
plt.xlabel("Modulation Scheme & Spectral Efficiency (bits/symbol/Hz)")
plt.title("Comparison of BER vs Spectral Efficiency at 10 dB SNR")
for i, v in enumerate(se_vals):
    plt.text(i, ber_vals[i] * 1.5, f"SE: {v} b/s/Hz", ha='center', fontweight='bold')
plt.grid(axis='y', which="both", ls="--")
plt.show()
