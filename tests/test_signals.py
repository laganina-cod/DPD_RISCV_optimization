import numpy as np
from src.signals.generation import generate_qam_symbols, ofdm_modulate

def test_generate_qam_symbols():
    syms = generate_qam_symbols(1000, m_order=16, seed=0)
    assert len(syms) == 1000
    assert np.abs(np.mean(np.abs(syms)**2) - 1.0) < 0.1

def test_ofdm_modulate():
    syms = generate_qam_symbols(600, 16, seed=0)
    sig = ofdm_modulate(syms, n_fft=1024, cp_len=72, n_active=600)
    
    os_factor = 4
    expected_len = (1024 + 72) * os_factor
    assert len(sig) == expected_len

def test_cyclic_prefix():
    syms = generate_qam_symbols(600, 16, seed=1)
    sig = ofdm_modulate(syms, n_fft=1024, cp_len=72, n_active=600)
    
    os_factor = 4
    cp_len_eff = 72 * os_factor
    n_fft_eff = 1024 * os_factor
    
    cp_part = sig[:cp_len_eff]
    symbol_end = sig[n_fft_eff : n_fft_eff + cp_len_eff]
    
    np.testing.assert_array_almost_equal(cp_part, symbol_end, decimal=6)