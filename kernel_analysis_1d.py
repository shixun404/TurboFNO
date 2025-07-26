#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def analyze_performance(target_variant_name):
    """
    Analyzes kernel performance data to find total times and identify problem sizes
    where a specific variant is the best performer.
    """
    # Read CSV
    try:
        df = pd.read_csv('nsys_kernel_analysis_1d.csv')
    except FileNotFoundError:
        print("Error: 'nsys_kernel_analysis_1d.csv' not found. Please ensure the file is in the correct directory.")
        return

    # Create more detailed kernel names to distinguish variants
    def get_detailed_kernel_name(row):
        kernel_name = row['kernel_name']
        kernel_name_full = row['kernel_name_full']
        
        if 'direct_copy' in kernel_name_full.lower():
            if 'zero_padding' in kernel_name_full.lower():
                return 'direct_copy_zero_padding'
            elif 'truncation' in kernel_name_full.lower():
                return 'direct_copy_truncation'
            else:
                return kernel_name
        elif 'fused' in kernel_name_full.lower():
            if 'fused_fft_cgemm_ifft' in kernel_name_full.lower():
                return 'fused_fft_cgemm_ifft'
            elif 'fused_fft_cgemm' in kernel_name_full.lower():
                return 'fused_fft_cgemm'
            elif 'fused_cgemm_ifft' in kernel_name_full.lower():
                return 'fused_cgemm_ifft'
            else:
                return kernel_name
        elif '_DY(' in kernel_name_full:
            return kernel_name + '_DY'
        else:
            return kernel_name
    
    df['detailed_kernel_name'] = df.apply(get_detailed_kernel_name, axis=1)
    
    # Create parameter combination identifier
    df['param_combo'] = df.apply(lambda row: f"bs{row['bs']}_dx{row['dx']}_dy{row['dy']}_n{row['n']}_k{row['k']}", axis=1)
    
    all_variants = sorted(df['variant'].unique())
    param_combos = sorted(df['param_combo'].unique())
    
    # --- Part 1: Calculate total time for each variant per problem size ---
    
    all_times = {}
    print("Calculating total execution time for all variants and problem sizes...")
    
    for param_combo in param_combos:
        param_data = df[df['param_combo'] == param_combo]
        variant_times = {}
        
        for variant in all_variants:
            variant_data = param_data[param_data['variant'] == variant]
            total_time = variant_data['total_time_ms'].sum()
            if total_time > 0:
                variant_times[variant] = total_time
        
        if variant_times:
            all_times[param_combo] = variant_times

    print("\n--- Total Execution Time (ms) per Problem Size ---")
    for param_combo, variant_times in all_times.items():
        print(f"\nProblem Size: {param_combo}")
        for variant, total_time in sorted(variant_times.items()):
            print(f"  {variant}: {total_time:.4f} ms")
    
    # --- Part 2: Find where the target variant is the best and show percentages ---
    
    if target_variant_name not in all_variants:
        print(f"\nError: Variant '{target_variant_name}' not found in the data.")
        print(f"Available variants are: {', '.join(all_variants)}")
        return

    winning_params = []
    for param_combo, variant_times in all_times.items():
        if not variant_times:
            continue
        
        # Find the variant with the minimum time for this param_combo
        # In case of a tie, min() returns the first one, which is fine.
        best_variant = min(variant_times, key=variant_times.get)
        if best_variant == target_variant_name:
            winning_params.append(param_combo)

    print(f"\n--- Analysis for Variant: {target_variant_name} ---")
    
    if not winning_params:
        print(f"The variant '{target_variant_name}' was not the best performer for any problem size.")
    else:
        print(f"Found {len(winning_params)} problem size(s) where '{target_variant_name}' is the best performer:")
        for param_combo in winning_params:
            print(f"\n* Problem Size: {param_combo}")
            
            variant_times = all_times[param_combo]
            baseline_time = variant_times.get('TurboFNO_1D_E')

            if baseline_time is None or baseline_time == 0:
                print("  Baseline 'TurboFNO_1D_E' time not available or is zero. Showing absolute times:")
                for variant, total_time in sorted(variant_times.items()):
                    print(f"    {variant}: {total_time:.4f} ms")
            else:
                print("  Performance relative to baseline 'TurboFNO_1D_E':")
                for variant, total_time in sorted(variant_times.items()):
                    percentage = (total_time / baseline_time) * 100
                    print(f"    {variant}: {total_time:.4f} ms ({percentage:.2f}%)")

def main():
    target_variant_name = input("请输入您想分析的variant name (例如: TurboFNO_1D_D): ")
    if target_variant_name:
        analyze_performance(target_variant_name.strip())

if __name__ == '__main__':
    main() 