#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Read CSV
    df = pd.read_csv('nsys_kernel_analysis_1d.csv')
    
    # Create more detailed kernel names to distinguish variants
    def get_detailed_kernel_name(row):
        kernel_name = row['kernel_name']
        kernel_name_full = row['kernel_name_full']
        
        # For direct_copy kernels, extract the specific variant
        if 'direct_copy' in kernel_name_full.lower():
            if 'zero_padding' in kernel_name_full.lower():
                return 'direct_copy_zero_padding'
            elif 'truncation' in kernel_name_full.lower():
                return 'direct_copy_truncation'
            else:
                return kernel_name
        # For fused kernels, distinguish between different types
        elif 'fused' in kernel_name_full.lower():
            if 'fused_fft_cgemm_ifft' in kernel_name_full.lower():
                return 'fused_fft_cgemm_ifft'
            elif 'fused_fft_cgemm' in kernel_name_full.lower():
                return 'fused_fft_cgemm'
            elif 'fused_cgemm_ifft' in kernel_name_full.lower():
                return 'fused_cgemm_ifft'
            else:
                return kernel_name
        # For other kernels, check if they have _DY suffix in full name
        elif '_DY(' in kernel_name_full:
            return kernel_name + '_DY'
        else:
            return kernel_name
    
    # Mapping function to convert detailed kernel names to display labels
    def get_display_label(detailed_kernel_name):
        label_mapping = {
            'fft_7_stride': 'fft_7',
            'fft_8_stride': 'fft_8', 
            'fft_7_stride_DY': 'fft_DY_7',
            'fft_8_stride_DY': 'fft_DY_8',
            'ifft_7_stride': 'ifft_7',
            'ifft_8_stride': 'ifft_8',
            'ifft_7_stride_DY': 'ifft_DY_7',
            'ifft_8_stride_DY': 'ifft_DY_8',
            'cgemm': 'cgemm',
            'fused_fft_cgemm': 'fft_cgemm',
            'fused_cgemm_ifft': 'cgemm_ifft', 
            'fused_fft_cgemm_ifft': 'fft_cgemm_ifft',
            'direct_copy_zero_padding': 'zero_padding',
            'direct_copy_truncation': 'truncation'
            

        }
        return label_mapping.get(detailed_kernel_name, detailed_kernel_name)
    
    # Apply the detailed kernel naming
    df['detailed_kernel_name'] = df.apply(get_detailed_kernel_name, axis=1)
    
    # Create parameter combination identifier
    df['param_combo'] = df.apply(lambda row: f"bs{row['bs']}_dx{row['dx']}_dy{row['dy']}_n{row['n']}_k{row['k']}", axis=1)
    
    # Get all unique variants and kernels
    all_variants = ['TurboFNO_1D_A', 'TurboFNO_1D_B', 'TurboFNO_1D_C', 'TurboFNO_1D_D', 'TurboFNO_1D_E']
    all_kernels = df['detailed_kernel_name'].unique()  # Use detailed kernel names
    
    # Create output directory
    os.makedirs('kernel_breakdown_plots_1d', exist_ok=True)
    
    # Get all parameter combinations
    param_combos = sorted(df['param_combo'].unique())
    
    print(f"Generating plots for {len(param_combos)} parameter combinations...")
    
    for i, param_combo in enumerate(param_combos):
        print(f"  {i+1}/{len(param_combos)}: {param_combo}")
        
        # Check if file already exists
        filename = f"kernel_breakdown_{param_combo}.png"
        filepath = os.path.join('kernel_breakdown_plots_1d', filename)
        
        if os.path.exists(filepath):
            print(f"    Skipping - file already exists: {filename}")
            continue
        
        # Get data for this parameter combination
        param_data = df[df['param_combo'] == param_combo]
        
        # Check if all 5 variants have data
        variants_in_data = set(param_data['variant'].unique())
        if len(variants_in_data) < 5 or not all(v in variants_in_data for v in all_variants):
            missing_variants = set(all_variants) - variants_in_data
            print(f"    Skipping - missing variants: {missing_variants}")
            continue
        
        # Create a matrix: variants x kernels
        variant_kernel_times = {}
        
        for variant in all_variants:
            variant_data = param_data[param_data['variant'] == variant]
            
            # Get kernel times for this variant
            kernel_times = {}
            for kernel in all_kernels:
                kernel_data = variant_data[variant_data['detailed_kernel_name'] == kernel]  # Use detailed kernel names
                total_time = kernel_data['total_time_ms'].sum() if len(kernel_data) > 0 else 0
                if total_time > 0:  # Only include kernels with non-zero time
                    kernel_times[kernel] = total_time
            
            variant_kernel_times[variant] = kernel_times
        
        # Get all kernels that appear in this parameter combination
        all_kernels_in_combo = set()
        for variant_kernels in variant_kernel_times.values():
            all_kernels_in_combo.update(variant_kernels.keys())
        
        if not all_kernels_in_combo:
            continue  # Skip if no kernels found
        
        all_kernels_in_combo = sorted(all_kernels_in_combo)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for stacked bar chart
        bottom = np.zeros(len(all_variants))
        # colors = plt.cm.Set3(np.linspace(0, 1, len(all_kernels_in_combo)))
        colors = sns.color_palette("tab10")
        
        for j, kernel in enumerate(all_kernels_in_combo):
            kernel_times = []
            for variant in all_variants:
                time = variant_kernel_times[variant].get(kernel, 0)
                kernel_times.append(time)
            
            # Only plot if at least one variant has this kernel
            if sum(kernel_times) > 0:
                color_idx = j % len(colors)
                bars = ax.bar(range(len(all_variants)), kernel_times, bottom=bottom, 
                    #    label=kernel, color=colors[j], alpha=0.8
                    label=kernel, color=colors[color_idx], alpha=0.8
                       )
                
                # Add text labels on bars for kernels with significant time
                display_label = get_display_label(kernel)
                for k, (bar, time) in enumerate(zip(bars, kernel_times)):
                    if time > 0:  # Only add label if there's actual time
                        bar_height = bar.get_height()
                        if bar_height > 0.1:  # Only label bars that are large enough to read
                            # Position text in the middle of the bar
                            text_y = bottom[k] + bar_height / 2
                            ax.text(bar.get_x() + bar.get_width()/2, text_y, 
                                   display_label, ha='center', va='center', 
                                   fontsize=8, fontweight='bold', color='black')
                
                bottom += kernel_times
        
        # Customize plot
        ax.set_xlabel('Variant')
        ax.set_ylabel('Total Time (ms)')
        ax.set_title(f'Kernel Breakdown - {param_combo}')
        ax.set_xticks(range(len(all_variants)))
        ax.set_xticklabels([v.replace('TurboFNO_1D_', '') for v in all_variants])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"\nDone! Generated {len(param_combos)} plots in 'kernel_breakdown_plots_1d/' directory")

if __name__ == '__main__':
    main() 