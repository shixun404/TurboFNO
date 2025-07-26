#!/usr/bin/env python3
"""
Analyze nsys stats output files and extract kernel timing information to CSV
Simple version without pandas dependency
"""
import os
import re
import csv
import argparse

def parse_filename(filename):
    """Parse variant name and parameters from filename"""
    # Example: TurboFNO_2D_A_bs1_dx128_dy128_n64_k8.txt
    parts = filename.replace('.txt', '').split('_')
    
    variant = None
    params = {}
    
    # Find variant name (TurboFNO_2D_X)
    for i, part in enumerate(parts):
        if part.startswith('TurboFNO') and i + 2 < len(parts):
            variant = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
            break
    
    # Extract parameters
    for part in parts:
        if part.startswith('bs') and part[2:].isdigit():
            params['bs'] = int(part[2:])
        elif part.startswith('dx') and part[2:].isdigit():
            params['dx'] = int(part[2:])
        elif part.startswith('dy') and part[2:].isdigit():
            params['dy'] = int(part[2:])
        elif part.startswith('n') and part[1:].isdigit():
            params['n'] = int(part[1:])
        elif part.startswith('k') and part[1:].isdigit():
            params['k'] = int(part[1:])
    
    return variant, params

def parse_nsys_stats(content):
    """Parse nsys stats output and extract kernel timing information"""
    kernels = []
    lines = content.split('\n')
    
    # Look for cuda_gpu_kern_sum section
    in_gpu_section = False
    header_found = False
    
    for line in lines:
        line_strip = line.strip()
        
        # Start of GPU kernel stats section
        if "Executing 'cuda_gpu_kern_sum' stats report" in line:
            in_gpu_section = True
            header_found = False
            continue
        
        # End of GPU kernel stats section
        if in_gpu_section and (line_strip.startswith('[') and 'Executing' in line_strip):
            in_gpu_section = False
            continue
        
        if in_gpu_section:
            # Skip header and separator lines
            if not header_found and ('Time (%)' in line_strip and 'Name' in line_strip):
                header_found = True
                continue
            
            if header_found and (line_strip.startswith('---') or line_strip.startswith('=')):
                continue
                
            if header_found and line_strip and not line_strip.startswith('['):
                # Parse kernel data lines - space separated values
                # Format: Time(%) Total_Time(ns) Instances Avg(ns) Med(ns) Min(ns) Max(ns) StdDev(ns) Name
                parts = line_strip.split()
                
                if len(parts) >= 9:  # At least 8 columns + name
                    try:
                        time_percent = float(parts[0])
                        total_time_ns = int(parts[1].replace(',', ''))
                        instances = int(parts[2].replace(',', ''))
                        avg_time_ns = float(parts[3].replace(',', ''))
                        
                        # The name is everything from index 8 onwards
                        kernel_name_full = ' '.join(parts[8:])
                        
                        # Skip if no meaningful data
                        if total_time_ns == 0:
                            continue
                        
                        # Clean up kernel name
                        kernel_name_clean = clean_kernel_name(kernel_name_full)
                        
                        # Convert nanoseconds to milliseconds
                        total_time_ms = total_time_ns / 1_000_000
                        avg_time_ms = avg_time_ns / 1_000_000
                        
                        kernels.append({
                            'kernel_name': kernel_name_clean,
                            'kernel_name_full': kernel_name_full,
                            'total_time_ms': total_time_ms,
                            'avg_time_ms': avg_time_ms,
                            'instances': instances,
                            'time_percent': time_percent
                        })
                        
                    except (ValueError, IndexError):
                        # Skip lines that can't be parsed
                        continue
    
    return kernels

def clean_kernel_name(kernel_name):
    """Clean kernel name by removing template parameters and long suffixes"""
    # Remove template parameters
    kernel_name = re.sub(r'<[^>]*>', '', kernel_name)
    # Remove function parameters
    kernel_name = re.sub(r'\([^)]*\)', '', kernel_name)
    # Remove long namespace prefixes
    if '::' in kernel_name:
        kernel_name = kernel_name.split('::')[-1]
    # Take first meaningful part
    parts = kernel_name.split('_')
    if len(parts) > 3:
        kernel_name = '_'.join(parts[:3])
    
    return kernel_name.strip()



def analyze_nsys_results(stats_dir, output_csv):
    """Analyze all nsys stats files and save to CSV"""
    
    if not os.path.exists(stats_dir):
        print(f"Stats directory not found: {stats_dir}")
        return
    
    all_results = []
    
    print(f"Analyzing files in {stats_dir}")
    
    # Process all .txt files
    for filename in os.listdir(stats_dir):
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(stats_dir, filename)
        print(f"Processing: {filename}")
        
        # Parse filename to extract variant and parameters
        variant, params = parse_filename(filename)
        
        if variant is None:
            print(f"  Warning: Could not parse variant from {filename}")
            continue
        
        # Read file content
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            continue
        
        # Parse nsys stats
        kernels = parse_nsys_stats(content)
        
        print(f"  Found {len(kernels)} kernels")
        if kernels:
            print(f"  Top kernel: {kernels[0]['kernel_name']} ({kernels[0]['total_time_ms']:.3f}ms)")
        
        # Add to results
        for kernel in kernels:
            result = {
                'variant': variant,
                'bs': params.get('bs', ''),
                'dx': params.get('dx', ''),
                'dy': params.get('dy', ''),
                'n': params.get('n', ''),
                'k': params.get('k', ''),
                'kernel_name': kernel['kernel_name'],
                'kernel_name_full': kernel['kernel_name_full'],
                'total_time_ms': kernel['total_time_ms'],
                'avg_time_ms': kernel['avg_time_ms'],
                'instances': kernel['instances'],
                'time_percent': kernel.get('time_percent', 0.0)
            }
            all_results.append(result)
    
    # Save to CSV
    if all_results:
        # Sort results
        all_results.sort(key=lambda x: (x['variant'], x['bs'], x['dx'], x['dy'], x['n'], x['k'], -x['total_time_ms']))
        
        # Write CSV
        fieldnames = ['variant', 'bs', 'dx', 'dy', 'n', 'k', 'kernel_name', 'kernel_name_full', 
                     'total_time_ms', 'avg_time_ms', 'instances', 'time_percent']
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Total records: {len(all_results)}")
        
        # Print summary
        variants = set(r['variant'] for r in all_results)
        kernels = set(r['kernel_name'] for r in all_results)
        
        print("\nSummary:")
        print(f"Variants: {len(variants)}")
        print(f"Unique kernels: {len(kernels)}")
        
        # Show top kernels by total time
        print("\nTop 10 kernels by total time:")
        top_kernels = sorted(all_results, key=lambda x: x['total_time_ms'], reverse=True)[:10]
        for i, kernel in enumerate(top_kernels, 1):
            print(f"{i:2d}. {kernel['variant']:15s} {kernel['kernel_name']:25s} {kernel['total_time_ms']:8.3f}ms ({kernel['time_percent']:5.1f}%) "
                  f"(bs={kernel['bs']}, dx={kernel['dx']}, dy={kernel['dy']}, n={kernel['n']}, k={kernel['k']})")
        
    else:
        print("No results found!")

def generate_summary_report(csv_file, output_dir):
    """Generate additional summary reports"""
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read CSV data
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric fields
            row['total_time_ms'] = float(row['total_time_ms'])
            row['avg_time_ms'] = float(row['avg_time_ms']) if row['avg_time_ms'] else 0
            row['instances'] = int(row['instances'])
            row['time_percent'] = float(row.get('time_percent', 0))
            data.append(row)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Variant comparison
    variant_stats = {}
    for row in data:
        variant = row['variant']
        time_ms = row['total_time_ms']
        if variant not in variant_stats:
            variant_stats[variant] = {'sum': 0, 'count': 0, 'times': []}
        variant_stats[variant]['sum'] += time_ms
        variant_stats[variant]['count'] += 1
        variant_stats[variant]['times'].append(time_ms)
    
    with open(os.path.join(output_dir, 'variant_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'total_time_ms', 'avg_time_ms', 'kernel_count'])
        for variant, stats in variant_stats.items():
            avg_time = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
            writer.writerow([variant, round(stats['sum'], 3), round(avg_time, 3), stats['count']])
    
    # 2. Kernel analysis
    kernel_stats = {}
    for row in data:
        kernel = row['kernel_name']
        time_ms = row['total_time_ms']
        if kernel not in kernel_stats:
            kernel_stats[kernel] = {'sum': 0, 'count': 0}
        kernel_stats[kernel]['sum'] += time_ms
        kernel_stats[kernel]['count'] += 1
    
    with open(os.path.join(output_dir, 'kernel_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kernel_name', 'total_time_ms', 'avg_time_ms', 'count'])
        sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]['sum'], reverse=True)
        for kernel, stats in sorted_kernels:
            avg_time = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
            writer.writerow([kernel, round(stats['sum'], 3), round(avg_time, 3), stats['count']])
    
    # 3. Parameter impact analysis
    param_stats = {}
    for row in data:
        param_key = f"bs{row['bs']}_dx{row['dx']}_dy{row['dy']}_n{row['n']}_k{row['k']}"
        if param_key not in param_stats:
            param_stats[param_key] = 0.0
        param_stats[param_key] += row['total_time_ms']
    
    with open(os.path.join(output_dir, 'parameter_analysis.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bs', 'dx', 'dy', 'n', 'k', 'total_time_ms'])
        sorted_params = sorted(param_stats.items(), key=lambda x: x[1], reverse=True)
        for param_key, total_time in sorted_params:
            # Parse back the parameters
            parts = param_key.split('_')
            bs = int(parts[0][2:])
            dx = int(parts[1][2:])
            dy = int(parts[2][2:])
            n = int(parts[3][1:])
            k = int(parts[4][1:])
            writer.writerow([bs, dx, dy, n, k, round(total_time, 3)])
    
    print(f"Summary reports saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze nsys stats output files')
    parser.add_argument('--stats-dir', '-s', 
                       default='nsys_stats_2d',
                       help='Directory containing nsys stats files')
    parser.add_argument('--output-csv', '-o',
                       default='nsys_kernel_analysis.csv',
                       help='Output CSV file')
    parser.add_argument('--summary-dir', 
                       default='analysis_reports',
                       help='Directory for summary reports')
    
    args = parser.parse_args()
    
    print("NSYS Results Analysis Tool (Simple Version)")
    print("=" * 50)
    
    # Analyze results
    analyze_nsys_results(args.stats_dir, args.output_csv)
    
    # Generate summary reports
    generate_summary_report(args.output_csv, args.summary_dir)
    
    print("\nAnalysis completed!")

if __name__ == '__main__':
    main() 