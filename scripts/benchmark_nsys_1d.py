#!/usr/bin/env python3
import subprocess
import sys
import os

# TurboFNO 1D variants configuration
variants = {
    'TurboFNO_1D_A': 'fusion_variants_benchmark/1D_A_exp_fft+cgemm+ifft/build/TurboFNO_1D_A',
    'TurboFNO_1D_B': 'fusion_variants_benchmark/1D_B_exp_fused_fft_cgemm+ifft/build/TurboFNO_1D_B',
    'TurboFNO_1D_C': 'fusion_variants_benchmark/1D_C_exp_fft+fused_cgemm_ifft/build/TurboFNO_1D_C',
    'TurboFNO_1D_D': 'fusion_variants_benchmark/1D_D_exp_fused_fft_cgemm_ifft/build/TurboFNO_1D_D',
    'TurboFNO_1D_E': 'fusion_variants_benchmark/1D_E_baseline/build/TurboFNO_1D_E'
}

# Read configuration from file
def parse_config_file(config_path):
    return {
        'bs_list': [32, 64, 128],
        'DX_list': [128, 256],
        'DY_list': [128, 256],
        'N_list': [64, 128],
        'K_list': [8, 16, 32, 64, 128]
    }
    
    return config

# Get project root
project_root = os.environ.get('PROJECT_ROOT', os.getcwd())

# Load configuration
config = parse_config_file(os.path.join(project_root, 'benchmark_config/problem_size_1d.txt'))

# Create output directories
output_folder = "nsys_profiles_1d"
output_stats_folder = "nsys_stats_1d"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_stats_folder, exist_ok=True)

print("TurboFNO 1D Variants Benchmark with nsys")
print("=" * 50)
print(f"Project root: {project_root}")
print(f"Config: {config}")
print(f"Variants: {len(variants)}")

total_runs = 0

for bs in config['bs_list']:
    for dx in config['DX_list']:
        for dy in config['DY_list']:
            for n in config['N_list']:
                for k in config['K_list']:
                    for variant_name, executable_path in variants.items():
                        full_path = os.path.join(project_root, executable_path)
                        if not os.path.exists(full_path):
                            print(f"Executable not found: {full_path}")
                            continue
                        
                        print(f"\nTesting variant: {variant_name}")
                        print(f"Executable: {full_path}")
                        
                        # Check if stats file already exists
                        stats_filename = os.path.join(output_stats_folder, f"{variant_name}_bs{bs}_dx{dx}_dy{dy}_n{n}_k{k}.txt")
                        if os.path.exists(stats_filename):
                            print(f"⏭️  Skipping {variant_name} - bs={bs}, dx={dx}, dy={dy}, n={n}, k={k} (stats file already exists)")
                            continue
                        
                        total_runs += 1
                        print(f"Run {total_runs}: {variant_name} - bs={bs}, dx={dx}, dy={dy}, n={n}, k={k}")
                        
                        # Create output filename with all parameters
                        output_filename = os.path.join(output_folder, f"{variant_name}_bs{bs}_dx{dx}_dy{dy}_n{n}_k{k}")
                        
                        cmd = [
                            'nsys', 'profile', 
                            '--trace=cuda,nvtx', 
                            '--stats', 'true', 
                            '--force-overwrite', 'true',
                            '--output', output_filename,
                            full_path,
                            str(bs), str(dx), str(dy), str(n), str(k)
                        ]
                        
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                            print(f"✓ Command completed successfully. Output saved to {output_filename}.nsys-rep")
                            
                            # Write nsys stats output to text file
                            with open(stats_filename, 'w') as f:
                                f.write(f"NSYS Profile Results for {variant_name}:\n")
                                f.write(f"bs={bs}, dx={dx}, dy={dy}, n={n}, k={k}\n")
                                f.write("=" * 80 + "\n\n")
                                if result.stdout:
                                    f.write("STDOUT:\n")
                                    f.write(result.stdout)
                                    f.write("\n\n")
                                if result.stderr:
                                    f.write("STDERR:\n")
                                    f.write(result.stderr)
                                    f.write("\n")
                            print(f"✓ Stats output saved to {stats_filename}")
                            
                        except subprocess.CalledProcessError as e:
                            print(f"✗ Command failed with exit code {e.returncode}")
                            # Write error output to text file
                            error_filename = f"{output_filename}_error.txt"
                            with open(error_filename, 'w') as f:
                                f.write(f"NSYS Profile ERROR for {variant_name}:\n")
                                f.write(f"bs={bs}, dx={dx}, dy={dy}, n={n}, k={k}\n")
                                f.write("=" * 80 + "\n\n")
                                f.write(f"Exit code: {e.returncode}\n\n")
                                if e.stdout:
                                    f.write("STDOUT:\n")
                                    f.write(e.stdout)
                                    f.write("\n\n")
                                if e.stderr:
                                    f.write("STDERR:\n")
                                    f.write(e.stderr)
                                    f.write("\n")
                            print(f"✗ Error output saved to {error_filename}")
                        except subprocess.TimeoutExpired:
                            print(f"✗ Command timed out after 300 seconds")
                        except Exception as e:
                            print(f"✗ Error running command: {e}")
                        
                        print("-" * 60)

print(f"\nBenchmark completed! Total runs: {total_runs}")
print(f"Profile files saved to: {output_folder}")
print(f"Stats files saved to: {output_stats_folder}") 