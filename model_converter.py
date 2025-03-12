import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Union, Any
import warnings

from soen_sim_v2.utils.physical_mappings.soen_conversion_utils import PhysicalConverter

# Try to import pandas, but handle the case where it might not be available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except (ImportError, ValueError):
    warnings.warn("Pandas is not available or incompatible. CSV export functionality will be limited.")
    PANDAS_AVAILABLE = False

def convert_model_to_physical(
    model_state_dict: Dict, 
    I_c: float = 100e-6, 
    gamma_c: float = 1.5e-9, 
    beta_c: float = 0.3,
    save_path: Optional[str] = None,
    create_plots: bool = False,
    skip_connections: bool = False  # Added parameter to optionally skip connection processing
) -> Dict:
    """
    Convert dimensionless SOEN model parameters to physical values.
    
    Args:
        model_state_dict: Dictionary containing model parameters (from torch.load)
        I_c: Critical current in Amperes (default: 100e-6)
        gamma_c: Capacitance proportionality in F/A (default: 1.5e-9)
        beta_c: Stewart-McCumber parameter (default: 0.3)
        save_path: Path to save CSV output (default: None)
        create_plots: Whether to create histograms of parameter distributions (default: False)
        skip_connections: Whether to skip connection processing (default: False)
        
    Returns:
        Dictionary containing physical parameter values for all layers and connections
    """
    # Create PhysicalConverter instance
    converter = PhysicalConverter(I_c=I_c, gamma_c=gamma_c, beta_c=beta_c)
    
    # Initialize results dictionary with separate dimensionless and physical sections
    results = {
        'base_parameters': converter.get_base_parameters(),
        'dimensionless': {
            'layers': {},
            'connections': {}
        },
        'physical': {
            'layers': {},
            'connections': {}
        }
    }
    
    # Extract parameters from state dict
    # Handle both cases: 
    # 1. When model_state_dict key is present (saved with model.state_dict())
    # 2. When the dict itself is the state dict (saved directly)
    state_dict = None
    if isinstance(model_state_dict, dict):
        if 'model_state_dict' in model_state_dict:
            state_dict = model_state_dict['model_state_dict']
        else:
            # Check if this looks like a state dict (contains 'layers' or 'connections')
            if any(k.startswith('layers.') or k.startswith('connections.') for k in model_state_dict.keys()):
                state_dict = model_state_dict
            else:
                # If it doesn't look like a state dict, check again inside
                for key, value in model_state_dict.items():
                    if isinstance(value, dict) and any(k.startswith('layers.') or k.startswith('connections.') for k in value.keys()):
                        state_dict = value
                        break
                
                # If we still don't have a state dict, use the original
                if state_dict is None:
                    state_dict = model_state_dict
    else:
        state_dict = model_state_dict
    
    # Process parameters with clear separation between physical and dimensionless
    for full_key, tensor in state_dict.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            continue
            
        # Convert tensor to numpy for easier processing
        param_value = tensor.detach().cpu().numpy()
        
        # Parse the parameter key to identify layer, type, and parameter name
        parts = full_key.split('.')
        
        try:
            # Handle different parameter types
            if 'layers' in full_key:
                # Extract layer index
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts):
                        try:
                            layer_idx = int(parts[i+1])
                            break
                        except ValueError:
                            continue
                
                if layer_idx is None:
                    continue
                    
                # Initialize layer entries if not exist
                if layer_idx not in results['dimensionless']['layers']:
                    results['dimensionless']['layers'][layer_idx] = {}
                if layer_idx not in results['physical']['layers']:
                    results['physical']['layers'][layer_idx] = {}
                    
                # Process layer parameters
                param_name = parts[-1]
                
                # Handle different parameter types - now with clear separation
                if 'log_gamma_plus' in param_name:
                    # Store original log value in dimensionless
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
                    # Convert log_gamma_plus to gamma_plus in dimensionless
                    gamma_plus = np.exp(param_value)
                    results['dimensionless']['layers'][layer_idx][param_name.replace('log_', '')] = gamma_plus
                    
                    # Calculate inductance L from gamma_plus for physical
                    beta_L = 1.0 / gamma_plus
                    L_values = np.array([converter.dimensionless_to_physical_inductance(bl) for bl in beta_L])
                    results['physical']['layers'][layer_idx]['L' + param_name.replace('log_gamma_plus', '')] = L_values
                    
                elif 'log_gamma_minus' in param_name:
                    # Store original log value in dimensionless
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
                    # Convert log_gamma_minus to gamma_minus in dimensionless
                    gamma_minus = np.exp(param_value)
                    results['dimensionless']['layers'][layer_idx][param_name.replace('log_', '')] = gamma_minus
                    
                    # If we have the corresponding gamma_plus, calculate r_leak for physical
                    gamma_plus_key = param_name.replace('log_gamma_minus', 'gamma_plus')
                    if gamma_plus_key in results['dimensionless']['layers'][layer_idx]:
                        gamma_plus = results['dimensionless']['layers'][layer_idx][gamma_plus_key]
                        alpha = gamma_minus / gamma_plus
                        r_leak = np.array([converter.dimensionless_to_physical_resistance(a) for a in alpha])
                        results['physical']['layers'][layer_idx]['r_leak' + param_name.replace('log_gamma_minus', '')] = r_leak
                    
                elif 'phi_offset' in param_name:
                    # Store original dimensionless value
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
                    # Convert phi_offset to physical flux
                    physical_flux = param_value * converter.Phi_0
                    results['physical']['layers'][layer_idx][param_name] = physical_flux
                    
                elif 'dendrite_offset' in param_name:
                    # Store original dimensionless value
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
                    # Convert dendrite_offset to physical flux
                    physical_flux = param_value * converter.Phi_0
                    results['physical']['layers'][layer_idx][param_name] = physical_flux
                    
                elif 'bias_current' in param_name:
                    # Store original dimensionless value
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
                    # Convert bias_current to physical current
                    physical_current = param_value * converter.I_c
                    results['physical']['layers'][layer_idx][param_name] = physical_current
                
                else:
                    # Store other parameters as-is in dimensionless only
                    results['dimensionless']['layers'][layer_idx][param_name] = param_value
                    
            elif 'connections' in full_key and not skip_connections:
                try:
                    # Extract connection key
                    conn_key = None
                    for i, part in enumerate(parts):
                        if part == 'connections' and i+1 < len(parts):
                            conn_key = parts[i+1]
                            break
                    
                    if conn_key is None:
                        print(f"Warning: Could not extract connection key from {full_key}")
                        continue
                        
                    # Initialize connection entries if not exist
                    if conn_key not in results['dimensionless']['connections']:
                        results['dimensionless']['connections'][conn_key] = {}
                    if conn_key not in results['physical']['connections']:
                        results['physical']['connections'][conn_key] = {}
                        
                    # Store J values in dimensionless
                    results['dimensionless']['connections'][conn_key]['J'] = param_value
                    
                    # Calculate mutual inductance M = (Φ₀/I_c) * J for physical
                    M_values = (converter.Phi_0 / converter.I_c) * param_value
                    results['physical']['connections'][conn_key]['M'] = M_values
                except Exception as conn_err:
                    print(f"Error processing connection {full_key}: {conn_err}")
                    continue
        except Exception as e:
            print(f"Error processing parameter {full_key}: {e}")
            continue
    
    # Generate visualizations with proper separation if requested
    if create_plots and save_path:
        generate_parameter_histograms(results, save_path)
    
    # Export to CSV if save_path is provided
    if save_path:
        export_to_csv(results, save_path)
    
    return results

def generate_parameter_histograms(results: Dict, save_path: str):
    """
    Generate histograms of physical parameter distributions with proper separation.
    
    Args:
        results: Dictionary of physical parameter values with dimensionless/physical separation
        save_path: Directory to save histogram images
    """
    hist_dir = os.path.join(save_path, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    
    # Use non-interactive Agg backend to avoid GUI issues
    import matplotlib
    matplotlib.use('Agg')
    
    # Use a modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set DPI for high-resolution plots
    dpi = 150
    
    # Create parameter categories with proper separation
    param_categories = {
        'dimensionless': {
            'layer': {},
            'connection': {}
        },
        'physical': {
            'layer': {},
            'connection': {}
        }
    }
    
    # Generate histograms for dimensionless layer parameters
    for layer_id, layer_data in results['dimensionless']['layers'].items():
        # Initialize categories for this layer if they don't exist
        if layer_id not in param_categories['dimensionless']['layer']:
            param_categories['dimensionless']['layer'][layer_id] = []
        
        for param_name, param_values in layer_data.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 1:
                param_categories['dimensionless']['layer'][layer_id].append(param_name)
                
                # Create histogram for dimensionless parameter
                create_histogram(
                    param_values, 
                    f'layer_{layer_id}_{param_name}', 
                    f'Layer {layer_id} - {param_name} Distribution',
                    hist_dir,
                    dpi,
                    is_connection=False
                )
    
    # Generate histograms for physical layer parameters
    for layer_id, layer_data in results['physical']['layers'].items():
        # Initialize categories for this layer if they don't exist
        if layer_id not in param_categories['physical']['layer']:
            param_categories['physical']['layer'][layer_id] = []
        
        for param_name, param_values in layer_data.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 1:
                param_categories['physical']['layer'][layer_id].append(param_name)
                
                # Create histogram for physical parameter
                create_histogram(
                    param_values, 
                    f'layer_{layer_id}_{param_name}_physical', 
                    f'Layer {layer_id} - {param_name} Distribution (Physical)',
                    hist_dir,
                    dpi,
                    is_connection=False
                )
    
    # Generate histograms for dimensionless connection parameters
    for conn_key, conn_data in results['dimensionless']['connections'].items():
        # Initialize categories for this connection if they don't exist
        if conn_key not in param_categories['dimensionless']['connection']:
            param_categories['dimensionless']['connection'][conn_key] = []
        
        for param_name, param_values in conn_data.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 1:
                param_categories['dimensionless']['connection'][conn_key].append(param_name)
                
                # Create histogram for dimensionless parameter
                create_histogram(
                    param_values, 
                    f'connection_{conn_key}_{param_name}', 
                    f'Connection {conn_key} - {param_name} Distribution',
                    hist_dir,
                    dpi,
                    is_connection=True
                )
    
    # Generate histograms for physical connection parameters
    for conn_key, conn_data in results['physical']['connections'].items():
        # Initialize categories for this connection if they don't exist
        if conn_key not in param_categories['physical']['connection']:
            param_categories['physical']['connection'][conn_key] = []
        
        for param_name, param_values in conn_data.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 1:
                param_categories['physical']['connection'][conn_key].append(param_name)
                
                # Create histogram for physical parameter
                create_histogram(
                    param_values, 
                    f'connection_{conn_key}_{param_name}_physical', 
                    f'Connection {conn_key} - {param_name} Distribution (Physical)',
                    hist_dir,
                    dpi,
                    is_connection=True
                )
    
    # Save parameter categories as JSON for front-end organization
    try:
        with open(os.path.join(hist_dir, 'param_categories.json'), 'w') as f:
            json.dump(param_categories, f)
    except Exception as e:
        print(f"Warning: Could not save parameter categories: {e}")


def create_histogram(param_values, filename, title, save_dir, dpi=150, is_connection=False):
    """
    Create a histogram for a parameter.
    
    Args:
        param_values: The parameter values to plot
        filename: The filename to save as (without extension)
        title: The plot title
        save_dir: Directory to save the plot
        dpi: Resolution of the output plot
        is_connection: Whether this is a connection parameter (affects filtering)
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    # Flatten and process values
    flat_values = param_values.flatten()
    
    # For connections, filter out zeros
    if is_connection:
        nonzero_values = flat_values[flat_values != 0]
        if nonzero_values.size > 0:
            flat_values = nonzero_values
    
    if flat_values.size > 0:
        # Calculate optimal number of bins
        q75, q25 = np.percentile(flat_values, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (flat_values.size ** (1/3)) if iqr > 0 else 'auto'
        bins = int(np.ceil((flat_values.max() - flat_values.min()) / bin_width)) if isinstance(bin_width, float) else 30
        bins = min(max(10, bins), 50)  # Limit bins to reasonable range
        
        # Plot histogram with appropriate color
        color = '#3498db' if not is_connection else '#e74c3c'
        n, bins, patches = ax.hist(flat_values, bins=bins, alpha=0.8, 
                             color=color, edgecolor='black', linewidth=0.5)
        
        # Add a kernel density estimate
        try:
            from scipy import stats
            kde_xs = np.linspace(flat_values.min(), flat_values.max(), 200)
            kde = stats.gaussian_kde(flat_values)
            kde_ys = kde(kde_xs)
            # Scale the density to match histogram height
            scale_factor = (n.max() / kde_ys.max()) if kde_ys.max() > 0 else 1
            ax.plot(kde_xs, kde_ys * scale_factor, 'r-' if not is_connection else 'b-', 
                   linewidth=2, label='Density Estimate')
            ax.legend()
        except (ImportError, np.linalg.LinAlgError):
            pass  # Skip density estimate if scipy not available or singular matrix
        
        # Add statistics as text box
        mean = np.mean(flat_values)
        median = np.median(flat_values)
        std = np.std(flat_values)
        min_val = np.min(flat_values)
        max_val = np.max(flat_values)
        
        # Format nicely for scientific notation
        def format_val(val):
            if abs(val) < 0.001 or abs(val) > 1000:
                return f"{val:.2e}"
            else:
                return f"{val:.4f}"
        
        stats_text = (f"Mean: {format_val(mean)}\n"
                     f"Median: {format_val(median)}\n"
                     f"Std Dev: {format_val(std)}\n"  # Make sure std dev is displayed
                     f"Min: {format_val(min_val)}\n"
                     f"Max: {format_val(max_val)}")
        
        if is_connection:
            # Add sparsity for connections
            all_values = param_values.flatten()
            sparsity = 1.0 - (flat_values.size / all_values.size) if all_values.size > 0 else 0
            stats_text += f"\nSparsity: {sparsity:.1%}"
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        param_label = filename.split('_')[-1].replace('_physical', ' (Physical)')
        ax.set_xlabel(f'{param_label} Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout
        fig.tight_layout()
        
        # Save figure
        fig.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=dpi)
        plt.close(fig)

def export_to_csv(results: Dict, save_path: str):
    """
    Export physical parameter values to CSV files with proper separation.
    
    Args:
        results: Dictionary of parameter values
        save_path: Directory to save CSV files
    """
    csv_dir = os.path.join(save_path, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    if PANDAS_AVAILABLE:
        # Use pandas for CSV export if available
        # Export base parameters
        pd.DataFrame([results['base_parameters']]).to_csv(
            os.path.join(csv_dir, 'base_parameters.csv'), index=False)
        
        # Export dimensionless layer parameters
        for layer_id, layer_data in results['dimensionless']['layers'].items():
            layer_df = {}
            for param_name, param_values in layer_data.items():
                if isinstance(param_values, np.ndarray):
                    if param_values.ndim == 1:
                        for i, value in enumerate(param_values):
                            layer_df[f'{param_name}_{i}'] = [value]
                    else:
                        # For multi-dimensional arrays, store as is
                        layer_df[param_name] = [param_values]
                else:
                    layer_df[param_name] = [param_values]
            
            if layer_df:
                pd.DataFrame(layer_df).to_csv(
                    os.path.join(csv_dir, f'layer_{layer_id}_parameters.csv'), index=False)
        
        # Export physical layer parameters
        for layer_id, layer_data in results['physical']['layers'].items():
            layer_df = {}
            for param_name, param_values in layer_data.items():
                if isinstance(param_values, np.ndarray):
                    if param_values.ndim == 1:
                        for i, value in enumerate(param_values):
                            layer_df[f'{param_name}_{i}'] = [value]
                    else:
                        # For multi-dimensional arrays, store as is
                        layer_df[param_name] = [param_values]
                else:
                    layer_df[param_name] = [param_values]
            
            if layer_df:
                pd.DataFrame(layer_df).to_csv(
                    os.path.join(csv_dir, f'layer_{layer_id}_parameters_physical.csv'), index=False)
        
        # Export dimensionless connection parameters
        for conn_key, conn_data in results['dimensionless']['connections'].items():
            for param_name, param_values in conn_data.items():
                pd.DataFrame(param_values).to_csv(
                    os.path.join(csv_dir, f'connection_{conn_key}_{param_name}.csv'))
        
        # Export physical connection parameters
        for conn_key, conn_data in results['physical']['connections'].items():
            for param_name, param_values in conn_data.items():
                pd.DataFrame(param_values).to_csv(
                    os.path.join(csv_dir, f'connection_{conn_key}_{param_name}_physical.csv'))
        
        # Create a summary CSV with key statistics including both dimensionless and physical
        summary_rows = []
        
        # Add base parameters
        for param_name, value in results['base_parameters'].items():
            summary_rows.append({
                'category': 'base_parameter',
                'name': param_name,
                'min': value,
                'max': value,
                'mean': value,
                'std': 0
            })
        
        # Add dimensionless layer parameters
        for layer_id, layer_data in results['dimensionless']['layers'].items():
            for param_name, param_values in layer_data.items():
                if isinstance(param_values, np.ndarray) and param_values.size > 0:
                    flat_values = param_values.flatten()
                    summary_rows.append({
                        'category': f'layer_{layer_id}_dimensionless',
                        'name': param_name,
                        'min': np.min(flat_values),
                        'max': np.max(flat_values),
                        'mean': np.mean(flat_values),
                        'std': np.std(flat_values)
                    })
        
        # Add physical layer parameters
        for layer_id, layer_data in results['physical']['layers'].items():
            for param_name, param_values in layer_data.items():
                if isinstance(param_values, np.ndarray) and param_values.size > 0:
                    flat_values = param_values.flatten()
                    summary_rows.append({
                        'category': f'layer_{layer_id}_physical',
                        'name': param_name,
                        'min': np.min(flat_values),
                        'max': np.max(flat_values),
                        'mean': np.mean(flat_values),
                        'std': np.std(flat_values)
                    })
        
        # Add dimensionless connection parameters
        for conn_key, conn_data in results['dimensionless']['connections'].items():
            for param_name, param_values in conn_data.items():
                if isinstance(param_values, np.ndarray) and param_values.size > 0:
                    flat_values = param_values.flatten()
                    nonzero_values = flat_values[flat_values != 0] if np.any(flat_values != 0) else flat_values
                    
                    summary_rows.append({
                        'category': f'connection_{conn_key}_dimensionless',
                        'name': param_name,
                        'min': np.min(nonzero_values) if nonzero_values.size > 0 else 0,
                        'max': np.max(nonzero_values) if nonzero_values.size > 0 else 0,
                        'mean': np.mean(nonzero_values) if nonzero_values.size > 0 else 0,
                        'std': np.std(nonzero_values) if nonzero_values.size > 0 else 0,
                        'sparsity': 1.0 - (nonzero_values.size / flat_values.size) if flat_values.size > 0 else 0
                    })
        
        # Add physical connection parameters
        for conn_key, conn_data in results['physical']['connections'].items():
            for param_name, param_values in conn_data.items():
                if isinstance(param_values, np.ndarray) and param_values.size > 0:
                    flat_values = param_values.flatten()
                    nonzero_values = flat_values[flat_values != 0] if np.any(flat_values != 0) else flat_values
                    
                    summary_rows.append({
                        'category': f'connection_{conn_key}_physical',
                        'name': param_name,
                        'min': np.min(nonzero_values) if nonzero_values.size > 0 else 0,
                        'max': np.max(nonzero_values) if nonzero_values.size > 0 else 0,
                        'mean': np.mean(nonzero_values) if nonzero_values.size > 0 else 0,
                        'std': np.std(nonzero_values) if nonzero_values.size > 0 else 0,
                        'sparsity': 1.0 - (nonzero_values.size / flat_values.size) if flat_values.size > 0 else 0
                    })
        
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(csv_dir, 'parameter_summary.csv'), index=False)
    else:
        # Fallback to basic CSV export using numpy
        # Export base parameters
        with open(os.path.join(csv_dir, 'base_parameters.csv'), 'w') as f:
            f.write(','.join(results['base_parameters'].keys()) + '\n')
            f.write(','.join(str(v) for v in results['base_parameters'].values()) + '\n')
        
        # Export dimensionless layer parameters (simplified)
        for layer_id, layer_data in results['dimensionless']['layers'].items():
            with open(os.path.join(csv_dir, f'layer_{layer_id}_parameters.txt'), 'w') as f:
                for param_name, param_values in layer_data.items():
                    f.write(f"{param_name}:\n")
                    if isinstance(param_values, np.ndarray):
                        if param_values.ndim == 1:
                            for i, value in enumerate(param_values):
                                f.write(f"  {i}: {value}\n")
                        else:
                            f.write(f"  Shape: {param_values.shape}\n")
                            f.write(f"  Min: {np.min(param_values)}\n")
                            f.write(f"  Max: {np.max(param_values)}\n")
                            f.write(f"  Mean: {np.mean(param_values)}\n")
                    else:
                        f.write(f"  {param_values}\n")
        
        # Export physical layer parameters (simplified)
        for layer_id, layer_data in results['physical']['layers'].items():
            with open(os.path.join(csv_dir, f'layer_{layer_id}_parameters_physical.txt'), 'w') as f:
                for param_name, param_values in layer_data.items():
                    f.write(f"{param_name}:\n")
                    if isinstance(param_values, np.ndarray):
                        if param_values.ndim == 1:
                            for i, value in enumerate(param_values):
                                f.write(f"  {i}: {value}\n")
                        else:
                            f.write(f"  Shape: {param_values.shape}\n")
                            f.write(f"  Min: {np.min(param_values)}\n")
                            f.write(f"  Max: {np.max(param_values)}\n")
                            f.write(f"  Mean: {np.mean(param_values)}\n")
                    else:
                        f.write(f"  {param_values}\n")
        
        # Export dimensionless connection parameters (simplified)
        for conn_key, conn_data in results['dimensionless']['connections'].items():
            with open(os.path.join(csv_dir, f'connection_{conn_key}_parameters.txt'), 'w') as f:
                for param_name, param_values in conn_data.items():
                    f.write(f"{param_name}:\n")
                    f.write(f"  Shape: {param_values.shape}\n")
                    f.write(f"  Min: {np.min(param_values)}\n")
                    f.write(f"  Max: {np.max(param_values)}\n")
                    f.write(f"  Mean: {np.mean(param_values)}\n")
                    
                    # For sparse matrices, also report non-zero statistics
                    flat_values = param_values.flatten()
                    nonzero_values = flat_values[flat_values != 0]
                    if nonzero_values.size < flat_values.size:
                        f.write(f"  Non-zero count: {nonzero_values.size}/{flat_values.size}\n")
                        f.write(f"  Sparsity: {1.0 - (nonzero_values.size / flat_values.size):.4f}\n")
                        if nonzero_values.size > 0:
                            f.write(f"  Non-zero min: {np.min(nonzero_values)}\n")
                            f.write(f"  Non-zero max: {np.max(nonzero_values)}\n")
                            f.write(f"  Non-zero mean: {np.mean(nonzero_values)}\n")
        
        # Export physical connection parameters (simplified)
        for conn_key, conn_data in results['physical']['connections'].items():
            with open(os.path.join(csv_dir, f'connection_{conn_key}_parameters_physical.txt'), 'w') as f:
                for param_name, param_values in conn_data.items():
                    f.write(f"{param_name}:\n")
                    f.write(f"  Shape: {param_values.shape}\n")
                    f.write(f"  Min: {np.min(param_values)}\n")
                    f.write(f"  Max: {np.max(param_values)}\n")
                    f.write(f"  Mean: {np.mean(param_values)}\n")
                    
                    # For sparse matrices, also report non-zero statistics
                    flat_values = param_values.flatten()
                    nonzero_values = flat_values[flat_values != 0]
                    if nonzero_values.size < flat_values.size:
                        f.write(f"  Non-zero count: {nonzero_values.size}/{flat_values.size}\n")
                        f.write(f"  Sparsity: {1.0 - (nonzero_values.size / flat_values.size):.4f}\n")
                        if nonzero_values.size > 0:
                            f.write(f"  Non-zero min: {np.min(nonzero_values)}\n")
                            f.write(f"  Non-zero max: {np.max(nonzero_values)}\n")
                            f.write(f"  Non-zero mean: {np.mean(nonzero_values)}\n")



def extract_param_df(model_loaded):
    """
    Extracts all parameter values from the model's state dictionary into a DataFrame.

    Parameters:
        model_loaded (dict): The model dictionary loaded by torch.load.
    
    Returns:
        DataFrame or dict: A DataFrame with columns for node (layer), parameter name, shape, and values
                          if pandas is available, otherwise a dictionary with the same information.
    """
    # Handle different model formats
    if isinstance(model_loaded, dict):
        if 'model_state_dict' in model_loaded:
            state_dict = model_loaded['model_state_dict']
        else:
            # Check if this looks like a state dict (contains 'layers' or 'connections')
            if any(k.startswith('layers.') or k.startswith('connections.') for k in model_loaded.keys()):
                state_dict = model_loaded
            else:
                # If it doesn't look like a state dict, check again inside
                found_state_dict = False
                for key, value in model_loaded.items():
                    if isinstance(value, dict) and any(k.startswith('layers.') or k.startswith('connections.') for k in value.keys()):
                        state_dict = value
                        found_state_dict = True
                        break
                
                # If we still don't have a state dict, use the original
                if not found_state_dict:
                    state_dict = model_loaded
    else:
        # For other types, try to use it directly
        state_dict = model_loaded
    
    rows = []
    
    # If state_dict is not a dict or doesn't have items method, return empty result
    if not hasattr(state_dict, 'items'):
        print(f"Warning: Could not extract parameters from model. Expected dict, got {type(state_dict)}")
        if PANDAS_AVAILABLE:
            return pd.DataFrame(columns=["node", "parameter", "shape", "values"])
        else:
            return rows
    
    for full_key, tensor in state_dict.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            continue
            
        # Split the key to extract node information and parameter name.
        parts = full_key.split('.')
        if len(parts) > 1:
            node = '.'.join(parts[:-1])
            param_name = parts[-1]
        else:
            node = ""
            param_name = full_key
        
        # Append details
        rows.append({
            "node": node,
            "parameter": param_name,
            "shape": tuple(tensor.shape),
            "values": tensor.detach().cpu().numpy()
        })
    
    if PANDAS_AVAILABLE:
        return pd.DataFrame(rows)
    else:
        return rows