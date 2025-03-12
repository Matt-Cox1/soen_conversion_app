# FILEPATH: src/soen_sim_v2/utils/physical_mappings/main.py

import os
import json
import tempfile
import time
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import numpy as np
import math

# Add necessary paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soen_sim_v2.utils.physical_mappings.soen_conversion_utils import PhysicalConverter
from soen_sim_v2.utils.physical_mappings.convert_model import ModelConverter

app = Flask(__name__)
converter = PhysicalConverter()
model_converter = ModelConverter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_converter')
def model_converter_page():
    return render_template('model_converter.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/model_plots/<model_id>/<path:path>')
def send_model_plots(model_id, path):
    """Serve model plots from the temporary directory"""
    # Path to the original plot
    plot_path = os.path.join(model_converter.temp_dir, model_id, path)
    
    # Check if it's a request for a histogram
    if path.startswith('histograms/') and path.endswith('.png'):
        # Check if the file exists, if not return 404
        if not os.path.exists(plot_path):
            return "Plot not found", 404
            
        # Check for timestamp parameter - if present, regenerate the plot
        has_timestamp = 'ts' in request.args
        
        # Only regenerate plot if timestamp is in the URL (forces reload)
        if has_timestamp:
            try:
                # Parse path to figure out what we're plotting
                plot_parts = os.path.basename(path).replace('.png', '').split('_')
                
                if len(plot_parts) >= 3:
                    item_type = plot_parts[0]  # 'layer' or 'connection'
                    item_id = plot_parts[1]    # The layer/connection ID
                    param_name = '_'.join(plot_parts[2:])  # Parameter name (might contain underscores)
                    
                    # Get the data directory to load the parameter data
                    data_dir = os.path.join(model_converter.temp_dir, model_id)
                    param_file = os.path.join(data_dir, 'csv', f'{item_type}_{item_id}_parameters.csv')
                    
                    if os.path.exists(param_file):
                        # Use pandas to read the parameter data if available
                        try:
                            import pandas as pd
                            df = pd.read_csv(param_file)
                        except (ImportError, Exception) as e:
                            print(f"Error reading parameter data: {e}")
                            return send_file(plot_path)  # Fallback to original plot
                        
                        # Find the column(s) containing this parameter
                        # Be more selective in column matching - don't just use substring matching
                        param_cols = []
                        for col in df.columns:
                            # Exact match
                            if col == param_name:
                                param_cols.append(col)
                            # Parameter with suffix (e.g. bias_current_1)
                            elif col.startswith(param_name + "_") and not col.endswith("_physical"):
                                param_cols.append(col)
                            # Physical parameter (e.g. bias_current_physical)
                            elif param_name.endswith("_physical") and col == param_name:
                                param_cols.append(col)
                                
                        # print(f"Parameter columns for '{param_name}': {param_cols} (out of {list(df.columns)})")
                        
                        if param_cols:
                            # Collect all values from matching columns
                            import numpy as np
                            import matplotlib.pyplot as plt
                            from io import BytesIO
                            
                            # Use a non-interactive backend
                            plt.switch_backend('agg')
                            
                            # Close any existing figures to reset matplotlib's state
                            plt.close('all')
                            
                            # Clear matplotlib cache
                            plt.rcdefaults()
                            plt.style.use('seaborn-v0_8-darkgrid')
                            
                            # Create completely new figure and axis
                            fig = plt.figure(figsize=(10, 6), dpi=150, clear=True)
                            ax = fig.add_subplot(111)
                            
                            # Get the values to plot
                            values = []
                            
                            # Debug print to see what's happening
                            print(f"Generating plot for {param_name}, found columns: {param_cols}")
                            
                            # For physical parameters, ONLY use the exact physical parameter
                            if param_name.endswith('_physical'):
                                if param_name in df.columns:
                                    print(f"Using exact physical parameter match: {param_name}")
                                    if isinstance(df[param_name].iloc[0], str):
                                        # Try to convert string representation of array to array
                                        try:
                                            val = np.array(eval(df[param_name].iloc[0]))
                                            if isinstance(val, np.ndarray):
                                                values.extend(val.flatten())
                                        except:
                                            print(f"Error parsing array in column {param_name}")
                                    else:
                                        values.extend(df[param_name].dropna().to_numpy())
                            
                            # For dimensionless parameters, make sure we DON'T use the physical version
                            else:
                                exact_match = False
                                
                                # First try exact column match
                                if param_name in df.columns:
                                    print(f"Using exact parameter match: {param_name}")
                                    exact_match = True
                                    if isinstance(df[param_name].iloc[0], str):
                                        # Try to convert string representation of array to array
                                        try:
                                            val = np.array(eval(df[param_name].iloc[0]))
                                            if isinstance(val, np.ndarray):
                                                values.extend(val.flatten())
                                        except:
                                            print(f"Error parsing array in column {param_name}")
                                    else:
                                        values.extend(df[param_name].dropna().to_numpy())
                                
                                # If there was no exact match, look for columns that contain this parameter
                                # but explicitly exclude any that end with _physical
                                if not exact_match:
                                    for col in param_cols:
                                        if col != param_name and not col.endswith('_physical'):
                                            # print(f"Using related parameter match: {col}")
                                            if isinstance(df[col].iloc[0], str):
                                                # Try to convert string representation of array to array
                                                try:
                                                    val = np.array(eval(df[col].iloc[0]))
                                                    if isinstance(val, np.ndarray):
                                                        values.extend(val.flatten())
                                                except:
                                                    print(f"Error parsing array in column {col}")
                                            else:
                                                values.extend(df[col].dropna().to_numpy())
                            
                            values = np.array(values)
                            
                            # For connections, filter out zeros
                            if item_type == 'connection':
                                nonzero_values = values[values != 0]
                                if nonzero_values.size > 0:
                                    values = nonzero_values
                            
                            if values.size > 0:
                                # Calculate optimal bins
                                q75, q25 = np.percentile(values, [75, 25])
                                iqr = q75 - q25
                                bin_width = 2 * iqr / (values.size ** (1/3)) if iqr > 0 else 'auto'
                                bins = int(np.ceil((values.max() - values.min()) / bin_width)) if isinstance(bin_width, float) else 30
                                bins = min(max(10, bins), 50)  # Limit bins to reasonable range
                                
                                # Plot histogram
                                color = '#3498db' if item_type == 'layer' else '#e74c3c'
                                n, bins, patches = ax.hist(values, bins=bins, alpha=0.8, 
                                                   color=color, edgecolor='black', linewidth=0.5)
                                
                                # Add a kernel density estimate
                                try:
                                    from scipy import stats
                                    kde_xs = np.linspace(values.min(), values.max(), 200)
                                    kde = stats.gaussian_kde(values)
                                    kde_ys = kde(kde_xs)
                                    # Scale the density to match histogram height
                                    scale_factor = (n.max() / kde_ys.max()) if kde_ys.max() > 0 else 1
                                    ax.plot(kde_xs, kde_ys * scale_factor, 'r-' if item_type == 'layer' else 'b-', 
                                           linewidth=2, label='Density Estimate')
                                    ax.legend()
                                except (ImportError, np.linalg.LinAlgError):
                                    pass  # Skip density estimate if scipy not available or singular matrix
                                
                                # Add statistics as text box
                                mean = np.mean(values)
                                median = np.median(values)
                                std = np.std(values)
                                min_val = np.min(values)
                                max_val = np.max(values)
                                
                                # Format nicely for scientific notation
                                def format_val(val):
                                    if abs(val) < 0.001 or abs(val) > 1000:
                                        return f"{val:.2e}"
                                    else:
                                        return f"{val:.4f}"
                                
                                stats_text = (f"Mean: {format_val(mean)}\n"
                                            f"Median: {format_val(median)}\n"
                                            f"Std Dev: {format_val(std)}\n"
                                            f"Min: {format_val(min_val)}\n"
                                            f"Max: {format_val(max_val)}")
                                
                                if item_type == 'connection':
                                    # Add sparsity for connections
                                    all_values = np.array(values)  # This may include zeros depending on filter above
                                    sparsity = 1.0 - (values.size / all_values.size) if all_values.size > 0 else 0
                                    stats_text += f"\nSparsity: {sparsity:.1%}"
                                
                                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                                       verticalalignment='top', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                                
                                # Add title and labels
                                type_name = "Layer" if item_type == 'layer' else "Connection"
                                ax.set_title(f'{type_name} {item_id} - {param_name} Distribution', fontsize=14, fontweight='bold')
                                ax.set_xlabel(f'{param_name} Value', fontsize=12)
                                ax.set_ylabel('Frequency', fontsize=12)
                                
                                # Add grid
                                ax.grid(True, alpha=0.3, linestyle='--')
                                
                                # Tight layout
                                fig.tight_layout()
                                
                                # Create a unique filename for this plot request
                                import uuid
                                unique_id = str(uuid.uuid4())
                                timestamp = str(int(time.time()))
                                unique_path = os.path.join(model_converter.temp_dir, model_id, f"temp_{unique_id}_{timestamp}.png")
                                
                                # Check if the unique file already exists (shouldn't happen)
                                if os.path.exists(unique_path):
                                    os.remove(unique_path)
                                
                                # Save to the unique file
                                fig.savefig(unique_path, format='png', dpi=150)
                                plt.close('all')  # Close all figures
                                
                                # Clear matplotlib's memory
                                plt.clf()
                                plt.cla()
                                
                                # print(f"Successfully generated plot to {unique_path} with {len(values)} values. Min={np.min(values)}, Max={np.max(values)}")
                                
                                # Send the file with strict cache control
                                response = send_file(unique_path, mimetype='image/png')
                                response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                                response.headers['Pragma'] = 'no-cache'
                                response.headers['Expires'] = '0'
                                
                                # Ensure the file gets removed after it's sent
                                @response.call_on_close
                                def cleanup():
                                    try:
                                        if os.path.exists(unique_path):
                                            os.remove(unique_path)
                                            print(f"Removed temporary plot file: {unique_path}")
                                    except Exception as e:
                                        print(f"Failed to remove temp file {unique_path}: {e}")
                                        
                                return response
            except Exception as e:
                print(f"Error regenerating plot: {e}")
                # If any error occurs, fall back to the original file
                pass
    
    # If we get here, just send the original file with cache control headers
    if os.path.exists(plot_path):
        response = send_file(plot_path)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return "Plot not found", 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download exported model data"""
    file_path = os.path.join(model_converter.temp_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

@app.route('/convert_model', methods=['POST'])
def convert_model():
    """Convert a SOEN model to physical parameters"""
    try:
        # Handle file upload
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not model_file.filename.endswith('.pth'):
            return jsonify({'error': 'Only .pth files are supported'}), 400
        
        # Save uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, model_file.filename)
        model_file.save(temp_model_path)
        
        # Get parameters from form data
        try:
            I_c = float(request.form.get('I_c', '100e-6').replace('e-', 'e-'))
            gamma_c = float(request.form.get('gamma_c', '1.5e-9').replace('e-', 'e-'))
            beta_c = float(request.form.get('beta_c', '0.3'))
        except ValueError as ve:
            return jsonify({'error': f'Invalid parameter values: {str(ve)}'}), 400
            
        create_plots = request.form.get('create_plots', 'true').lower() in ['true', '1', 'on']
        
        # Print debug information
        print(f"Processing model with parameters: I_c={I_c}, gamma_c={gamma_c}, beta_c={beta_c}, create_plots={create_plots}")
        
        try:
            # Convert the model
            try:
                result = model_converter.convert_model(
                    model_path=temp_model_path,
                    I_c=I_c,
                    gamma_c=gamma_c,
                    beta_c=beta_c,
                    create_plots=create_plots
                )
                
                # Clean up temporary file
                os.unlink(temp_model_path)
                os.rmdir(temp_dir)
                
                # Use custom encoder to handle numpy arrays
                return app.response_class(
                    response=json.dumps(result, cls=NumpyJSONEncoder),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                # If conversion fails, try one last technique by loading just the weights directly
                print(f"Standard conversion failed, attempting manual extraction: {str(e)}")
                
                try:
                    import torch
                    
                    # Try to load the model with pickle safety disabled
                    model_data = torch.load(
                        temp_model_path,
                        map_location=torch.device('cpu'),
                        weights_only=False,
                        pickle_module=None  # Disable pickle safety checks
                    )
                    
                    # Extract any state dict information found
                    if hasattr(model_data, 'state_dict'):
                        # If it's a model instance, get its state dict
                        print("Found model instance with state_dict method")
                        state_dict = model_data.state_dict()
                        model_data = {'model_state_dict': state_dict}
                    
                    # Try the conversion again with our manually extracted data
                    result = model_converter.convert_model(
                        model_path=temp_model_path,  # Not used directly since we're passing the loaded model
                        I_c=I_c,
                        gamma_c=gamma_c,
                        beta_c=beta_c,
                        create_plots=create_plots,
                        model_data=model_data  # Pass extracted data directly
                    )
                    
                    # Clean up temporary file
                    os.unlink(temp_model_path)
                    os.rmdir(temp_dir)
                    
                    # Use custom encoder to handle numpy arrays
                    return app.response_class(
                        response=json.dumps(result, cls=NumpyJSONEncoder),
                        status=200,
                        mimetype='application/json'
                    )
                except Exception as manual_e:
                    print(f"Manual extraction also failed: {str(manual_e)}")
                    # Both methods failed, return the original error
                    raise e
        except Exception as conv_e:
            print(f"Error during model conversion: {str(conv_e)}")
            return jsonify({'error': f'Model conversion error: {str(conv_e)}'}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/export_model_data', methods=['POST'])
def export_model_data():
    """Export model data according to user selections"""
    try:
        data = request.json
        model_id = data.get('modelId')
        export_format = data.get('format', 'csv')
        selections = data.get('selections', ['summary', 'layers', 'connections'])
        
        if not model_id:
            return jsonify({'error': 'No model ID provided'}), 400
        
        # Export the data
        result = model_converter.export_model_data(
            model_id=model_id,
            export_format=export_format,
            selections=selections
        )
        
        # Use custom encoder to handle numpy arrays
        return app.response_class(
            response=json.dumps(result, cls=NumpyJSONEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        print(f"Error during data export: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_constants', methods=['GET'])
def get_constants():
    """Return fundamental physical constants used in calculations"""
    from scipy.constants import h, e
    
    constants = {
        'Phi_0': {'value': converter.Phi_0, 'latex': '\\Phi_0', 'description': 'Magnetic flux quantum (h/2e)', 'unit': 'Wb'},
        'h': {'value': h, 'latex': 'h', 'description': 'Planck constant', 'unit': 'J⋅s'},
        'e': {'value': e, 'latex': 'e', 'description': 'Elementary charge', 'unit': 'C'},
        'h_over_2e': {'value': h/(2*e), 'latex': 'h/2e', 'description': 'Flux quantum calculation', 'unit': 'Wb'}
    }
    
    return jsonify(constants)

@app.route('/update_base_parameters', methods=['POST'])
def update_base_parameters():
    """Update base physical parameters and return all derived values"""
    data = request.json
    try:
        # Get and validate input parameters
        I_c = data.get('I_c')
        gamma_c = data.get('gamma_c')
        beta_c = data.get('beta_c')

        if any(x is not None and float(x) <= 0 for x in [I_c, gamma_c, beta_c]):
            return jsonify({'error': 'All parameters must be positive'}), 400

        # Update converter parameters
        if I_c is not None:
            converter.I_c = float(I_c)
        if gamma_c is not None:
            converter.gamma_c = float(gamma_c)
        if beta_c is not None:
            converter.beta_c = float(beta_c)

        # Get updated parameters including derived ones
        params = converter.get_base_parameters()
        
        # Format for display with MathJax
        return jsonify({
            'I_c': {'value': params['I_c'], 'latex': 'I_c'},
            'gamma_c': {'value': params['gamma_c'], 'latex': '\\gamma_c'},
            'beta_c': {'value': params['beta_c'], 'latex': '\\beta_c'},
            'c_j': {'value': params['c_j'], 'latex': 'c_j'},
            'r_jj': {'value': params['r_jj'], 'latex': 'r_{jj}'},
            'omega_c': {'value': params['omega_c'], 'latex': '\\omega_c'},
            'omega_p': {'value': params['omega_p'], 'latex': '\\omega_p'},
            'tau_0': {'value': params['tau_0'], 'latex': '\\tau_0'},
            'V_j': {'value': params['V_j'], 'latex': 'V_j'}
        })
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400

@app.route('/convert_to_physical', methods=['POST'])
def convert_to_physical():
    """Convert dimensionless quantities to physical units"""
    data = request.json
    result = {}
    
    try:
        # Current
        if 'i' in data and data['i'] is not None:
            i_val = float(data['i'])
            result['I'] = {
                'value': converter.dimensionless_to_physical_current(i_val),
                'latex': 'I',
                'unit': 'A'
            }
        
        # Flux
        if 'phi' in data and data['phi'] is not None:
            phi_val = float(data['phi'])
            result['Phi'] = {
                'value': converter.dimensionless_to_physical_flux(phi_val),
                'latex': '\\Phi',
                'unit': 'Wb'
            }
        
        # Inductance from beta_L
        if 'beta_L' in data and data['beta_L'] is not None:
            beta_L_val = float(data['beta_L'])
            result['L'] = {
                'value': converter.dimensionless_to_physical_inductance(beta_L_val),
                'latex': 'L',
                'unit': 'H'
            }
        
        # From gamma_plus (alternative inductance representation)
        if 'gamma_plus' in data and data['gamma_plus'] is not None:
            gamma_plus_val = float(data['gamma_plus'])
            beta_L = converter.gamma_to_beta_L(gamma_plus_val)
            result['L'] = {
                'value': converter.dimensionless_to_physical_inductance(beta_L),
                'latex': 'L',
                'unit': 'H'
            }
            
        # From gamma_minus and alpha (alternative time constant calculation)
        if 'gamma_minus' in data and data['gamma_minus'] is not None:
            gamma_minus_val = float(data['gamma_minus'])
            tau = converter.gamma_minus_to_tau(gamma_minus_val)
            result['time_constant'] = {
                'value': tau / converter.omega_c,  # convert dimensionless tau to physical time
                'latex': '\\tau',
                'unit': 's'
            }
            
            # Calculate r_leak if we have beta_L (gamma_minus = alpha/beta_L)
            if 'beta_L' in data and data['beta_L'] is not None:
                beta_L_val = float(data['beta_L'])
                alpha = gamma_minus_val * beta_L_val  # alpha = gamma_minus * beta_L
                result['r_leak'] = {
                    'value': converter.dimensionless_to_physical_resistance(alpha),
                    'latex': 'r_{\\text{leak}}',
                    'unit': 'Ω'
                }
            elif 'gamma_plus' in data and data['gamma_plus'] is not None:
                gamma_plus_val = float(data['gamma_plus'])
                beta_L = converter.gamma_to_beta_L(gamma_plus_val)
                alpha = gamma_minus_val * beta_L  # alpha = gamma_minus * beta_L
                result['r_leak'] = {
                    'value': converter.dimensionless_to_physical_resistance(alpha),
                    'latex': 'r_{\\text{leak}}',
                    'unit': 'Ω'
                }
        
        # Time
        if 't_prime' in data and data['t_prime'] is not None:
            t_prime_val = float(data['t_prime'])
            result['t'] = {
                'value': converter.dimensionless_to_physical_time(t_prime_val),
                'latex': 't',
                'unit': 's'
            }
        
        # Resistance
        if 'alpha' in data and data['alpha'] is not None:
            alpha_val = float(data['alpha'])
            result['r_leak'] = {
                'value': converter.dimensionless_to_physical_resistance(alpha_val),
                'latex': 'r_{\\text{leak}}',
                'unit': 'Ω'
            }
        
        # Flux quantum rate
        if 'g_fq' in data and data['g_fq'] is not None:
            g_fq_val = float(data['g_fq'])
            result['G_fq'] = {
                'value': converter.dimensionless_to_physical_fq_rate(g_fq_val),
                'latex': 'G_{fq}',
                'unit': 'Hz'
            }

        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Conversion error: {str(e)}'}), 400

@app.route('/convert_to_dimensionless', methods=['POST'])
def convert_to_dimensionless():
    """Convert physical quantities to dimensionless units"""
    data = request.json
    result = {}
    
    try:
        # Current
        if 'I' in data and data['I'] is not None:
            I_val = float(data['I'])
            result['i'] = {
                'value': converter.physical_to_dimensionless_current(I_val),
                'latex': 'i'
            }
        
        # Flux
        if 'Phi' in data and data['Phi'] is not None:
            Phi_val = float(data['Phi'])
            result['phi'] = {
                'value': converter.physical_to_dimensionless_flux(Phi_val),
                'latex': '\\phi'
            }
        
        # Inductance (returns both beta_L and gamma)
        if 'L' in data and data['L'] is not None:
            L_val = float(data['L'])
            beta_L = converter.physical_to_dimensionless_inductance(L_val)
            result['beta_L'] = {
                'value': beta_L,
                'latex': '\\beta_L'
            }
            result['gamma'] = {
                'value': converter.beta_L_to_gamma(beta_L),
                'latex': '\\gamma'
            }
        
        # Time
        if 't' in data and data['t'] is not None:
            t_val = float(data['t'])
            result['t_prime'] = {
                'value': converter.physical_to_dimensionless_time(t_val),
                'latex': "t'"
            }
        
        # Resistance
        if 'r_leak' in data and data['r_leak'] is not None:
            r_leak_val = float(data['r_leak'])
            result['alpha'] = {
                'value': converter.physical_to_dimensionless_resistance(r_leak_val),
                'latex': '\\alpha'
            }
        
        # Flux quantum rate
        if 'G_fq' in data and data['G_fq'] is not None:
            G_fq_val = float(data['G_fq'])
            result['g_fq'] = {
                'value': converter.physical_to_dimensionless_fq_rate(G_fq_val),
                'latex': 'g_{fq}'
            }

        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Conversion error: {str(e)}'}), 400

@app.route('/calculate_derived', methods=['POST'])
def calculate_derived():
    """Calculate derived dimensionless parameters"""
    data = request.json
    result = {}
    
    try:
        # Calculate tau if we have both beta_L and alpha
        if 'beta_L' in data and 'alpha' in data:
            beta_L = float(data['beta_L'])
            alpha = float(data['alpha'])
            tau = converter.calculate_tau(beta_L, alpha)
            result['tau'] = {
                'value': tau,
                'latex': '\\tau'
            }
            # Also calculate gamma_minus from tau
            result['gamma_minus'] = {
                'value': converter.tau_to_gamma_minus(tau),
                'latex': '\\gamma_-'
            }
        
        # Calculate beta_L from gamma_plus
        if 'gamma_plus' in data:
            gamma_plus = float(data['gamma_plus'])
            beta_L = converter.gamma_plus_to_beta_L(gamma_plus)
            result['beta_L'] = {
                'value': beta_L,
                'latex': '\\beta_L'
            }
            
            # If alpha is available, calculate tau and gamma_minus
            if 'alpha' in data:
                alpha = float(data['alpha'])
                tau = converter.calculate_tau(beta_L, alpha)
                result['tau'] = {
                    'value': tau,
                    'latex': '\\tau'
                }
                result['gamma_minus'] = {
                    'value': converter.tau_to_gamma_minus(tau),
                    'latex': '\\gamma_-'
                }
        
        # Calculate alpha from gamma_plus and gamma_minus
        if 'gamma_plus' in data and 'gamma_minus' in data:
            gamma_plus = float(data['gamma_plus'])
            gamma_minus = float(data['gamma_minus'])
            alpha_val = converter.gamma_plus_gamma_minus_to_alpha(gamma_plus, gamma_minus)
            result['alpha'] = {
                'value': alpha_val,
                'latex': '\\alpha'
            }
            
            # Also calculate r_leak
            result['r_leak'] = {
                'value': converter.dimensionless_to_physical_resistance(alpha_val),
                'latex': 'r_{\\text{leak}}',
                'unit': 'Ω'
            }
            
        # Calculate alpha from gamma_minus and beta_L
        if 'gamma_minus' in data and 'beta_L' in data and 'alpha' not in result:
            gamma_minus = float(data['gamma_minus'])
            beta_L = float(data['beta_L']) 
            alpha_val = converter.gamma_minus_to_alpha_beta_L(gamma_minus, beta_L)
            result['alpha'] = {
                'value': alpha_val,
                'latex': '\\alpha'
            }
            
            # Also calculate r_leak for display
            result['r_leak'] = {
                'value': converter.dimensionless_to_physical_resistance(alpha_val),
                'latex': 'r_{\\text{leak}}',
                'unit': 'Ω'
            }
            
        # Calculate gamma_plus from beta_L
        if 'beta_L' in data and 'gamma_plus' not in data and 'gamma_plus' not in result:
            beta_L = float(data['beta_L'])
            result['gamma_plus'] = {
                'value': converter.beta_L_to_gamma_plus(beta_L),
                'latex': '\\gamma_+'
            }
            
        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)