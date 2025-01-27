# FILEPATH: src/soen/utils/physical_mappings/main.py

from flask import Flask, request, jsonify, render_template, send_from_directory
from soen_conversion_utils import PhysicalConverter
import math

app = Flask(__name__)
converter = PhysicalConverter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)



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
        
        # Inductance
        if 'beta_L' in data and data['beta_L'] is not None:
            beta_L_val = float(data['beta_L'])
            result['L'] = {
                'value': converter.dimensionless_to_physical_inductance(beta_L_val),
                'latex': 'L',
                'unit': 'H'
            }
        
        # From gamma (alternative inductance representation)
        if 'gamma' in data and data['gamma'] is not None:
            gamma_val = float(data['gamma'])
            beta_L = converter.gamma_to_beta_L(gamma_val)
            result['L'] = {
                'value': converter.dimensionless_to_physical_inductance(beta_L),
                'latex': 'L',
                'unit': 'H'
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
            result['tau'] = {
                'value': converter.calculate_tau(beta_L, alpha),
                'latex': '\\tau'
            }
        
        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
