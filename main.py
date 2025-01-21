# 0.3


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

@app.route('/update_constants', methods=['POST'])
def update_constants():
    data = request.json
    try:
        # Get input parameters
        I_c = data.get('I_c')
        r_leak = data.get('r_leak')
        L = data.get('L')
        gamma_c = data.get('gamma_c')
        beta_c = data.get('beta_c')

        # Validate inputs
        if any(x is not None and float(x) <= 0 for x in [I_c, r_leak, L, gamma_c, beta_c]):
            return jsonify({'error': 'All parameters must be positive'}), 400

        # Update base parameters
        if I_c is not None:
            converter.I_c = float(I_c)
        if r_leak is not None:
            converter.r_leak = float(r_leak)
        if L is not None:
            converter.L = float(L)
        if gamma_c is not None:
            converter.gamma_c = float(gamma_c)
        if beta_c is not None:
            converter.beta_c = float(beta_c)

        # Calculate derived parameters
        c_j = converter.gamma_c * converter.I_c
        
        # Calculate r_jj using beta_c
        r_jj = math.sqrt((converter.beta_c * converter.Phi_0) / (2 * math.pi * c_j * converter.I_c))
        
        # Update converter parameters
        converter.c_j = c_j
        converter.r_jj = r_jj
        
        # Calculate additional derived parameters
        tau_0 = converter.Phi_0 / (2 * math.pi * converter.I_c * r_jj)
        V_j = converter.I_c * r_jj
        omega_c = (2 * math.pi * converter.I_c * r_jj) / converter.Phi_0
        omega_p = math.sqrt((2 * math.pi * converter.I_c) / (converter.Phi_0 * c_j))
        beta_L = (2 * math.pi * converter.I_c * converter.L) / converter.Phi_0
        alpha = converter.r_leak / r_jj
        gamma = 1 / beta_L if beta_L != 0 else float('inf')
        tau = beta_L / alpha if alpha != 0 else float('inf')

        return jsonify({
            'I_c': converter.I_c,
            'r_jj': r_jj,
            'r_leak': converter.r_leak,
            'L': converter.L,
            'gamma_c': converter.gamma_c,
            'beta_c': converter.beta_c,
            'c_j': c_j,
            'tau_0': tau_0,
            'V_j': V_j,
            'omega_c': omega_c,
            'omega_p': omega_p,
            'beta_L': beta_L,
            'alpha': alpha,
            'gamma': gamma,
            'tau': tau
        })
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400

@app.route('/convert_to_physical', methods=['POST'])
def convert_to_physical():
    data = request.json
    result = {}
    
    if 'i' in data and data['i'] is not None:
        result['I'] = converter.dimensionless_to_physical_current(data['i'])
    if 'phi' in data and data['phi'] is not None:
        result['Phi'] = converter.dimensionless_to_physical_flux(data['phi'])
    if 'J' in data and data['J'] is not None:
        result['M'] = converter.dimensionless_to_physical_inductance(data['J'])
    if 't_prime' in data and data['t_prime'] is not None:
        result['t'] = converter.dimensionless_to_physical_time(data['t_prime'])
    if 'g_fq' in data and data['g_fq'] is not None:
        result['G_fq'] = converter.dimensionless_to_physical_fq_rate(data['g_fq'])
    if 'beta_L' in data and data['beta_L'] is not None:
        result['L'] = converter.dimensionless_to_physical_inductance(data['beta_L'])

    return jsonify(result)

@app.route('/convert_to_dimensionless', methods=['POST'])
def convert_to_dimensionless():
    data = request.json
    result = {}
    
    if 'I' in data and data['I'] is not None:
        result['i'] = converter.physical_to_dimensionless_current(data['I'])
    if 'Phi' in data and data['Phi'] is not None:
        result['phi'] = converter.physical_to_dimensionless_flux(data['Phi'])
    if 'M' in data and data['M'] is not None:
        result['J'] = converter.physical_to_dimensionless_inductance(data['M'])
    if 't' in data and data['t'] is not None:
        result['t_prime'] = converter.physical_to_dimensionless_time(data['t'])
    if 'G_fq' in data and data['G_fq'] is not None:
        result['g_fq'] = converter.physical_to_dimensionless_fq_rate(data['G_fq'])
    if 'L' in data and data['L'] is not None:
        result['beta_L'] = converter.physical_to_dimensionless_inductance(data['L'])

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)




