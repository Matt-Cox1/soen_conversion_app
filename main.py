from flask import Flask, request, jsonify, render_template
import math
import os

app = Flask(__name__)


class PhysicalConverter:

    def __init__(self,
                 I_c=100e-6,
                 R=2.56,
                 L=1e-9,
                 C=0.151e-12,
                 Phi_0=2.07e-15,
                 R_leak=1.0):
        self.I_c = I_c
        self.R = R
        self.L = L
        self.C = C
        self.Phi_0 = Phi_0
        self.R_leak = R_leak
        self.omega_c = self.calculate_omega_c()
        self.alpha = self.calculate_alpha()

    def calculate_omega_c(self):
        return (2 * math.pi * self.R * self.I_c) / self.Phi_0

    def calculate_alpha(self):
        return self.R_leak / self.R

    def convert_abstract_to_physical_current(self, i):
        return float(i) * self.I_c

    def convert_physical_to_abstract_current(self, I):
        return float(I) / self.I_c

    def convert_abstract_to_physical_flux(self, phi):
        return float(phi) * self.Phi_0

    def convert_physical_to_abstract_flux(self, flux):
        return float(flux) / self.Phi_0

    def convert_abstract_to_physical_mutual_inductance(self, J):
        return (float(J) * self.Phi_0) / self.I_c

    def convert_physical_to_abstract_mutual_inductance(self, M):
        return (float(M) * self.I_c) / self.Phi_0

    def convert_physical_to_abstract_time(self, t):
        return float(t) * self.omega_c

    def convert_abstract_to_physical_time(self, t_prime):
        return float(t_prime) / self.omega_c


converter = PhysicalConverter()


@app.route('/update_constants', methods=['POST'])
def update_constants():
    data = request.json
    try:
        converter.I_c = float(data.get('I_c', converter.I_c))
        converter.R = float(data.get('R', converter.R))
        converter.L = float(data.get('L', converter.L))
        converter.C = float(data.get('C', converter.C))
        converter.Phi_0 = float(data.get('Phi_0', converter.Phi_0))
        converter.R_leak = float(data.get('R_leak', converter.R_leak))

        # Recalculate dependent parameters
        converter.omega_c = converter.calculate_omega_c()
        converter.alpha = converter.calculate_alpha()

        return jsonify({
            'I_c': converter.I_c,
            'R': converter.R,
            'L': converter.L,
            'C': converter.C,
            'Phi_0': converter.Phi_0,
            'R_leak': converter.R_leak,
            'omega_c': converter.omega_c,
            'alpha': converter.alpha
        })
    except ValueError as e:
        return jsonify({'error': 'Invalid input values'}), 400


# Route to serve the HTML template
@app.route('/')
def index():
    return render_template('index.html')


# API endpoint for converting to physical units
@app.route('/convert_to_physical', methods=['POST'])
def convert_to_physical():
    data = request.json
    i_val = data.get('i')
    phi_val = data.get('phi')
    J_val = data.get('J')
    t_prime_val = data.get('t_prime')

    result = {}
    if i_val is not None:
        result['I'] = converter.convert_abstract_to_physical_current(i_val)
    if phi_val is not None:
        result['flux'] = converter.convert_abstract_to_physical_flux(phi_val)
    if J_val is not None:
        result['M'] = converter.convert_abstract_to_physical_mutual_inductance(
            J_val)
    if t_prime_val is not None:
        result['t'] = converter.convert_abstract_to_physical_time(t_prime_val)

    return jsonify(result)


# API endpoint for converting to abstract units
@app.route('/convert_to_abstract', methods=['POST'])
def convert_to_abstract():
    data = request.json
    I_val = data.get('I')
    flux_val = data.get('flux')
    M_val = data.get('M')
    t_val = data.get('t')

    result = {}
    if I_val is not None:
        result['i'] = converter.convert_physical_to_abstract_current(I_val)
    if flux_val is not None:
        result['phi'] = converter.convert_physical_to_abstract_flux(flux_val)
    if M_val is not None:
        result['J'] = converter.convert_physical_to_abstract_mutual_inductance(
            M_val)
    if t_val is not None:
        result['t_prime'] = converter.convert_physical_to_abstract_time(t_val)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
