from flask import Flask, request, jsonify, render_template
import math

class PhenomenologicalConverter:
    def __init__(self,
                 I_c=100e-6,    # Critical current [A]
                 r_jj=2.56,     # Shunt resistance [Ω]
                 r_leak=1.0,    # Leak resistance [Ω]
                 L=1e-9,        # Inductance [H]
                 Phi_0=2.07e-15): # Flux quantum [Wb]
        
        # Store base physical constants
        self.I_c = I_c
        self.r_jj = r_jj
        self.r_leak = r_leak
        self.L = L
        self.Phi_0 = Phi_0
        
        # Calculate derived quantities
        self.omega_c = self.calculate_omega_c()  # Characteristic frequency
        self.alpha = self.calculate_alpha()      # Resistance ratio
        self.beta = self.calculate_beta()        # Dimensionless inductance
        self.gamma = self.calculate_gamma()      # Inverse dimensionless inductance
        self.tau = self.calculate_tau()          # Dimensionless time constant

    # Derived quantities calculations
    def calculate_omega_c(self):
        return (2 * math.pi * self.r_jj * self.I_c) / self.Phi_0
    
    def calculate_alpha(self):
        return self.r_leak / self.r_jj
    
    def calculate_beta(self):
        return (2 * math.pi * self.L * self.I_c) / self.Phi_0
    
    def calculate_gamma(self):
        return 1 / self.beta
    
    def calculate_tau(self):
        return self.beta / self.alpha

    # Current conversions
    def physical_to_dimensionless_current(self, I):
        return float(I) / self.I_c

    def dimensionless_to_physical_current(self, i):
        return float(i) * self.I_c

    # Flux conversions
    def physical_to_dimensionless_flux(self, Phi):
        return float(Phi) / self.Phi_0

    def dimensionless_to_physical_flux(self, phi):
        return float(phi) * self.Phi_0

    # Time conversions
    def physical_to_dimensionless_time(self, t):
        return float(t) * self.omega_c

    def dimensionless_to_physical_time(self, t_prime):
        return float(t_prime) / self.omega_c

    # Mutual inductance conversions
    def physical_to_dimensionless_inductance(self, M):
        return (float(M) * self.I_c) / self.Phi_0

    def dimensionless_to_physical_inductance(self, J):
        return (float(J) * self.Phi_0) / self.I_c
    
    # Flux quantum rate conversion
    def physical_to_dimensionless_fq_rate(self, G_fq):
        return (2 * math.pi / self.omega_c) * G_fq

    def dimensionless_to_physical_fq_rate(self, g_fq):
        return (self.omega_c / (2 * math.pi)) * g_fq

app = Flask(__name__)
converter = PhenomenologicalConverter()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_constants', methods=['POST'])
def update_constants():
    data = request.json
    try:
        converter.I_c = float(data.get('I_c', converter.I_c))
        converter.r_jj = float(data.get('r_jj', converter.r_jj))
        converter.r_leak = float(data.get('r_leak', converter.r_leak))
        converter.L = float(data.get('L', converter.L))
        converter.Phi_0 = float(data.get('Phi_0', converter.Phi_0))

        # Recalculate all derived quantities
        converter.omega_c = converter.calculate_omega_c()
        converter.alpha = converter.calculate_alpha()
        converter.beta = converter.calculate_beta()
        converter.gamma = converter.calculate_gamma()
        converter.tau = converter.calculate_tau()

        return jsonify({
            'I_c': converter.I_c,
            'r_jj': converter.r_jj,
            'r_leak': converter.r_leak,
            'L': converter.L,
            'Phi_0': converter.Phi_0,
            'omega_c': converter.omega_c,
            'alpha': converter.alpha,
            'beta': converter.beta,
            'gamma': converter.gamma,
            'tau': converter.tau
        })
    except ValueError as e:
        return jsonify({'error': 'Invalid input values'}), 400

@app.route('/convert_to_physical', methods=['POST'])
def convert_to_physical():
    data = request.json
    i = data.get('i')
    phi = data.get('phi')
    J = data.get('J')
    t_prime = data.get('t_prime')
    g_fq = data.get('g_fq')

    result = {}
    if i is not None:
        result['I'] = converter.dimensionless_to_physical_current(i)
    if phi is not None:
        result['Phi'] = converter.dimensionless_to_physical_flux(phi)
    if J is not None:
        result['M'] = converter.dimensionless_to_physical_inductance(J)
    if t_prime is not None:
        result['t'] = converter.dimensionless_to_physical_time(t_prime)
    if g_fq is not None:
        result['G_fq'] = converter.dimensionless_to_physical_fq_rate(g_fq)

    return jsonify(result)

@app.route('/convert_to_dimensionless', methods=['POST'])
def convert_to_dimensionless():
    data = request.json
    I = data.get('I')
    Phi = data.get('Phi')
    M = data.get('M')
    t = data.get('t')
    G_fq = data.get('G_fq')

    result = {}
    if I is not None:
        result['i'] = converter.physical_to_dimensionless_current(I)
    if Phi is not None:
        result['phi'] = converter.physical_to_dimensionless_flux(Phi)
    if M is not None:
        result['J'] = converter.physical_to_dimensionless_inductance(M)
    if t is not None:
        result['t_prime'] = converter.physical_to_dimensionless_time(t)
    if G_fq is not None:
        result['g_fq'] = converter.physical_to_dimensionless_fq_rate(G_fq)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
