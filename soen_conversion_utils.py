# FILEPATH: soen/utils/physical_mappings/soen_conversion_utils.py


import math
import argparse
from scipy.constants import h, e

class PhysicalConverter:
    """
    A class to convert dimensionless values into physical values for superconducting optoelectronic networks.
    
    Attributes:
        I_c (float): Critical current in amperes
        L (float): Inductance in henries
        Phi_0 (float): Magnetic flux quantum in weber (fixed constant)
        r_leak (float): Leak resistance in ohms
        r_jj (float): Junction shunt resistance in ohms
        omega_c (float): Characteristic frequency in rad/s
        alpha (float): Resistance ratio (dimensionless)
        beta_L (float): Dimensionless inductance
        gamma (float): Inverse dimensionless inductance
        tau (float): Dimensionless time constant
        gamma_c (float): Proportionality between capacitance and Ic (units of farads per amp)
        beta_c (float): Dimensionless parameter related to the junction
    """
    
    # Fixed constant
    Phi_0 = h / (2 * e)  # Magnetic flux quantum in weber
    
    def __init__(self, I_c=100e-6, L=1e-9, r_leak=1.0, r_jj=2.56, gamma_c=1.5e-9, beta_c=1.0):
        """
        Initializes the PhysicalConverter with given physical parameters.
        
        Args:
            I_c (float): Critical current in amperes
            L (float): Inductance in henries
            r_leak (float): Leak resistance in ohms
            r_jj (float): Junction shunt resistance in ohms
            gamma_c (float): Proportionality between capacitance and Ic (units of farads per amp)
            beta_c (float): Dimensionless parameter related to the junction
        """
        self.I_c = I_c
        self.r_jj = r_jj
        self.L = L
        self.r_leak = r_leak
        self.gamma_c = gamma_c
        self.beta_c = beta_c
        
        # Calculate derived quantities
        self.omega_c = self.calculate_omega_c()
        self.alpha = self.calculate_alpha()
        self.beta_L = self.calculate_beta_L()
        self.gamma = self.calculate_gamma()
        self.tau = self.calculate_tau()
        self.jj_params = self.get_jj_params()

    def calculate_omega_c(self):
        """Calculates characteristic frequency ωc = 2πrjjIc/Φ0"""
        return (2 * math.pi * self.r_jj * self.I_c) / self.Phi_0

    def calculate_alpha(self):
        """Calculates resistance ratio α = rleak/rjj"""
        return self.r_leak / self.r_jj

    def calculate_beta_L(self):
        """Calculates dimensionless inductance β_L = 2πLIc/Φ0"""
        return (2 * math.pi * self.L * self.I_c) / self.Phi_0

    def calculate_gamma(self):
        """Calculates inverse dimensionless inductance γ = 1/β_L"""
        return 1 / self.beta_L

    def calculate_tau(self):
        """Calculates dimensionless time constant τ = β_L/α"""
        return self.beta_L / self.alpha

    def get_jj_params(self):
        """Calculates JJ parameters based on Ic and beta_c"""
        c_j = self.gamma_c * self.I_c  # JJ capacitance
        r_jj = math.sqrt((self.beta_c * self.Phi_0) / (2 * math.pi * c_j * self.I_c))
        tau_0 = self.Phi_0 / (2 * math.pi * self.I_c * r_jj)
        V_j = self.I_c * r_jj
        omega_c = 2 * math.pi * self.I_c * r_jj / self.Phi_0
        omega_p = math.sqrt(2 * math.pi * self.I_c / (self.Phi_0 * c_j))
        
        return {'c_j': c_j, 'r_jj': r_jj, 'tau_0': tau_0, 'Ic': self.I_c, 'beta_c': self.beta_c, 'gamma_c': self.gamma_c, 'V_j': V_j, 'omega_c': omega_c, 'omega_p': omega_p}

    def physical_to_dimensionless_current(self, I):
        """Converts physical current to dimensionless (i = I/Ic)"""
        return float(I) / self.I_c

    def dimensionless_to_physical_current(self, i):
        """Converts dimensionless current to physical (I = i·Ic)"""
        return float(i) * self.I_c

    def physical_to_dimensionless_time(self, t):
        """Converts physical time to dimensionless (t' = t·ωc)"""
        return float(t) * self.omega_c

    def dimensionless_to_physical_time(self, t_prime):
        """Converts dimensionless time to physical (t = t'/ωc)"""
        return float(t_prime) / self.omega_c

    def physical_to_dimensionless_flux(self, Phi):
        """Converts physical flux to dimensionless (φ = Φ/Φ0)"""
        return float(Phi) / self.Phi_0

    def dimensionless_to_physical_flux(self, phi):
        """Converts dimensionless flux to physical (Φ = φ·Φ0)"""
        return float(phi) * self.Phi_0

    def physical_to_dimensionless_inductance(self, M):
        """Converts physical mutual inductance to dimensionless (J = M·Ic/Φ0)"""
        return (float(M) * self.I_c) / self.Phi_0

    def dimensionless_to_physical_inductance(self, J):
        """Converts dimensionless inductance to physical (M = J·Φ0/Ic)"""
        return (float(J) * self.Phi_0) / self.I_c

    def physical_to_dimensionless_fq_rate(self, G_fq):
        """Converts physical flux quantum rate to dimensionless (gfq = 2π·Gfq/ωc)"""
        return (2 * math.pi / self.omega_c) * float(G_fq)

    def dimensionless_to_physical_fq_rate(self, g_fq):
        """Converts dimensionless flux quantum rate to physical (Gfq = ωc·gfq/2π)"""
        return (self.omega_c / (2 * math.pi)) * float(g_fq)

def main():
    parser = argparse.ArgumentParser(description="Convert between dimensionless and physical values for superconducting optoelectronic networks.")
    parser.add_argument('--I_c', type=float, default=100e-6, help="Critical current [A]")
    parser.add_argument('--r_jj', type=float, default=2.56, help="Junction shunt resistance [Ω]")
    parser.add_argument('--L', type=float, default=1e-9, help="Inductance [H]")
    parser.add_argument('--r_leak', type=float, default=1.0, help="Leak resistance [Ω]")
    parser.add_argument('--gamma_c', type=float, default=1.5e-9, help="Proportionality between capacitance and Ic [F/A]")
    parser.add_argument('--beta_c', type=float, default=1.0, help="Dimensionless parameter related to the junction")
    
    # Add arguments for dimensionless quantities
    parser.add_argument('--i', type=float, help="Dimensionless current")
    parser.add_argument('--phi', type=float, help="Dimensionless flux")
    parser.add_argument('--J', type=float, help="Dimensionless inductance")
    parser.add_argument('--t_prime', type=float, help="Dimensionless time")
    parser.add_argument('--g_fq', type=float, help="Dimensionless flux quantum rate")
    
    args = parser.parse_args()
    
    converter = PhysicalConverter(
        I_c=args.I_c,
        r_jj=args.r_jj,
        L=args.L,
        r_leak=args.r_leak,
        gamma_c=args.gamma_c,
        beta_c=args.beta_c
    )
    
    # Print derived quantities
    print("\nDerived Quantities:")
    print(f"ωc = {converter.omega_c:.2e} rad/s")
    print(f"α = {converter.alpha:.2e}")
    print(f"β_L = {converter.beta_L:.2e}")
    print(f"γ = {converter.gamma:.2e}")
    print(f"τ = {converter.tau:.2e}")
    
    # Print JJ parameters
    print("\nJJ Parameters:")
    for key, value in converter.jj_params.items():
        print(f"{key} = {value:.2e}")
    
    # Convert and print requested values
    print("\nConversions:")
    if args.i is not None:
        I = converter.dimensionless_to_physical_current(args.i)
        print(f"i = {args.i} → I = {I:.2e} A")
    if args.phi is not None:
        Phi = converter.dimensionless_to_physical_flux(args.phi)
        print(f"φ = {args.phi} → Φ = {Phi:.2e} Wb")
    if args.J is not None:
        M = converter.dimensionless_to_physical_inductance(args.J)
        print(f"J = {args.J} → M = {M:.2e} H")
    if args.t_prime is not None:
        t = converter.dimensionless_to_physical_time(args.t_prime)
        print(f"t' = {args.t_prime} → t = {t:.2e} s")
    if args.g_fq is not None:
        G_fq = converter.dimensionless_to_physical_fq_rate(args.g_fq)
        print(f"g_fq = {args.g_fq} → G_fq = {G_fq:.2e} Hz")

if __name__ == "__main__":
    main()
