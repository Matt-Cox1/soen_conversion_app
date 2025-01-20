# FILEPATH: soen/utils/physical_mappings/soen_conversion_utils.py


import math
import argparse

class PhysicalConverter:
    """
    A class to convert dimensionless values into physical values for superconducting optoelectronic networks.
    
    Attributes:
        I_c (float): Critical current in amperes
        L (float): Inductance in henries
        Phi_0 (float): Magnetic flux quantum in weber
        r_leak (float): Leak resistance in ohms
        r_jj (float): Junction shunt resistance in ohms
        omega_c (float): Characteristic frequency in rad/s
        alpha (float): Resistance ratio (dimensionless)
        beta (float): Dimensionless inductance
        gamma (float): Inverse dimensionless inductance
        tau (float): Dimensionless time constant
    """
    
    def __init__(self, I_c=100e-6, L=1e-9, Phi_0=2.07e-15, r_leak=1.0,r_jj=2.56):
        """
        Initializes the PhysicalConverter with given physical parameters.
        
        Args:
            I_c (float): Critical current in amperes
            L (float): Inductance in henries
            Phi_0 (float): Magnetic flux quantum in weber
            r_leak (float): Leak resistance in ohms
            r_jj (float): Junction shunt resistance in ohms
        """
        self.I_c = I_c
        self.r_jj = r_jj
        self.L = L
        self.Phi_0 = Phi_0
        self.r_leak = r_leak
        
        # Calculate derived quantities
        self.omega_c = self.calculate_omega_c()
        self.alpha = self.calculate_alpha()
        self.beta = self.calculate_beta()
        self.gamma = self.calculate_gamma()
        self.tau = self.calculate_tau()

    def calculate_omega_c(self):
        """Calculates characteristic frequency ωc = 2πrjjIc/Φ0"""
        return (2 * math.pi * self.r_jj * self.I_c) / self.Phi_0

    def calculate_alpha(self):
        """Calculates resistance ratio α = rleak/rjj"""
        return self.r_leak / self.r_jj

    def calculate_beta(self):
        """Calculates dimensionless inductance β = 2πLIc/Φ0"""
        return (2 * math.pi * self.L * self.I_c) / self.Phi_0

    def calculate_gamma(self):
        """Calculates inverse dimensionless inductance γ = 1/β"""
        return 1 / self.beta

    def calculate_tau(self):
        """Calculates dimensionless time constant τ = β/α"""
        return self.beta / self.alpha

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
    parser.add_argument('--Phi_0', type=float, default=2.07e-15, help="Magnetic flux quantum [Wb]")
    parser.add_argument('--r_leak', type=float, default=1.0, help="Leak resistance [Ω]")
    
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
        Phi_0=args.Phi_0,
        r_leak=args.r_leak
    )
    
    # Print derived quantities
    print("\nDerived Quantities:")
    print(f"ωc = {converter.omega_c:.2e} rad/s")
    print(f"α = {converter.alpha:.2e}")
    print(f"β = {converter.beta:.2e}")
    print(f"γ = {converter.gamma:.2e}")
    print(f"τ = {converter.tau:.2e}")
    
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
