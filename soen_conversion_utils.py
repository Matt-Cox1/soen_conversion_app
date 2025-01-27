# FILEPATH: src/soen/utils/physical_mappings/soen_conversion_utils.py

import math
from scipy.constants import h, e

class PhysicalConverter:
    """
    Core converter class for SOEN calculations.
    
    Base Constants:
        Phi_0: Magnetic flux quantum (h/2e)
        I_c: Critical current
        gamma_c: Capacitance proportionality
        beta_c: Stewart-McCumber parameter
    """
    
    # Fixed constant: Magnetic flux quantum in Weber
    Phi_0 = h / (2 * e)  # ~2.067833848e-15 Wb

    def __init__(
        self,
        I_c: float = 100e-6,       # Critical current [A]
        gamma_c: float = 1.5e-9,   # Proportionality between capacitance and Ic [F/A]
        beta_c: float = 0.3,       # Stewart-McCumber parameter
    ):
        """Initialize with base physical constants."""
        # Base constants - using properties to ensure derived values are updated
        self._I_c = I_c
        self._gamma_c = gamma_c
        self._beta_c = beta_c
        self._update_derived_parameters()
        
    def _update_derived_parameters(self):
        """Update all derived parameters based on current base values"""
        # First calculate c_j as it's needed for other calculations
        self._c_j = self._calculate_junction_capacitance()
        
        # Then calculate all other derived parameters
        self._r_jj = self._calculate_junction_resistance()
        self._omega_c = self._calculate_josephson_frequency()
        self._omega_p = self._calculate_plasma_frequency()
        self._tau_0 = self._calculate_characteristic_time()
        self._V_j = self._calculate_junction_voltage()
        
    # Properties to ensure derived parameters update when base parameters change
    @property
    def I_c(self):
        return self._I_c
        
    @I_c.setter
    def I_c(self, value):
        self._I_c = value
        self._update_derived_parameters()
        
    @property
    def gamma_c(self):
        return self._gamma_c
        
    @gamma_c.setter
    def gamma_c(self, value):
        self._gamma_c = value
        self._update_derived_parameters()
        
    @property
    def beta_c(self):
        return self._beta_c
        
    @beta_c.setter
    def beta_c(self, value):
        self._beta_c = value
        self._update_derived_parameters()

    # Properties for derived parameters
    @property
    def c_j(self):
        return self._c_j

    @property
    def r_jj(self):
        return self._r_jj

    @property
    def omega_c(self):
        return self._omega_c

    @property
    def omega_p(self):
        return self._omega_p

    @property
    def tau_0(self):
        return self._tau_0

    @property
    def V_j(self):
        return self._V_j

    # ---------- Base Parameter Calculations ----------
    def _calculate_junction_capacitance(self) -> float:
        """Junction capacitance: c_j = γ_c * I_c"""
        return self.gamma_c * self.I_c

    def _calculate_junction_resistance(self) -> float:
        """Junction resistance from β_c: r_jj = sqrt((β_c * Φ_0)/(2π * c_j * I_c))"""
        return math.sqrt(
            (self.beta_c * self.Phi_0) / (2 * math.pi * self.c_j * self.I_c)
        )

    def _calculate_josephson_frequency(self) -> float:
        """Josephson frequency: ω_c = (2π * I_c * r_jj) / Φ_0"""
        return (2 * math.pi * self.I_c * self.r_jj) / self.Phi_0

    def _calculate_plasma_frequency(self) -> float:
        """Plasma frequency: ω_p = sqrt((2π * I_c)/(Φ_0 * c_j))"""
        try:
            return math.sqrt((2 * math.pi * self.I_c) / (self.Phi_0 * self.c_j))
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error calculating plasma frequency: {e}")
            print(f"I_c: {self.I_c}, c_j: {self.c_j}")
            raise

    def _calculate_characteristic_time(self) -> float:
        """Characteristic time: τ_0 = Φ_0/(2π * I_c * r_jj)"""
        return self.Phi_0 / (2 * math.pi * self.I_c * self.r_jj)

    def _calculate_junction_voltage(self) -> float:
        """Junction voltage: V_j = I_c * r_jj"""
        return self.I_c * self.r_jj

    # ---------- Physical ↔ Dimensionless Conversions ----------
    
    # Current
    def physical_to_dimensionless_current(self, I: float) -> float:
        """i = I/I_c"""
        return I / self.I_c

    def dimensionless_to_physical_current(self, i: float) -> float:
        """I = i * I_c"""
        return i * self.I_c

    # Flux
    def physical_to_dimensionless_flux(self, Phi: float) -> float:
        """φ = Φ/Φ_0"""
        return Phi / self.Phi_0

    def dimensionless_to_physical_flux(self, phi: float) -> float:
        """Φ = φ * Φ_0"""
        return phi * self.Phi_0

    # Inductance (and gamma)
    def physical_to_dimensionless_inductance(self, L: float) -> float:
        """β_L = (2π * I_c * L)/Φ_0"""
        return (2 * math.pi * self.I_c * L) / self.Phi_0

    def dimensionless_to_physical_inductance(self, beta_L: float) -> float:
        """L = (β_L * Φ_0)/(2π * I_c)"""
        return (beta_L * self.Phi_0) / (2 * math.pi * self.I_c)

    def beta_L_to_gamma(self, beta_L: float) -> float:
        """γ = 1/β_L"""
        return 1.0 / beta_L if beta_L != 0 else float('inf')

    def gamma_to_beta_L(self, gamma: float) -> float:
        """β_L = 1/γ"""
        return 1.0 / gamma if gamma != 0 else float('inf')

    # Resistance
    def physical_to_dimensionless_resistance(self, r_leak: float) -> float:
        """α = r_leak/r_jj"""
        return r_leak / self.r_jj

    def dimensionless_to_physical_resistance(self, alpha: float) -> float:
        """r_leak = α * r_jj"""
        return alpha * self.r_jj

    # Time
    def physical_to_dimensionless_time(self, t: float) -> float:
        """t' = t * ω_c"""
        return t * self.omega_c

    def dimensionless_to_physical_time(self, t_prime: float) -> float:
        """t = t'/ω_c"""
        return t_prime / self.omega_c

    # Flux quantum rate
    def physical_to_dimensionless_fq_rate(self, G_fq: float) -> float:
        """g_fq = (2π * G_fq)/ω_c"""
        return (2 * math.pi * G_fq) / self.omega_c

    def dimensionless_to_physical_fq_rate(self, g_fq: float) -> float:
        """G_fq = (g_fq * ω_c)/(2π)"""
        return (g_fq * self.omega_c) / (2 * math.pi)

    # ---------- Derived Dimensionless Parameters ----------
    def calculate_tau(self, beta_L: float, alpha: float) -> float:
        """τ = β_L/α"""
        return beta_L / alpha if alpha != 0 else float('inf')

    def get_base_parameters(self) -> dict:
        """Return all base physical parameters"""
        return {
            'I_c': self.I_c,
            'gamma_c': self.gamma_c,
            'beta_c': self.beta_c,
            'c_j': self.c_j,
            'r_jj': self.r_jj,
            'omega_c': self.omega_c,
            'omega_p': self.omega_p,
            'tau_0': self.tau_0,
            'V_j': self.V_j
        }
