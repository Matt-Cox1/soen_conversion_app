// FILEPATH: scr/soen/utils/physical_mappings/static/js/script.js

// Variable name mapping for LaTeX rendering
const latexNames = {
    // Conversion parameters
    'I': 'I',
    'i': 'i',
    'Phi': '\\Phi',
    'phi': '\\phi',
    'M': 'M',
    'J': 'J',
    't': 't',
    't_prime': "t'",
    'G_fq': 'G_{fq}',
    'g_fq': 'g_{fq}',
    'beta_L': '\\beta_L',
    'L': 'L',
    
    // Input parameters
    'I_c': 'I_c',
    'r_leak': 'r_{leak}',
    'gamma_c': '\\gamma_c',
    'beta_c': '\\beta_c',
    
    // Junction parameters
    'c_j': 'c_j',
    'r_jj': 'r_{jj}',
    'V_j': 'V_j',
    
    // Time scales
    'tau_0': '\\tau_0',
    'omega_c': '\\omega_c',
    'omega_p': '\\omega_p',
    
    // Dimensionless parameters
    'alpha': '\\alpha',
    'gamma': '\\gamma',
    'tau': '\\tau'
};

// Format numbers in scientific notation with proper LaTeX
function formatScientific(num) {
    if (num === undefined || num === null) return '';
    if (num === Infinity) return '\\infty';
    if (num === -Infinity) return '-\\infty';
    if (isNaN(num)) return 'undefined';
    
    const str = num.toExponential(4);
    let [coef, exp] = str.split('e');
    exp = parseInt(exp);
    
    // Format coefficients near 1 more nicely
    if (Math.abs(parseFloat(coef)) === 1) {
        coef = coef[0] === '-' ? '-1' : '1';
    }
    
    return `${coef} \\times 10^{${exp}}`;
}

// Update all physical constants and derived parameters
async function updateConstants() {
    const button = document.querySelector('.update-constants-btn');
    try {
        button.classList.add('loading');
        button.textContent = 'Updating...';

        // Collect input values
        const constants = {
            I_c: parseFloat(document.getElementById('I_c').value),
            L: parseFloat(document.getElementById('L').value),
            r_leak: parseFloat(document.getElementById('r_leak').value),
            gamma_c: parseFloat(document.getElementById('gamma_c').value),
            beta_c: parseFloat(document.getElementById('beta_c').value)
        };

        // Validate inputs
        if (Object.values(constants).some(val => val <= 0)) {
            throw new Error('All parameters must be positive');
        }

        const response = await fetch('/update_constants', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(constants)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const data = await response.json();

        // Update all parameter displays
        Object.entries(data).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                // Use LaTeX names for proper rendering
                element.innerHTML = `\\(${formatScientific(value)}\\)`;
            }
        });

        // Trigger MathJax to rerender equations
        MathJax.typesetPromise();

    } catch (error) {
        // Display error in a user-friendly way
        const errorMessage = error.message || 'An error occurred while updating constants';
        alert(`Error: ${errorMessage}`);
    } finally {
        button.classList.remove('loading');
        button.textContent = 'Update Constants';
    }
}

// Convert from dimensionless to physical units
async function convertToPhysical() {
    const button = document.querySelector('.convert-physical-btn');
    const result = document.getElementById('physicalResult');
    try {
        button.classList.add('loading');

        // Collect input values
        const data = {
            i: parseFloat(document.getElementById('i').value),
            phi: parseFloat(document.getElementById('phi').value),
            J: parseFloat(document.getElementById('J').value),
            t_prime: parseFloat(document.getElementById('t_prime').value),
            g_fq: parseFloat(document.getElementById('g_fq').value),
            beta_L: parseFloat(document.getElementById('beta_L').value)
        };

        // Filter out undefined/null values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        const response = await fetch('/convert_to_physical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const resultData = await response.json();

        // Format and display results
        result.innerHTML = Object.entries(resultData)
            .map(([key, value]) => `\\[${latexNames[key]} = ${formatScientific(value)}\\]`)
            .join('\n');

        // Render new equations
        MathJax.typesetPromise([result]);

    } catch (error) {
        result.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    } finally {
        button.classList.remove('loading');
    }
}

// Convert from physical to dimensionless units
async function convertToDimensionless() {
    const button = document.querySelector('.convert-dimensionless-btn');
    const result = document.getElementById('dimensionlessResult');
    try {
        button.classList.add('loading');

        // Collect input values
        const data = {
            I: parseFloat(document.getElementById('I').value),
            Phi: parseFloat(document.getElementById('Phi').value),
            M: parseFloat(document.getElementById('M').value),
            t: parseFloat(document.getElementById('t').value),
            G_fq: parseFloat(document.getElementById('G_fq').value),
            L: parseFloat(document.getElementById('L').value)
        };

        // Filter out undefined/null values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        const response = await fetch('/convert_to_dimensionless', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const resultData = await response.json();

        // Format and display results
        result.innerHTML = Object.entries(resultData)
            .map(([key, value]) => `\\[${latexNames[key]} = ${formatScientific(value)}\\]`)
            .join('\n');

        // Render new equations
        MathJax.typesetPromise([result]);

    } catch (error) {
        result.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    } finally {
        button.classList.remove('loading');
    }
}

// Tab switching functionality
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
}

// Initialize when document loads
document.addEventListener('DOMContentLoaded', () => {
    updateConstants();
});
