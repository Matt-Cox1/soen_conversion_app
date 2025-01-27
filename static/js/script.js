// FILEPATH: src/soen/utils/physical_mappings/static/js/script.js

// Format number for display with appropriate scientific notation
function formatValue(value) {
    if (value === undefined || value === null) return '';
    if (value === Infinity) return '\\infty';
    if (value === -Infinity) return '-\\infty';
    if (isNaN(value)) return 'undefined';
    
    // Convert to scientific notation with 4 significant figures
    const str = value.toExponential(4);
    let [coef, exp] = str.split('e');
    exp = parseInt(exp);
    
    // Format coefficients near 1 more nicely
    if (Math.abs(parseFloat(coef)) === 1) {
        coef = coef[0] === '-' ? '-1' : '1';
    }
    
    return `${coef} \\times 10^{${exp}}`;
}

// Format full equation with value for display
function formatEquation(latex, value, unit = '') {
    const formattedValue = formatValue(value);
    if (unit) {
        return `$${latex} = ${formattedValue}\\,\\text{${unit}}$`;
    }
    return `$${latex} = ${formattedValue}$`;
}

// Show error message in a result div
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="error">${message}</div>`;
}



async function updateBaseParameters() {
    const button = document.querySelector('.base-parameters button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            I_c: parseFloat(document.getElementById('I_c').value),
            gamma_c: parseFloat(document.getElementById('gamma_c').value),
            beta_c: parseFloat(document.getElementById('beta_c').value)
        };

        // Validate inputs
        if (Object.values(data).some(val => isNaN(val) || val <= 0)) {
            throw new Error('All parameters must be positive numbers');
        }

        const response = await fetch('/update_base_parameters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const result = await response.json();

        // Update all parameter displays with LaTeX formatting
        for (const [key, data] of Object.entries(result)) {
            const element = document.getElementById(key);
            if (element) {
                let unit = '';
                // Add appropriate units
                switch(key) {
                    case 'I_c': unit = 'A'; break;
                    case 'gamma_c': unit = 'F/A'; break;
                    case 'c_j': unit = 'F'; break;
                    case 'r_jj': unit = '\u03A9'; break;
                    case 'V_j': unit = 'V'; break;
                    case 'tau_0': unit = 's'; break;
                    case 'omega_c':
                    case 'omega_p': unit = 'rad/s'; break;
                }
                element.innerHTML = formatEquation(data.latex, data.value, unit);
            }
        }

        // After updating base parameters, also update any derived parameters
        await updateDerivedParameters();

        // Trigger MathJax to rerender all equations
        MathJax.typesetPromise();

    } catch (error) {
        showError('baseParametersResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}

// Convert physical values to dimensionless
async function convertToDimensionless() {
    const button = document.querySelector('.conversion-column:first-child button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            I: parseFloat(document.getElementById('I').value),
            Phi: parseFloat(document.getElementById('Phi').value),
            L: parseFloat(document.getElementById('L').value),
            t: parseFloat(document.getElementById('t').value),
            r_leak: parseFloat(document.getElementById('r_leak').value),
            G_fq: parseFloat(document.getElementById('G_fq').value)
        };

        // Remove NaN values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        // Validate we have at least one value
        if (Object.keys(data).length === 0) {
            throw new Error('Please enter at least one value to convert');
        }

        const response = await fetch('/convert_to_dimensionless', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Format and display results
        const resultHtml = Object.entries(result)
            .map(([key, data]) => formatEquation(data.latex, data.value))
            .join('<br>');
        
        document.getElementById('dimensionlessResult').innerHTML = resultHtml;

        // Calculate derived parameters if we have the necessary values
        if (result.beta_L && result.alpha) {
            calculateDerivedParameters(result.beta_L.value, result.alpha.value);
        }

        // Trigger MathJax to rerender equations
        MathJax.typesetPromise();

    } catch (error) {
        showError('dimensionlessResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}

// Convert dimensionless values to physical
async function convertToPhysical() {
    const button = document.querySelector('.conversion-column:last-child button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            i: parseFloat(document.getElementById('i').value),
            phi: parseFloat(document.getElementById('phi').value),
            beta_L: parseFloat(document.getElementById('beta_L').value),
            gamma: parseFloat(document.getElementById('gamma').value),
            t_prime: parseFloat(document.getElementById('t_prime').value),
            alpha: parseFloat(document.getElementById('alpha').value),
            g_fq: parseFloat(document.getElementById('g_fq').value)
        };

        // Remove NaN values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        // Validate we have at least one value
        if (Object.keys(data).length === 0) {
            throw new Error('Please enter at least one value to convert');
        }

        const response = await fetch('/convert_to_physical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Format and display results
        const resultHtml = Object.entries(result)
            .map(([key, data]) => formatEquation(data.latex, data.value, data.unit))
            .join('<br>');
        
        document.getElementById('physicalResult').innerHTML = resultHtml;

        // Calculate derived parameters if we have beta_L and alpha
        if (data.beta_L && data.alpha) {
            calculateDerivedParameters(data.beta_L, data.alpha);
        }

        // Trigger MathJax to rerender equations
        MathJax.typesetPromise();

    } catch (error) {
        showError('physicalResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}


// Add event listeners for beta_L and alpha inputs
document.getElementById('beta_L').addEventListener('input', updateDerivedParameters);
document.getElementById('alpha').addEventListener('input', updateDerivedParameters);
document.getElementById('gamma').addEventListener('input', updateDerivedParameters);

// Function to update derived parameters
async function updateDerivedParameters() {
    let beta_L = parseFloat(document.getElementById('beta_L').value);
    const gamma = parseFloat(document.getElementById('gamma').value);
    const alpha = parseFloat(document.getElementById('alpha').value);
    
    // If gamma is provided but beta_L isn't, calculate beta_L
    if (isNaN(beta_L) && !isNaN(gamma) && gamma !== 0) {
        beta_L = 1 / gamma;
    }

    if (!isNaN(beta_L) && !isNaN(alpha) && alpha !== 0) {
        const tau = beta_L / alpha;
        document.getElementById('tau').innerHTML = formatEquation('\\tau', tau);
        MathJax.typesetPromise();
    }
}

// Calculate and update derived dimensionless parameters
async function calculateDerivedParameters(beta_L, alpha) {
    try {
        const response = await fetch('/calculate_derived', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ beta_L, alpha })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Update derived parameters display
        if (result.tau) {
            document.getElementById('tau').innerHTML = formatEquation(result.tau.latex, result.tau.value);
            MathJax.typesetPromise();
        }
        

    } catch (error) {
        console.error('Error calculating derived parameters:', error);
    }
}

// Link gamma and beta_L inputs
document.getElementById('gamma').addEventListener('input', function(e) {
    const gamma = parseFloat(e.target.value);
    if (!isNaN(gamma) && gamma !== 0) {
        document.getElementById('beta_L').value = 1 / gamma;
    }
});

document.getElementById('beta_L').addEventListener('input', function(e) {
    const beta_L = parseFloat(e.target.value);
    if (!isNaN(beta_L) && beta_L !== 0) {
        document.getElementById('gamma').value = 1 / beta_L;
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    updateBaseParameters();
});
