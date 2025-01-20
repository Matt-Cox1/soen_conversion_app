// static/js/script.js


// variable name mapping
const latexNames = {
    'I': 'I',
    'i': 'i',
    'Phi': '\\Phi',
    'phi': '\\phi',
    'M': 'M',
    'J': 'J',
    't': 't',
    't_prime': "t'",
    'G_fq': 'G_{fq}',
    'g_fq': 'g_{fq}'
};

function formatScientific(num) {
    if (num === undefined || num === null) return '';
    const str = num.toExponential(4);
    let [coef, exp] = str.split('e');
    exp = parseInt(exp);
    return `${coef} \\times 10^{${exp}}`;
}

async function updateConstants() {
    const button = document.querySelector('.update-constants-btn');
    try {
        button.classList.add('loading');
        button.textContent = 'Updating...';

        const constants = {
            I_c: parseFloat(document.getElementById('I_c').value),
            r_jj: parseFloat(document.getElementById('r_jj').value),
            r_leak: parseFloat(document.getElementById('r_leak').value),
            L: parseFloat(document.getElementById('L').value)
        };

        const response = await fetch('/update_constants', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(constants)
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();

        // derived constants display
        Object.keys(data).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.innerHTML = `\\(${formatScientific(data[key])}\\)`;
            }
        });

        MathJax.typeset();

    } catch (error) {
        alert('Error updating constants: ' + error.message);
    } finally {
        button.classList.remove('loading');
        button.textContent = 'Update Constants';
    }
}

async function convertToPhysical() {
    const button = document.querySelector('.convert-physical-btn');
    const result = document.getElementById('physicalResult');
    try {
        button.classList.add('loading');

        const data = {
            i: parseFloat(document.getElementById('i').value),
            phi: parseFloat(document.getElementById('phi').value),
            J: parseFloat(document.getElementById('J').value),
            t_prime: parseFloat(document.getElementById('t_prime').value),
            g_fq: parseFloat(document.getElementById('g_fq').value)
        };

        const response = await fetch('/convert_to_physical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const resultData = await response.json();

        result.innerHTML = Object.entries(resultData)
            .map(([key, value]) => `\\[${latexNames[key]} = ${formatScientific(value)}\\]`)
            .join('\n');

        MathJax.typeset([result]);

    } catch (error) {
        result.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    } finally {
        button.classList.remove('loading');
    }
}

async function convertToDimensionless() {
    const button = document.querySelector('.convert-dimensionless-btn');
    const result = document.getElementById('dimensionlessResult');
    try {
        button.classList.add('loading');

        const data = {
            I: parseFloat(document.getElementById('I').value),
            Phi: parseFloat(document.getElementById('Phi').value),
            M: parseFloat(document.getElementById('M').value),
            t: parseFloat(document.getElementById('t').value),
            G_fq: parseFloat(document.getElementById('G_fq').value)
        };

        const response = await fetch('/convert_to_dimensionless', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const resultData = await response.json();

        result.innerHTML = Object.entries(resultData)
            .map(([key, value]) => `\\[${latexNames[key]} = ${formatScientific(value)}\\]`)
            .join('\n');

        MathJax.typeset([result]);

    } catch (error) {
        result.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    } finally {
        button.classList.remove('loading');
    }
}

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
