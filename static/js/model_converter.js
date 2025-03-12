// FILEPATH: src/soen_sim_v2/utils/physical_mappings/static/js/model_converter.js

document.addEventListener('DOMContentLoaded', function() {
    // File input change handler
    const fileInput = document.getElementById('model-file');
    const fileNameDisplay = document.getElementById('file-name');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                fileNameDisplay.textContent = this.files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file selected';
            }
        });
    }

    // Form submission handler
    const uploadForm = document.getElementById('upload-form');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const uploadStatus = document.getElementById('upload-status');
    const modelSummarySection = document.getElementById('model-summary-section');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show progress section
            uploadStatus.style.display = 'block';
            progressBar.style.width = '0%';
            statusMessage.textContent = 'Processing model...';
            
            // Get form data
            const formData = new FormData(uploadForm);
            
            // Make API call to upload and process model
            uploadAndProcessModel(formData);
        });
    }
    
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to current button
            this.classList.add('active');
            
            // Show corresponding content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Layer selector event handler
    const layerSelect = document.getElementById('layer-select');
    if (layerSelect) {
        layerSelect.addEventListener('change', function() {
            updateLayerParametersTable(this.value);
        });
    }
    
    // Connection selector event handler
    const connectionSelect = document.getElementById('connection-select');
    if (connectionSelect) {
        connectionSelect.addEventListener('change', function() {
            updateConnectionParametersTable(this.value);
        });
    }
    
    // Export button event handler
    const exportButton = document.getElementById('export-button');
    if (exportButton) {
        exportButton.addEventListener('click', function() {
            exportData();
        });
    }
});

// Global variable to store the current model data
let currentModelData = null;

/**
 * Upload and process the model file
 * @param {FormData} formData - Form data containing the model file and parameters
 */
function uploadAndProcessModel(formData) {
    // Simulate file upload with progress
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const modelSummarySection = document.getElementById('model-summary-section');
    
    // Start progress animation
    let progress = 0;
    const progressInterval = setInterval(function() {
        progress += 5;
        if (progress > 90) {
            clearInterval(progressInterval);
        }
        progressBar.style.width = progress + '%';
    }, 100);
    
    // Make API call to process the model
    fetch('/convert_model', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        clearInterval(progressInterval);
        
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Network response was not ok');
            });
        }
        
        return response.json();
    })
    .then(data => {
        // Store the model data globally
        currentModelData = data;
        
        // Complete progress bar
        progressBar.style.width = '100%';
        statusMessage.textContent = 'Model processed successfully!';
        
        // Show model summary section
        modelSummarySection.style.display = 'block';
        
        // Update UI with model data
        populateModelSummary(data);
        populateParameterOptions(data);
        
        // Hide progress bar after a delay
        setTimeout(function() {
            document.getElementById('upload-status').style.display = 'none';
        }, 2000);
    })
    .catch(error => {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        progressBar.style.backgroundColor = '#f44336';
        statusMessage.textContent = 'Error: ' + error.message;
        console.error('Error:', error);
    });
}

/**
 * Populate the model summary section with data
 * @param {Object} data - The processed model data
 */
function populateModelSummary(data) {
    // Populate model info table
    const modelInfoTable = document.getElementById('model-info-table');
    if (modelInfoTable) {
        modelInfoTable.innerHTML = '';
        
        // Add model metadata
        if (data.metadata) {
            const metadata = data.metadata;
            
            addTableRow(modelInfoTable, 'Model Type', 'SOEN Model');
            addTableRow(modelInfoTable, 'Epochs', metadata.epoch || 'N/A');
            addTableRow(modelInfoTable, 'Timestep (dt)', metadata.dt ? metadata.dt + ' (dimensionless)' : 'N/A');
            
            // Count number of layers and connections from appropriate data structure
            let layerCount = 0;
            let connectionCount = 0;
            
            if (data.results.dimensionless && data.results.dimensionless.layers) {
                layerCount = Object.keys(data.results.dimensionless.layers).length;
            } else if (data.results.layers) {
                // Fallback for old data structure
                layerCount = Object.keys(data.results.layers).length;
            }
            
            if (data.results.dimensionless && data.results.dimensionless.connections) {
                connectionCount = Object.keys(data.results.dimensionless.connections).length;
            } else if (data.results.connections) {
                // Fallback for old data structure
                connectionCount = Object.keys(data.results.connections).length;
            }
            
            addTableRow(modelInfoTable, 'Layer Count', layerCount);
            addTableRow(modelInfoTable, 'Connection Count', connectionCount);
        }
    }
    
    // Populate physical parameters table
    const physicalParamsTable = document.getElementById('physical-params-table');
    if (physicalParamsTable) {
        physicalParamsTable.innerHTML = '';
        
        // Add base physical parameters
        const baseParams = data.results.base_parameters;
        if (baseParams) {
            addTableRow(physicalParamsTable, 'Critical Current (I_c)', formatValue(baseParams.I_c) + ' A');
            addTableRow(physicalParamsTable, 'Capacitance Prop. (γ_c)', formatValue(baseParams.gamma_c) + ' F/A');
            addTableRow(physicalParamsTable, 'Stewart-McCumber (β_c)', formatValue(baseParams.beta_c));
            addTableRow(physicalParamsTable, 'Junction Capacitance (c_j)', formatValue(baseParams.c_j) + ' F');
            addTableRow(physicalParamsTable, 'Junction Resistance (r_jj)', formatValue(baseParams.r_jj) + ' Ω');
            addTableRow(physicalParamsTable, 'Josephson Frequency (ω_c)', formatValue(baseParams.omega_c) + ' rad/s');
            addTableRow(physicalParamsTable, 'Plasma Frequency (ω_p)', formatValue(baseParams.omega_p) + ' rad/s');
        }
    }
}

/**
 * Populate the parameter selectors with options
 * @param {Object} data - The processed model data
 */
function populateParameterOptions(data) {
    // Populate layer selector
    const layerSelect = document.getElementById('layer-select');
    if (layerSelect) {
        layerSelect.innerHTML = '';
        
        // Add layer options - handle both data structures
        let layers = null;
        if (data.results.dimensionless && data.results.dimensionless.layers) {
            layers = data.results.dimensionless.layers;
        } else if (data.results.layers) {
            // Fallback for old data structure
            layers = data.results.layers;
        }
        
        if (layers) {
            Object.keys(layers).sort((a, b) => parseInt(a) - parseInt(b)).forEach(layerId => {
                const option = document.createElement('option');
                option.value = layerId;
                option.textContent = `Layer ${layerId}`;
                layerSelect.appendChild(option);
            });
            
            // Update layer parameters table for the first layer
            if (layerSelect.options.length > 0) {
                updateLayerParametersTable(layerSelect.options[0].value);
            }
        }
    }
    
    // Populate connection selector
    const connectionSelect = document.getElementById('connection-select');
    if (connectionSelect) {
        connectionSelect.innerHTML = '';
        
        // Add connection options - handle both data structures
        let connections = null;
        if (data.results.dimensionless && data.results.dimensionless.connections) {
            connections = data.results.dimensionless.connections;
        } else if (data.results.connections) {
            // Fallback for old data structure
            connections = data.results.connections;
        }
        
        if (connections) {
            Object.keys(connections).forEach(connId => {
                const option = document.createElement('option');
                option.value = connId;
                option.textContent = `Connection ${connId}`;
                connectionSelect.appendChild(option);
            });
            
            // Update connection parameters table for the first connection
            if (connectionSelect.options.length > 0) {
                updateConnectionParametersTable(connectionSelect.options[0].value);
            }
        }
    }
    
    // Set up the plots section with categories
    setupPlotsSection(data);
}

/**
 * Update the layer parameters table for the selected layer
 * @param {string} layerId - The ID of the layer to display
 */
function updateLayerParametersTable(layerId) {
    if (!currentModelData) {
        return;
    }
    
    // Handle both data structures
    let dimensionlessLayerData = null;
    let physicalLayerData = null;
    
    if (currentModelData.results.dimensionless && currentModelData.results.dimensionless.layers && 
        currentModelData.results.dimensionless.layers[layerId]) {
        dimensionlessLayerData = currentModelData.results.dimensionless.layers[layerId];
    } else if (currentModelData.results.layers && currentModelData.results.layers[layerId]) {
        // Fallback for old data structure
        dimensionlessLayerData = currentModelData.results.layers[layerId];
    }
    
    if (currentModelData.results.physical && currentModelData.results.physical.layers && 
        currentModelData.results.physical.layers[layerId]) {
        physicalLayerData = currentModelData.results.physical.layers[layerId];
    }
    
    // Check if we have data to display
    if (!dimensionlessLayerData && !physicalLayerData) {
        return;
    }
    
    const layerParamsTable = document.getElementById('layer-params-table');
    
    if (layerParamsTable) {
        // Clear existing rows except header
        const tbody = layerParamsTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        // Process dimensionless parameters
        if (dimensionlessLayerData) {
            for (const paramName in dimensionlessLayerData) {
                const paramValue = dimensionlessLayerData[paramName];
                
                // Skip non-array parameters or empty arrays
                if (!Array.isArray(paramValue) && !isNdarray(paramValue) || 
                    (paramValue.length === 0 || paramValue.size === 0)) {
                    continue;
                }
                
                // Get statistics for the parameter
                const stats = calculateStatistics(paramValue);
                
                // Add the row with no unit (dimensionless)
                addParameterRow(tbody, paramName, stats, '');
            }
        }
        
        // Process physical parameters
        if (physicalLayerData) {
            for (const paramName in physicalLayerData) {
                const paramValue = physicalLayerData[paramName];
                
                // Skip non-array parameters or empty arrays
                if (!Array.isArray(paramValue) && !isNdarray(paramValue) || 
                    (paramValue.length === 0 || paramValue.size === 0)) {
                    continue;
                }
                
                // Get statistics for the parameter
                const stats = calculateStatistics(paramValue);
                
                // Determine unit based on parameter name
                let unit = '';
                if (paramName.includes('L')) unit = 'H';
                else if (paramName.includes('r_leak')) unit = 'Ω';
                else if (paramName.includes('phi_offset')) unit = 'Wb';
                else if (paramName.includes('bias_current')) unit = 'A';
                
                // Add the row with physical unit
                addParameterRow(tbody, paramName + ' (Physical)', stats, unit);
            }
        }
    }
}

/**
 * Add a parameter row to the table
 * @param {HTMLElement} tbody - The table body element
 * @param {string} paramName - The parameter name
 * @param {Object} stats - The parameter statistics
 * @param {string} unit - The unit of the parameter
 */


function addParameterRow(tbody, paramName, stats, unit) {
    const row = document.createElement('tr');
    
    const nameCell = document.createElement('td');
    nameCell.textContent = paramName;
    
    const minCell = document.createElement('td');
    minCell.textContent = formatValue(stats.min);
    
    const maxCell = document.createElement('td');
    maxCell.textContent = formatValue(stats.max);
    
    const meanCell = document.createElement('td');
    meanCell.textContent = formatValue(stats.mean);
    
    // Add std deviation cell
    const stdCell = document.createElement('td');
    stdCell.textContent = formatValue(stats.std);
    
    const unitCell = document.createElement('td');
    unitCell.textContent = unit;
    
    row.appendChild(nameCell);
    row.appendChild(minCell);
    row.appendChild(maxCell);
    row.appendChild(meanCell);
    row.appendChild(stdCell); // Add the std cell to the row
    row.appendChild(unitCell);
    
    tbody.appendChild(row);
}

/**
 * Update the connection parameters table for the selected connection
 * @param {string} connId - The ID of the connection to display
 */
function updateConnectionParametersTable(connId) {
    if (!currentModelData) {
        return;
    }
    
    // Handle both data structures
    let dimensionlessConnData = null;
    let physicalConnData = null;
    
    if (currentModelData.results.dimensionless && currentModelData.results.dimensionless.connections && 
        currentModelData.results.dimensionless.connections[connId]) {
        dimensionlessConnData = currentModelData.results.dimensionless.connections[connId];
    } else if (currentModelData.results.connections && currentModelData.results.connections[connId]) {
        // Fallback for old data structure
        dimensionlessConnData = currentModelData.results.connections[connId];
    }
    
    if (currentModelData.results.physical && currentModelData.results.physical.connections && 
        currentModelData.results.physical.connections[connId]) {
        physicalConnData = currentModelData.results.physical.connections[connId];
    }
    
    // Check if we have data to display
    if (!dimensionlessConnData && !physicalConnData) {
        return;
    }
    
    const connParamsTable = document.getElementById('connection-params-table');
    
    if (connParamsTable) {
        // Clear existing rows except header
        const tbody = connParamsTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        // Process dimensionless parameters
        if (dimensionlessConnData) {
            for (const paramName in dimensionlessConnData) {
                const paramValue = dimensionlessConnData[paramName];
                
                // Skip non-array parameters
                if (!Array.isArray(paramValue) && !isNdarray(paramValue)) {
                    continue;
                }
                
                // Get statistics for the parameter
                const stats = calculateStatistics(paramValue, true);
                
                // Add the row with no unit (dimensionless)
                addConnectionRow(tbody, paramName, stats, '');
            }
        }
        
        // Process physical parameters
        if (physicalConnData) {
            for (const paramName in physicalConnData) {
                const paramValue = physicalConnData[paramName];
                
                // Skip non-array parameters
                if (!Array.isArray(paramValue) && !isNdarray(paramValue)) {
                    continue;
                }
                
                // Get statistics for the parameter
                const stats = calculateStatistics(paramValue, true);
                
                // Determine unit based on parameter name
                let unit = '';
                if (paramName === 'M') unit = 'H';
                
                // Add the row with physical unit
                addConnectionRow(tbody, paramName + ' (Physical)', stats, unit);
            }
        }
    }
}

/**
 * Add a connection parameter row to the table
 * @param {HTMLElement} tbody - The table body element
 * @param {string} paramName - The parameter name
 * @param {Object} stats - The parameter statistics
 * @param {string} unit - The unit of the parameter
 */

function addConnectionRow(tbody, paramName, stats, unit) {
    const row = document.createElement('tr');
    
    const nameCell = document.createElement('td');
    nameCell.textContent = paramName;
    
    const minCell = document.createElement('td');
    minCell.textContent = formatValue(stats.min);
    
    const maxCell = document.createElement('td');
    maxCell.textContent = formatValue(stats.max);
    
    const meanCell = document.createElement('td');
    meanCell.textContent = formatValue(stats.mean);
    
    // Add std deviation cell
    const stdCell = document.createElement('td');
    stdCell.textContent = formatValue(stats.std);
    
    const sparsityCell = document.createElement('td');
    sparsityCell.textContent = (stats.sparsity * 100).toFixed(2) + '%';
    
    const unitCell = document.createElement('td');
    unitCell.textContent = unit;
    
    row.appendChild(nameCell);
    row.appendChild(minCell);
    row.appendChild(maxCell);
    row.appendChild(meanCell);
    row.appendChild(stdCell); // Add the std cell to the row
    row.appendChild(sparsityCell);
    row.appendChild(unitCell);
    
    tbody.appendChild(row);
}

/**
 * Set up the plots section with categorized plots
 * @param {Object} data - The processed model data
 */
function setupPlotsSection(data) {
    const plotsTab = document.getElementById('plots-tab');
    if (!plotsTab || !data.plots || data.plots.length === 0) {
        return;
    }
    
    // Clear existing content
    plotsTab.innerHTML = '<h3>Parameter Distribution Plots</h3>';
    
    // Create container for the tabs
    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'parameter-tabs';
    
    // Create tab buttons
    const dimensionlessTabBtn = document.createElement('button');
    dimensionlessTabBtn.className = 'parameter-tab-btn active';
    dimensionlessTabBtn.textContent = 'Dimensionless Parameters';
    dimensionlessTabBtn.dataset.tab = 'dimensionless';
    
    const physicalTabBtn = document.createElement('button');
    physicalTabBtn.className = 'parameter-tab-btn';
    physicalTabBtn.textContent = 'Physical Parameters';
    physicalTabBtn.dataset.tab = 'physical';
    
    tabsContainer.appendChild(dimensionlessTabBtn);
    tabsContainer.appendChild(physicalTabBtn);
    plotsTab.appendChild(tabsContainer);
    
    // Create containers for both parameter types
    const dimensionlessContainer = document.createElement('div');
    dimensionlessContainer.className = 'parameter-tab-content active';
    dimensionlessContainer.id = 'dimensionless-parameters';
    
    const physicalContainer = document.createElement('div');
    physicalContainer.className = 'parameter-tab-content';
    physicalContainer.id = 'physical-parameters';
    
    // Create dashboard layout for both types
    const dimensionlessDashboard = document.createElement('div');
    dimensionlessDashboard.className = 'plot-dashboard';
    dimensionlessContainer.appendChild(dimensionlessDashboard);
    
    const physicalDashboard = document.createElement('div');
    physicalDashboard.className = 'plot-dashboard';
    physicalContainer.appendChild(physicalDashboard);
    
    // Create navigation sidebars for both types
    const dimensionlessNav = document.createElement('div');
    dimensionlessNav.className = 'plot-nav';
    dimensionlessDashboard.appendChild(dimensionlessNav);
    
    const physicalNav = document.createElement('div');
    physicalNav.className = 'plot-nav';
    physicalDashboard.appendChild(physicalNav);
    
    // Create display areas for both types
    const dimensionlessDisplay = document.createElement('div');
    dimensionlessDisplay.className = 'plot-display';
    dimensionlessDashboard.appendChild(dimensionlessDisplay);
    
    const physicalDisplay = document.createElement('div');
    physicalDisplay.className = 'plot-display';
    physicalDashboard.appendChild(physicalDisplay);
    
    // Add image containers for both types
    createImageContainer(dimensionlessDisplay, 'dimensionless');
    createImageContainer(physicalDisplay, 'physical');
    
    // Add tabs to the document
    plotsTab.appendChild(dimensionlessContainer);
    plotsTab.appendChild(physicalContainer);
    
    // Add tab switching behavior
    dimensionlessTabBtn.addEventListener('click', function() {
        dimensionlessTabBtn.classList.add('active');
        physicalTabBtn.classList.remove('active');
        dimensionlessContainer.classList.add('active');
        physicalContainer.classList.remove('active');
    });
    
    physicalTabBtn.addEventListener('click', function() {
        physicalTabBtn.classList.add('active');
        dimensionlessTabBtn.classList.remove('active');
        physicalContainer.classList.add('active');
        dimensionlessContainer.classList.remove('active');
    });
    
    // Helper function to create image container
    function createImageContainer(displayElement, type) {
        const container = document.createElement('div');
        container.className = 'plot-image-container';
        container.innerHTML = `
            <img id="plot-image-${type}" src="" alt="Parameter distribution plot" class="plot-image">
            <div class="plot-controls">
                <button id="download-plot-${type}" class="plot-control-btn" title="Download Plot">
                    <svg viewBox="0 0 24 24" width="24" height="24">
                        <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                    </svg>
                </button>
            </div>
            <div class="plot-title">Select a plot from the sidebar</div>
        `;
        displayElement.appendChild(container);
    }
    
    // Fetch parameter categories JSON
    const paramCategoriesUrl = data.modelId ? `/model_plots/${data.modelId}/histograms/param_categories.json` : '';
    
    if (paramCategoriesUrl) {
        // Fetch the parameter categories
        fetch(paramCategoriesUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to fetch parameter categories");
                }
                return response.json();
            })
            .then(categories => {
                // Check if we have the new separated categories format
                if (categories.dimensionless && categories.physical) {
                    // Build structured navigation for both parameter types
                    createStructuredCategoryNav(
                        dimensionlessNav,
                        categories.dimensionless,
                        data.plots.filter(p => !p.path.includes('_physical.png')),
                        'dimensionless'
                    );
                    
                    createStructuredCategoryNav(
                        physicalNav,
                        categories.physical,
                        data.plots.filter(p => p.path.includes('_physical.png')),
                        'physical'
                    );
                    
                    // Select the first plot by default for both types
                    const dimensionlessPlots = data.plots.filter(p => !p.path.includes('_physical.png'));
                    const physicalPlots = data.plots.filter(p => p.path.includes('_physical.png'));
                    
                    if (dimensionlessPlots.length > 0) {
                        updatePlotImage(dimensionlessPlots[0].path, 'dimensionless');
                        dimensionlessDisplay.querySelector('.plot-title').textContent = dimensionlessPlots[0].name;
                    }
                    
                    if (physicalPlots.length > 0) {
                        updatePlotImage(physicalPlots[0].path, 'physical');
                        physicalDisplay.querySelector('.plot-title').textContent = physicalPlots[0].name;
                    }
                } else {
                    // Legacy format - sort plots into dimensionless and physical
                    handleLegacyPlotStructure(data, dimensionlessNav, physicalNav, dimensionlessDisplay, physicalDisplay);
                }
            })
            .catch(error => {
                console.warn("Using flat plot navigation:", error);
                handleLegacyPlotStructure(data, dimensionlessNav, physicalNav, dimensionlessDisplay, physicalDisplay);
            });
    } else {
        handleLegacyPlotStructure(data, dimensionlessNav, physicalNav, dimensionlessDisplay, physicalDisplay);
    }
    
    // Setup download buttons for plots
    setupDownloadButton('dimensionless');
    setupDownloadButton('physical');
}

/**
 * Helper function to handle legacy plot structure
 */
function handleLegacyPlotStructure(data, dimensionlessNav, physicalNav, dimensionlessDisplay, physicalDisplay) {
    // Sort plots into dimensionless and physical based on filename
    const dimensionlessPlots = data.plots.filter(p => !p.path.includes('_physical.png'));
    const physicalPlots = data.plots.filter(p => p.path.includes('_physical.png'));
    
    // Create flat navigation
    createFlatPlotNav(dimensionlessNav, dimensionlessPlots, 'dimensionless');
    createFlatPlotNav(physicalNav, physicalPlots, 'physical');
    
    // Select default plots
    if (dimensionlessPlots.length > 0) {
        updatePlotImage(dimensionlessPlots[0].path, 'dimensionless');
        dimensionlessDisplay.querySelector('.plot-title').textContent = dimensionlessPlots[0].name;
    }
    
    if (physicalPlots.length > 0) {
        updatePlotImage(physicalPlots[0].path, 'physical');
        physicalDisplay.querySelector('.plot-title').textContent = physicalPlots[0].name;
    }
}

/**
 * Create structured navigation based on parameter categories
 * @param {HTMLElement} navContainer - The container for the navigation
 * @param {Object} categories - The parameter categories structure
 * @param {Array} plots - The list of available plots
 * @param {string} plotType - The type of plot ('dimensionless' or 'physical')
 */
function createStructuredCategoryNav(navContainer, categories, plots, plotType) {
    // Create Layer section
    if (categories.layer && Object.keys(categories.layer).length > 0) {
        const layersSection = document.createElement('div');
        layersSection.className = 'plot-nav-section';
        
        const layersHeader = document.createElement('h4');
        layersHeader.textContent = 'Layers';
        layersSection.appendChild(layersHeader);
        
        // Add each layer as a collapsible section
        Object.keys(categories.layer).sort((a, b) => parseInt(a) - parseInt(b)).forEach(layerId => {
            const layerData = categories.layer[layerId];
            
            // Create a group for this layer
            const layerGroup = document.createElement('div');
            layerGroup.className = 'plot-nav-group';
            
            const layerTitle = document.createElement('div');
            layerTitle.className = 'plot-nav-group-title';
            layerTitle.innerHTML = `Layer ${layerId} <span class="toggle-icon">▼</span>`;
            layerTitle.addEventListener('click', function() {
                this.classList.toggle('collapsed');
                this.nextElementSibling.classList.toggle('collapsed');
                this.querySelector('.toggle-icon').textContent = 
                    this.classList.contains('collapsed') ? '▶' : '▼';
            });
            
            const layerItems = document.createElement('div');
            layerItems.className = 'plot-nav-group-items';
            
            // Add each parameter as a link
            if (Array.isArray(layerData)) {
                layerData.forEach(param => {
                    // Construct the plot path based on the type
                    let plotPath;
                    if (plotType === 'dimensionless') {
                        plotPath = `/model_plots/${currentModelData.modelId}/histograms/layer_${layerId}_${param}.png`;
                    } else {
                        plotPath = `/model_plots/${currentModelData.modelId}/histograms/layer_${layerId}_${param}_physical.png`;
                    }
                    
                    const matchingPlot = plots.find(p => p.path === plotPath);
                    
                    if (matchingPlot) {
                        const plotLink = document.createElement('a');
                        plotLink.href = '#';
                        plotLink.className = 'plot-nav-item';
                        plotLink.dataset.path = matchingPlot.path;
                        plotLink.dataset.name = matchingPlot.name || `Layer ${layerId} - ${param}`;
                        plotLink.textContent = formatParamName(param);
                        
                        plotLink.addEventListener('click', function(e) {
                            e.preventDefault();
                            // Only update items in the same tab
                            document.querySelectorAll(`#${plotType}-parameters .plot-nav-item`).forEach(item => {
                                item.classList.remove('active');
                            });
                            this.classList.add('active');
                            
                            updatePlotImage(this.dataset.path, plotType);
                            document.querySelector(`#${plotType}-parameters .plot-title`).textContent = this.dataset.name;
                        });
                        
                        layerItems.appendChild(plotLink);
                    }
                });
            }
            
            // Only add the group if it has items
            if (layerItems.children.length > 0) {
                layerGroup.appendChild(layerTitle);
                layerGroup.appendChild(layerItems);
                layersSection.appendChild(layerGroup);
            }
        });
        
        navContainer.appendChild(layersSection);
    }
    
    // Create Connection section
    if (categories.connection && Object.keys(categories.connection).length > 0) {
        const connectionsSection = document.createElement('div');
        connectionsSection.className = 'plot-nav-section';
        
        const connectionsHeader = document.createElement('h4');
        connectionsHeader.textContent = 'Connections';
        connectionsSection.appendChild(connectionsHeader);
        
        // Add each connection as a collapsible section
        Object.keys(categories.connection).forEach(connId => {
            const connParams = categories.connection[connId];
            
            if (connParams.length > 0) {
                const connGroup = document.createElement('div');
                connGroup.className = 'plot-nav-group';
                
                const connTitle = document.createElement('div');
                connTitle.className = 'plot-nav-group-title';
                connTitle.innerHTML = `Connection ${connId} <span class="toggle-icon">▼</span>`;
                connTitle.addEventListener('click', function() {
                    this.classList.toggle('collapsed');
                    this.nextElementSibling.classList.toggle('collapsed');
                    this.querySelector('.toggle-icon').textContent = 
                        this.classList.contains('collapsed') ? '▶' : '▼';
                });
                
                const connItems = document.createElement('div');
                connItems.className = 'plot-nav-group-items';
                
                // Add each parameter as a link
                connParams.forEach(param => {
                    // Construct the plot path based on the type
                    let plotPath;
                    if (plotType === 'dimensionless') {
                        plotPath = `/model_plots/${currentModelData.modelId}/histograms/connection_${connId}_${param}.png`;
                    } else {
                        plotPath = `/model_plots/${currentModelData.modelId}/histograms/connection_${connId}_${param}_physical.png`;
                    }
                    
                    const matchingPlot = plots.find(p => p.path === plotPath);
                    
                    if (matchingPlot) {
                        const plotLink = document.createElement('a');
                        plotLink.href = '#';
                        plotLink.className = 'plot-nav-item';
                        plotLink.dataset.path = matchingPlot.path;
                        plotLink.dataset.name = matchingPlot.name || `Connection ${connId} - ${param}`;
                        plotLink.textContent = formatParamName(param);
                        
                        plotLink.addEventListener('click', function(e) {
                            e.preventDefault();
                            document.querySelectorAll(`#${plotType}-parameters .plot-nav-item`).forEach(item => {
                                item.classList.remove('active');
                            });
                            this.classList.add('active');
                            
                            updatePlotImage(this.dataset.path, plotType);
                            document.querySelector(`#${plotType}-parameters .plot-title`).textContent = this.dataset.name;
                        });
                        
                        connItems.appendChild(plotLink);
                    }
                });
                
                // Only add if there are items
                if (connItems.children.length > 0) {
                    connGroup.appendChild(connTitle);
                    connGroup.appendChild(connItems);
                    connectionsSection.appendChild(connGroup);
                }
            }
        });
        
        // Only add if there are items
        if (connectionsSection.children.length > 1) { // More than just the header
            navContainer.appendChild(connectionsSection);
        }
    }
    
    // If no items were added, add a message
    if (navContainer.children.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.className = 'empty-message';
        emptyMessage.textContent = `No ${plotType} parameter plots available`;
        navContainer.appendChild(emptyMessage);
    }
}

/**
 * Create a flat navigation without categories
 * @param {HTMLElement} navContainer - The container for the navigation
 * @param {Array} plots - The array of available plots
 * @param {string} plotType - The type of plot ('dimensionless' or 'physical')
 */
function createFlatPlotNav(navContainer, plots, plotType) {
    const plotList = document.createElement('div');
    plotList.className = 'plot-nav-list';
    
    // Group plots by type and organize by layer/connection ID
    const layerPlots = {};
    const connectionPlots = {};
    const otherPlots = [];
    
    // Process each plot and organize them
    plots.forEach(plot => {
        const plotName = plot.name.toLowerCase();
        
        // Extract info from the plot name using regex
        const layerMatch = plotName.match(/layer\s*(\d+)/i);
        const connectionMatch = plotName.match(/connection\s*([\w_]+)/i);
        
        if (layerMatch) {
            const layerId = layerMatch[1];
            if (!layerPlots[layerId]) {
                layerPlots[layerId] = [];
            }
            layerPlots[layerId].push(plot);
        } else if (connectionMatch) {
            const connId = connectionMatch[1];
            if (!connectionPlots[connId]) {
                connectionPlots[connId] = [];
            }
            connectionPlots[connId].push(plot);
        } else {
            otherPlots.push(plot);
        }
    });
    
    // Create a function to add a plot link
    const addPlotLink = (container, plot) => {
        const plotLink = document.createElement('a');
        plotLink.href = '#';
        plotLink.className = 'plot-nav-item';
        plotLink.dataset.path = plot.path;
        plotLink.dataset.name = plot.name;
        
        // Extract parameter name from the plot name
        let displayName = plot.name;
        const paramMatch = plot.name.match(/(layer|connection)\s*[\w_]+\s*-\s*(.*)/i);
        if (paramMatch) {
            displayName = formatParamName(paramMatch[2].trim());
            // Remove "(Physical)" suffix if it's in the physical tab
            if (plotType === 'physical' && displayName.endsWith('(Physical)')) {
                displayName = displayName.replace('(Physical)', '').trim();
            }
        }
        
        plotLink.textContent = displayName;
        
        plotLink.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelectorAll(`#${plotType}-parameters .plot-nav-item`).forEach(item => {
                item.classList.remove('active');
            });
            this.classList.add('active');
            
            updatePlotImage(this.dataset.path, plotType);
            document.querySelector(`#${plotType}-parameters .plot-title`).textContent = this.dataset.name;
        });
        
        container.appendChild(plotLink);
        return plotLink;
    };
    
    // Add layers section with collapsible groups
    if (Object.keys(layerPlots).length > 0) {
        const layersHeader = document.createElement('h4');
        layersHeader.textContent = 'Layers';
        plotList.appendChild(layersHeader);
        
        // Add each layer as a collapsible group
        Object.keys(layerPlots).sort((a, b) => parseInt(a) - parseInt(b)).forEach(layerId => {
            const layerGroup = document.createElement('div');
            layerGroup.className = 'plot-nav-group';
            
            const layerTitle = document.createElement('div');
            layerTitle.className = 'plot-nav-group-title';
            layerTitle.innerHTML = `Layer ${layerId} <span class="toggle-icon">▼</span>`;
            layerTitle.addEventListener('click', function() {
                this.classList.toggle('collapsed');
                this.nextElementSibling.classList.toggle('collapsed');
                this.querySelector('.toggle-icon').textContent = 
                    this.classList.contains('collapsed') ? '▶' : '▼';
            });
            
            const layerItems = document.createElement('div');
            layerItems.className = 'plot-nav-group-items';
            
            // Sort plots by parameter name
            layerPlots[layerId].sort((a, b) => {
                // Extract parameter name (after the hyphen)
                const paramA = a.name.split('-')[1]?.trim() || a.name;
                const paramB = b.name.split('-')[1]?.trim() || b.name;
                return paramA.localeCompare(paramB);
            });
            
            // Add all plots for this layer
            layerPlots[layerId].forEach(plot => {
                addPlotLink(layerItems, plot);
            });
            
            // Add the group to the list
            layerGroup.appendChild(layerTitle);
            layerGroup.appendChild(layerItems);
            plotList.appendChild(layerGroup);
        });
    }
    
    // Add connections section with collapsible groups
    if (Object.keys(connectionPlots).length > 0) {
        const connectionsHeader = document.createElement('h4');
        connectionsHeader.textContent = 'Connections';
        plotList.appendChild(connectionsHeader);
        
        // Add each connection as a collapsible group
        Object.keys(connectionPlots).sort().forEach(connId => {
            const connGroup = document.createElement('div');
            connGroup.className = 'plot-nav-group';
            
            const connTitle = document.createElement('div');
            connTitle.className = 'plot-nav-group-title';
            connTitle.innerHTML = `Connection ${connId} <span class="toggle-icon">▼</span>`;
            connTitle.addEventListener('click', function() {
                this.classList.toggle('collapsed');
                this.nextElementSibling.classList.toggle('collapsed');
                this.querySelector('.toggle-icon').textContent = 
                    this.classList.contains('collapsed') ? '▶' : '▼';
            });
            
            const connItems = document.createElement('div');
            connItems.className = 'plot-nav-group-items';
            
            // Sort plots by parameter name
            connectionPlots[connId].sort((a, b) => {
                // Extract parameter name (after the hyphen)
                const paramA = a.name.split('-')[1]?.trim() || a.name;
                const paramB = b.name.split('-')[1]?.trim() || b.name;
                return paramA.localeCompare(paramB);
            });
            
            // Add all plots for this connection
            connectionPlots[connId].forEach(plot => {
                addPlotLink(connItems, plot);
            });
            
            // Add the group to the list
            connGroup.appendChild(connTitle);
            connGroup.appendChild(connItems);
            plotList.appendChild(connGroup);
        });
    }
    
    // Add other plots if any
    if (otherPlots.length > 0) {
        const otherHeader = document.createElement('h4');
        otherHeader.textContent = 'Other Plots';
        plotList.appendChild(otherHeader);
        
        otherPlots.forEach(plot => {
            addPlotLink(plotList, plot);
        });
    }
    
    // If no plots, add a message
    if (plots.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.className = 'empty-message';
        emptyMessage.textContent = `No ${plotType} parameter plots available`;
        plotList.appendChild(emptyMessage);
    } else {
        // Activate the first plot
        const firstLink = plotList.querySelector('.plot-nav-item');
        if (firstLink) {
            firstLink.classList.add('active');
        }
    }
    
    navContainer.appendChild(plotList);
}

/**
 * Format parameter name for display
 * @param {string} paramName - The raw parameter name
 * @returns {string} A formatted parameter name
 */
function formatParamName(paramName) {
    // Check if it already has "(Physical)" suffix
    const hasPhysicalSuffix = paramName.includes('_physical');
    
    // Remove _physical suffix if present
    let cleanName = paramName.replace('_physical', '');
    
    // Replace underscores with spaces and capitalize
    let formatted = cleanName.replace(/_/g, ' ');
    
    // Capitalize first letter of each word
    formatted = formatted.replace(/\b\w/g, l => l.toUpperCase());
    
    // Special formatting for known abbreviations
    formatted = formatted.replace(/\bL\b/g, 'L (inductance)');
    formatted = formatted.replace(/\bJ\b/g, 'J (coupling)');
    formatted = formatted.replace(/\bM\b/g, 'M (mutual inductance)');
    
    // Add Physical suffix if it had _physical
    if (hasPhysicalSuffix) {
        formatted += ' (Physical)';
    }
    
    return formatted;
}

/**
 * Update the plot image for the selected plot
 * @param {string} plotPath - The path to the plot image
 * @param {string} plotType - The type of plot ('dimensionless' or 'physical')
 */
function updatePlotImage(plotPath, plotType = 'dimensionless') {
    const plotImage = document.getElementById(`plot-image-${plotType}`);
    const plotDisplay = document.querySelector(`#${plotType}-parameters .plot-display`);
    
    if (plotImage && plotDisplay) {
        // Show loading state
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'plot-loading';
        loadingIndicator.textContent = 'Generating plot...';
        
        // Remove any existing image
        if (plotImage.parentNode) {
            // Create a completely new image element to avoid any browser caching
            const newImage = document.createElement('img');
            newImage.id = `plot-image-${plotType}`;
            newImage.className = 'plot-image';
            newImage.alt = 'Parameter distribution plot';
            
            // Replace the old image with the loading indicator
            plotImage.replaceWith(loadingIndicator);
            
            // Clean the plotPath (remove any existing timestamps)
            let newPath = plotPath.split('?')[0];
            
            // Generate a unique identifier to prevent browser caching
            const randomId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            const timestamp = new Date().getTime();
            
            // Add a timestamp to force reload and avoid caching
            newPath += '?ts=' + timestamp + '&uid=' + randomId;
            
            console.log(`Loading ${plotType} plot:`, newPath);
            
            // Wait a moment to ensure the server doesn't reuse previous plot
            setTimeout(() => {
                // Set up load handler for new image
                newImage.onload = function() {
                    // Replace loading indicator with the new image once loaded
                    loadingIndicator.replaceWith(newImage);
                };
                
                // Set the source for the new image
                newImage.src = newPath;
                
                // If image fails to load after 10 seconds, show error
                setTimeout(() => {
                    if (!newImage.complete) {
                        loadingIndicator.textContent = 'Error loading plot. Please try again.';
                    }
                }, 10000);
            }, 200);
        }
    }
}

/**
 * Set up download button for a plot image
 * @param {string} plotType - The type of plot ('dimensionless' or 'physical')
 */
function setupDownloadButton(plotType) {
    const downloadBtn = document.getElementById(`download-plot-${plotType}`);
    const plotImg = document.getElementById(`plot-image-${plotType}`);
    
    if (!downloadBtn || !plotImg) {
        return;
    }
    
    // Download handler
    downloadBtn.addEventListener('click', function() {
        if (plotImg.src) {
            // Create a temporary link to download the image
            const link = document.createElement('a');
            
            // Ensure we're using a clean URL without timestamp params
            let cleanUrl = plotImg.src.split('?')[0];
            
            // If URL is relative, ensure it starts with the right base
            if (!cleanUrl.startsWith('http')) {
                // If using local server path, use the full URL
                cleanUrl = window.location.origin + cleanUrl;
            }
            
            link.href = cleanUrl;
            
            // Extract filename from path or use a default name
            let filename = `parameter_plot_${plotType}.png`;
            try {
                // Split by slashes and take the last part
                const pathSegments = cleanUrl.split('/');
                filename = pathSegments[pathSegments.length - 1];
                
                // Ensure it has .png extension
                if (!filename.endsWith('.png')) {
                    filename += '.png';
                }
            } catch (e) {
                console.error("Error extracting filename:", e);
            }
            
            console.log(`Downloading ${plotType} image:`, filename, "URL:", cleanUrl);
            link.download = filename;
            document.body.appendChild(link); // Required for Firefox
            link.click();
            document.body.removeChild(link); // Clean up
        }
    });
}

/**
 * Export data according to user selections
 */
function exportData() {
    if (!currentModelData) {
        return;
    }
    
    const exportFormat = document.querySelector('input[name="export-format"]:checked').value;
    const exportSelections = Array.from(document.querySelectorAll('input[name="export-data"]:checked')).map(el => el.value);
    
    // Create export data object
    const exportData = {
        format: exportFormat,
        selections: exportSelections,
        modelId: currentModelData.modelId
    };
    
    // Make API call to export data
    fetch('/export_model_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Show success message with download link if applicable
        const exportStatus = document.getElementById('export-status');
        if (exportStatus) {
            if (data.download_url) {
                exportStatus.innerHTML = `
                    <div class="success-message">
                        Data exported successfully!
                        <a href="${data.download_url}" download class="download-link">Download</a>
                    </div>
                `;
            } else {
                exportStatus.innerHTML = `
                    <div class="success-message">
                        Data exported successfully!
                    </div>
                `;
            }
        }
    })
    .catch(error => {
        // Show error message
        const exportStatus = document.getElementById('export-status');
        if (exportStatus) {
            exportStatus.innerHTML = `
                <div class="error-message">
                    Error exporting data: ${error.message}
                </div>
            `;
        }
        console.error('Error:', error);
    });
}

/**
 * Calculate statistics for an array of values
 * @param {Array} values - Array of numeric values
 * @param {boolean} checkSparsity - Whether to calculate sparsity
 * @returns {Object} Object containing min, max, mean, and optionally sparsity
 */

function calculateStatistics(values, checkSparsity = false) {
    // Flatten the array if it's multi-dimensional
    const flatValues = values.flat ? values.flat(Infinity) : values;
    
    // Filter out any non-numeric values
    const numericValues = flatValues.filter(v => typeof v === 'number' && !isNaN(v));
    
    // Return default values if no numeric values are found
    if (numericValues.length === 0) {
        return { min: 0, max: 0, mean: 0, std: 0, sparsity: 0 };
    }
    
    // Calculate statistics
    const min = Math.min(...numericValues);
    const max = Math.max(...numericValues);
    const sum = numericValues.reduce((acc, val) => acc + val, 0);
    const mean = sum / numericValues.length;
    
    // Calculate standard deviation
    const squareDiffs = numericValues.map(value => {
        const diff = value - mean;
        return diff * diff;
    });
    const avgSquareDiff = squareDiffs.reduce((acc, val) => acc + val, 0) / numericValues.length;
    const std = Math.sqrt(avgSquareDiff);
    
    // Calculate sparsity if requested
    let sparsity = 0;
    if (checkSparsity) {
        const zeroCount = numericValues.filter(v => v === 0).length;
        sparsity = zeroCount / numericValues.length;
    }
    
    return {
        min,
        max,
        mean,
        std,
        sparsity
    };
}

/**
 * Format a numeric value for display
 * @param {number} value - The numeric value to format
 * @returns {string} Formatted value
 */
function formatValue(value) {
    if (value === undefined || value === null) {
        return 'N/A';
    }
    
    // Check if it's a very small or very large number
    const absValue = Math.abs(value);
    
    if (absValue === 0) {
        return '0';
    } else if (absValue < 0.0001 || absValue > 10000) {
        // Use scientific notation for very small or very large numbers
        return value.toExponential(4);
    } else {
        // Use fixed notation with appropriate precision
        return value.toPrecision(5);
    }
}

/**
 * Add a row to a table with key-value pair
 * @param {HTMLElement} table - The table element
 * @param {string} key - The key (left column)
 * @param {string|number} value - The value (right column)
 */
function addTableRow(table, key, value) {
    const row = document.createElement('tr');
    
    const keyCell = document.createElement('td');
    keyCell.textContent = key;
    keyCell.className = 'param-name';
    
    const valueCell = document.createElement('td');
    valueCell.className = 'param-value';
    
    // For numeric values, apply special formatting
    if (typeof value === 'number' || (typeof value === 'string' && !isNaN(parseFloat(value)))) {
        // If it's a numeric string with a unit at the end
        const match = typeof value === 'string' && value.match(/^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z\u03A9]+)?$/);
        
        if (match) {
            // Has a numeric part and possibly a unit
            const numericPart = match[1];
            const unitPart = match[2] || '';
            
            valueCell.innerHTML = `<code>${numericPart}</code> ${unitPart}`;
        } else {
            valueCell.innerHTML = `<code>${value}</code>`;
        }
    } else {
        valueCell.textContent = value;
    }
    
    row.appendChild(keyCell);
    row.appendChild(valueCell);
    
    table.appendChild(row);
}

/**
 * Check if a value is an ndarray
 * @param {*} value - The value to check
 * @returns {boolean} True if the value is an ndarray
 */
function isNdarray(value) {
    return value && typeof value === 'object' && value.hasOwnProperty('shape') && value.hasOwnProperty('data');
}