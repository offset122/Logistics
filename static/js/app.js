// Urban Logistics Optimizer JavaScript

class UrbanLogisticsApp {
    constructor() {
        this.currentRunId = null;
        this.optimizationInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadOptimizationHistory();
    }

    setupEventListeners() {
        // File upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            this.handleFileUpload(e);
        });

        // Start optimization button
        document.getElementById('startOptimization').addEventListener('click', () => {
            this.startOptimization();
        });

        // History items click handlers will be added dynamically
    }

    async handleFileUpload(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const fileInputs = ['nodes', 'demands', 'vehicles', 'metro', 'params'];
        
        let hasFiles = false;
        fileInputs.forEach(inputName => {
            const fileInput = document.querySelector(`input[name="${inputName}"]`);
            if (fileInput.files[0]) {
                formData.append(inputName, fileInput.files[0]);
                hasFiles = true;
            }
        });

        if (!hasFiles) {
            this.showAlert('Please select at least one file to upload.', 'warning');
            return;
        }

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.showAlert(`Successfully uploaded ${result.files.length} files.`, 'success');
                this.updateUploadStatus(result.files);
            } else {
                this.showAlert('File upload failed. Please try again.', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('An error occurred during file upload.', 'error');
        }
    }

    async startOptimization() {
        // Get selected algorithms
        const selectedAlgorithms = [];
        for (let i = 1; i <= 6; i++) {
            if (document.getElementById(`alg${i}`).checked) {
                selectedAlgorithms.push(i);
            }
        }

        if (selectedAlgorithms.length === 0) {
            this.showAlert('Please select at least one algorithm.', 'warning');
            return;
        }

        // Get optimization settings
        const config = {
            algorithms: selectedAlgorithms,
            max_iterations: parseInt(document.getElementById('maxIterations').value),
            time_limit: parseInt(document.getElementById('timeLimit').value),
            tolerance: parseFloat(document.getElementById('tolerance').value)
        };

        // Show loading modal
        const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        loadingModal.show();

        // Disable start button
        const startBtn = document.getElementById('startOptimization');
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Optimizing...';

        try {
            const response = await fetch('/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.currentRunId = result.run_id;
                this.displayResults(result.results);
                this.showAlert('Optimization completed successfully!', 'success');
                this.loadOptimizationHistory(); // Refresh history
            } else {
                this.showAlert(`Optimization failed: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('Optimization error:', error);
            this.showAlert('An error occurred during optimization.', 'error');
        } finally {
            // Hide loading modal and restore button
            loadingModal.hide();
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Optimization';
        }
    }

    displayResults(results) {
        // Update status panel
        this.updateStatusPanel('success', results);

        // Display summary
        this.displaySummary(results);

        // Display routes
        this.displayRoutes(results.routes);

        // Display metro schedules
        this.displayMetroSchedules(results.metro_schedules);

        // Display performance metrics
        this.displayPerformanceMetrics(results);

        // Add export button
        this.addExportButton();
    }

    updateStatusPanel(status, results) {
        const statusPanel = document.getElementById('statusPanel');
        
        if (status === 'success') {
            statusPanel.innerHTML = `
                <div class="status-success">
                    <i class="fas fa-check-circle fa-2x mb-2"></i>
                    <h5>Optimization Completed Successfully</h5>
                    <p class="mb-0">
                        Objective Value: <strong>${results.objective_value.toFixed(2)}</strong> | 
                        Runtime: <strong>${results.runtime.toFixed(2)}s</strong> | 
                        Iterations: <strong>${results.iterations}</strong>
                    </p>
                </div>
            `;
        }
    }

    displaySummary(results) {
        const summaryContent = document.getElementById('summaryContent');
        
        const droneRoutes = results.routes.filter(r => r.type === 'drone').length;
        const truckRoutes = results.routes.filter(r => r.type === 'truck').length;
        const metroSchedules = results.metro_schedules.length;
        
        summaryContent.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${results.objective_value.toFixed(2)}</div>
                        <div class="metric-label">Total Cost</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${droneRoutes}</div>
                        <div class="metric-label">Drone Routes</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${truckRoutes}</div>
                        <div class="metric-label">Truck Routes</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${metroSchedules}</div>
                        <div class="metric-label">Metro Schedules</div>
                    </div>
                </div>
            </div>
            
            <div class="mt-4">
                <h6>Optimization Details</h6>
                <div class="table-responsive">
                    <table class="table table-sm results-table">
                        <tbody>
                            <tr>
                                <td><strong>Status:</strong></td>
                                <td><span class="badge bg-success">Completed</span></td>
                            </tr>
                            <tr>
                                <td><strong>Runtime:</strong></td>
                                <td>${results.runtime.toFixed(2)} seconds</td>
                            </tr>
                            <tr>
                                <td><strong>Iterations:</strong></td>
                                <td>${results.iterations}</td>
                            </tr>
                            <tr>
                                <td><strong>Columns Generated:</strong></td>
                                <td>${results.columns_generated || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td><strong>Algorithm Status:</strong></td>
                                <td>${results.status}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    displayRoutes(routes) {
        const routesContent = document.getElementById('routesContent');
        
        if (!routes || routes.length === 0) {
            routesContent.innerHTML = '<p class="text-muted">No routes found in the solution.</p>';
            return;
        }

        let routesHtml = '<div class="row">';
        
        // Group routes by type
        const droneRoutes = routes.filter(r => r.type === 'drone');
        const truckRoutes = routes.filter(r => r.type === 'truck');
        
        // Drone routes
        if (droneRoutes.length > 0) {
            routesHtml += '<div class="col-md-6"><h6 class="text-success"><i class="fas fa-helicopter me-2"></i>Drone Routes</h6>';
            droneRoutes.forEach((route, index) => {
                const details = route.details || {};
                routesHtml += `
                    <div class="route-item drone">
                        <div><strong>Route ${index + 1}</strong></div>
                        <div class="small">Path: ${(details.route || []).join(' → ')}</div>
                        <div class="small">Distance: ${(details.total_distance || 0).toFixed(2)} km</div>
                        <div class="small">Energy: ${(details.energy_consumption || 0).toFixed(2)} units</div>
                        <div class="small">Cost: $${route.cost.toFixed(2)}</div>
                    </div>
                `;
            });
            routesHtml += '</div>';
        }
        
        // Truck routes
        if (truckRoutes.length > 0) {
            routesHtml += '<div class="col-md-6"><h6 class="text-warning"><i class="fas fa-truck me-2"></i>Truck Routes</h6>';
            truckRoutes.forEach((route, index) => {
                const details = route.details || {};
                routesHtml += `
                    <div class="route-item truck">
                        <div><strong>Route ${index + 1}</strong></div>
                        <div class="small">Path: ${(details.route || []).join(' → ')}</div>
                        <div class="small">Distance: ${(details.total_distance || 0).toFixed(2)} km</div>
                        <div class="small">Time: ${(details.total_time || 0).toFixed(2)} min</div>
                        <div class="small">Cost: $${route.cost.toFixed(2)}</div>
                    </div>
                `;
            });
            routesHtml += '</div>';
        }
        
        routesHtml += '</div>';
        routesContent.innerHTML = routesHtml;
    }

    displayMetroSchedules(schedules) {
        const metroContent = document.getElementById('metroContent');
        
        if (!schedules || schedules.length === 0) {
            metroContent.innerHTML = '<p class="text-muted">No metro schedules found in the solution.</p>';
            return;
        }

        const upSchedules = schedules.filter(s => s.direction === 'up');
        const downSchedules = schedules.filter(s => s.direction === 'down');
        
        let metroHtml = '<div class="row">';
        
        // Up-line schedules
        if (upSchedules.length > 0) {
            metroHtml += '<div class="col-md-6"><h6 class="text-primary"><i class="fas fa-arrow-up me-2"></i>Up-line Schedules</h6>';
            upSchedules.forEach((schedule, index) => {
                metroHtml += `
                    <div class="route-item metro">
                        <div><strong>Schedule ${index + 1}</strong></div>
                        <div class="small">Departure: ${this.formatTime(schedule.departure)}</div>
                        <div class="small">Arrival: ${this.formatTime(schedule.arrival)}</div>
                        <div class="small">Load: ${schedule.load} units</div>
                        <div class="small">Cost: $${schedule.cost.toFixed(2)}</div>
                    </div>
                `;
            });
            metroHtml += '</div>';
        }
        
        // Down-line schedules
        if (downSchedules.length > 0) {
            metroHtml += '<div class="col-md-6"><h6 class="text-info"><i class="fas fa-arrow-down me-2"></i>Down-line Schedules</h6>';
            downSchedules.forEach((schedule, index) => {
                metroHtml += `
                    <div class="route-item metro">
                        <div><strong>Schedule ${index + 1}</strong></div>
                        <div class="small">Departure: ${this.formatTime(schedule.departure)}</div>
                        <div class="small">Arrival: ${this.formatTime(schedule.arrival)}</div>
                        <div class="small">Load: ${schedule.load} units</div>
                        <div class="small">Cost: $${schedule.cost.toFixed(2)}</div>
                    </div>
                `;
            });
            metroHtml += '</div>';
        }
        
        metroHtml += '</div>';
        metroContent.innerHTML = metroHtml;
    }

    displayPerformanceMetrics(results) {
        const performanceContent = document.getElementById('performanceContent');
        
        // Create runtime comparison chart if data is available
        const runtimeData = results.runtime_comparison || {};
        
        let performanceHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Algorithm Performance</h6>
                    <div class="table-responsive">
                        <table class="table table-sm results-table">
                            <thead>
                                <tr>
                                    <th>Algorithm</th>
                                    <th>Type</th>
                                    <th>Runtime (s)</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        const algorithms = [
            {id: 1, name: 'ULA-T', type: 'Basic Truck'},
            {id: 2, name: 'ULA-D', type: 'Basic Drone'},
            {id: 3, name: 'SELA-M', type: 'Basic Metro'},
            {id: 4, name: 'BLA-T', type: 'Optimized Truck'},
            {id: 5, name: 'BLA-D', type: 'Optimized Drone'},
            {id: 6, name: 'BALA-M', type: 'Optimized Metro'}
        ];
        
        algorithms.forEach(alg => {
            const runtime = runtimeData[alg.id] ? runtimeData[alg.id][0] || 0 : 0;
            const status = runtime > 0 ? 'Completed' : 'Not Run';
            performanceHtml += `
                <tr>
                    <td><strong>${alg.name}</strong></td>
                    <td>${alg.type}</td>
                    <td>${runtime.toFixed(3)}</td>
                    <td><span class="badge ${status === 'Completed' ? 'bg-success' : 'bg-secondary'}">${status}</span></td>
                </tr>
            `;
        });
        
        performanceHtml += `
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>Iteration Progress</h6>
                    <div class="chart-container">
                        <canvas id="iterationChart"></canvas>
                    </div>
                </div>
            </div>
        `;
        
        performanceContent.innerHTML = performanceHtml;
        
        // Create iteration chart if data is available
        if (results.iteration_log && results.iteration_log.length > 0) {
            this.createIterationChart(results.iteration_log);
        }
    }

    createIterationChart(iterationLog) {
        const ctx = document.getElementById('iterationChart');
        if (!ctx) return;
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: iterationLog.map(log => `Iteration ${log.iteration}`),
                datasets: [{
                    label: 'Objective Value',
                    data: iterationLog.map(log => log.objective),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Objective Value'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Convergence Progress'
                    }
                }
            }
        });
    }

    addExportButton() {
        if (!this.currentRunId) return;
        
        const statusPanel = document.getElementById('statusPanel');
        const exportBtn = document.createElement('div');
        exportBtn.className = 'export-btn';
        exportBtn.innerHTML = `
            <a href="/export/${this.currentRunId}" class="btn btn-outline-success btn-sm">
                <i class="fas fa-download me-1"></i>Export to Excel
            </a>
        `;
        statusPanel.appendChild(exportBtn);
    }

    async loadOptimizationHistory() {
        try {
            const response = await fetch('/history');
            const history = await response.json();
            
            const historyList = document.getElementById('historyList');
            
            if (history.length === 0) {
                historyList.innerHTML = '<div class="text-muted text-center">No optimization history</div>';
                return;
            }
            
            let historyHtml = '';
            history.forEach(run => {
                const timestamp = new Date(run.timestamp).toLocaleString();
                const statusClass = run.status === 'completed' ? 'completed' : 'failed';
                
                historyHtml += `
                    <div class="history-item" onclick="app.loadHistoryResults(${run.id})">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <div><strong>Run #${run.id}</strong></div>
                                <div class="history-timestamp">${timestamp}</div>
                            </div>
                            <div class="text-end">
                                <div class="history-status ${statusClass}">${run.status}</div>
                                <div class="small text-muted">${run.objective_value ? '$' + run.objective_value.toFixed(2) : 'N/A'}</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            historyList.innerHTML = historyHtml;
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    async loadHistoryResults(runId) {
        try {
            const response = await fetch(`/results/${runId}`);
            const data = await response.json();
            
            if (data.run_info) {
                this.currentRunId = runId;
                
                // Convert database results to expected format
                const results = {
                    objective_value: data.run_info.objective_value || 0,
                    runtime: data.run_info.runtime || 0,
                    iterations: data.run_info.iterations || 0,
                    status: data.run_info.status || 'unknown',
                    routes: data.routes || [],
                    metro_schedules: data.metro_schedules || [],
                    columns_generated: 0,
                    iteration_log: [],
                    runtime_comparison: {}
                };
                
                this.displayResults(results);
                this.showAlert(`Loaded results from Run #${runId}`, 'success');
            }
        } catch (error) {
            console.error('Error loading history results:', error);
            this.showAlert('Error loading historical results', 'error');
        }
    }

    updateUploadStatus(files) {
        // Visual feedback for uploaded files
        files.forEach(file => {
            const input = document.querySelector(`input[name="${file.type}"]`);
            if (input) {
                input.classList.add('is-valid');
                setTimeout(() => input.classList.remove('is-valid'), 3000);
            }
        });
    }

    showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of container
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    formatTime(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new UrbanLogisticsApp();
});