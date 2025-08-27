from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import json
import os
from datetime import datetime
from main_bap_simple import SimpleBranchAndPriceOptimizer
from data_loader import DataLoader
import pandas as pd

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('logistics.db')
    cursor = conn.cursor()
    
    # Create tables for storing optimization results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            config TEXT,
            status TEXT,
            objective_value REAL,
            runtime REAL,
            iterations INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS solution_routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            vehicle_type TEXT,
            route_id TEXT,
            details TEXT,
            cost REAL,
            FOREIGN KEY (run_id) REFERENCES optimization_runs (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metro_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            direction TEXT,
            departure_time INTEGER,
            arrival_time INTEGER,
            load INTEGER,
            FOREIGN KEY (run_id) REFERENCES optimization_runs (id)
        )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads for the 5 required Excel inputs"""
    uploaded_files = []
    
    for file_key in ['nodes', 'demands', 'vehicles', 'metro', 'params']:
        if file_key in request.files:
            file = request.files[file_key]
            if file.filename != '':
                filename = f"{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                filepath = os.path.join('data', filename)
                file.save(filepath)
                uploaded_files.append({'type': file_key, 'filename': filename})
    
    return jsonify({'status': 'success', 'files': uploaded_files})

@app.route('/optimize', methods=['POST'])
def run_optimization():
    """Run the Branch-and-Price optimization"""
    try:
        config = request.json
        
        # Create sample data if no files uploaded
        sample_data = create_sample_data()
        
        # Initialize optimizer with sample data
        optimizer = SimpleBranchAndPriceOptimizer(
            sample_data['nodes'], 
            sample_data['demands'], 
            sample_data['vehicles'], 
            sample_data['metro_data'], 
            sample_data['params']
        )
        
        # Run optimization with selected algorithms
        selected_algorithms = config.get('algorithms', [1, 2, 3, 4, 5, 6])
        results = optimizer.solve(algorithms=selected_algorithms)
        
        # Store results in database
        conn = sqlite3.connect('logistics.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO optimization_runs (config, status, objective_value, runtime, iterations)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            json.dumps(config),
            'completed',
            results['objective_value'],
            results['runtime'],
            results['iterations']
        ))
        
        run_id = cursor.lastrowid
        
        # Store route details
        for route in results['routes']:
            cursor.execute('''
                INSERT INTO solution_routes (run_id, vehicle_type, route_id, details, cost)
                VALUES (?, ?, ?, ?, ?)
            ''', (run_id, route['type'], route['id'], json.dumps(route['details']), route['cost']))
        
        # Store metro schedules
        for schedule in results['metro_schedules']:
            cursor.execute('''
                INSERT INTO metro_schedules (run_id, direction, departure_time, arrival_time, load)
                VALUES (?, ?, ?, ?, ?)
            ''', (run_id, schedule['direction'], schedule['departure'], schedule['arrival'], schedule['load']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'run_id': run_id, 'results': results})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/results/<int:run_id>')
def get_results(run_id):
    """Get optimization results for a specific run"""
    conn = sqlite3.connect('logistics.db')
    
    # Get run details
    run_data = pd.read_sql_query('''
        SELECT * FROM optimization_runs WHERE id = ?
    ''', conn, params=(run_id,))
    
    # Get routes
    routes_data = pd.read_sql_query('''
        SELECT * FROM solution_routes WHERE run_id = ?
    ''', conn, params=(run_id,))
    
    # Get metro schedules
    metro_data = pd.read_sql_query('''
        SELECT * FROM metro_schedules WHERE run_id = ?
    ''', conn, params=(run_id,))
    
    conn.close()
    
    return jsonify({
        'run_info': run_data.to_dict('records')[0] if not run_data.empty else None,
        'routes': routes_data.to_dict('records'),
        'metro_schedules': metro_data.to_dict('records')
    })

@app.route('/export/<int:run_id>')
def export_results(run_id):
    """Export results to Excel file"""
    from excel_exporter import ExcelExporter
    
    # Get results from database
    conn = sqlite3.connect('logistics.db')
    
    run_data = pd.read_sql_query('''
        SELECT * FROM optimization_runs WHERE id = ?
    ''', conn, params=(run_id,))
    
    routes_data = pd.read_sql_query('''
        SELECT * FROM solution_routes WHERE run_id = ?
    ''', conn, params=(run_id,))
    
    metro_data = pd.read_sql_query('''
        SELECT * FROM metro_schedules WHERE run_id = ?
    ''', conn, params=(run_id,))
    
    conn.close()
    
    # Export to Excel
    exporter = ExcelExporter()
    filename = f"optimization_results_run_{run_id}.xlsx"
    filepath = os.path.join('output', filename)
    
    exporter.export_all_results(
        run_data.to_dict('records')[0] if not run_data.empty else {},
        routes_data.to_dict('records'),
        metro_data.to_dict('records'),
        filepath
    )
    
    return send_file(filepath, as_attachment=True)

@app.route('/history')
def get_history():
    """Get optimization run history"""
    conn = sqlite3.connect('logistics.db')
    history = pd.read_sql_query('''
        SELECT id, timestamp, status, objective_value, runtime, iterations
        FROM optimization_runs
        ORDER BY timestamp DESC
    ''', conn)
    conn.close()
    
    return jsonify(history.to_dict('records'))

def create_sample_data():
    """Create sample data for demonstration"""
    nodes = [
        {'node_id': 0, 'latitude': 40.7128, 'longitude': -74.0060, 'node_name': 'Depot'},
        {'node_id': 1, 'latitude': 40.7614, 'longitude': -73.9776, 'node_name': 'Customer 1'},
        {'node_id': 2, 'latitude': 40.7505, 'longitude': -73.9934, 'node_name': 'Customer 2'},
        {'node_id': 3, 'latitude': 40.7282, 'longitude': -73.7949, 'node_name': 'Customer 3'},
        {'node_id': 4, 'latitude': 40.6892, 'longitude': -74.0445, 'node_name': 'Customer 4'},
        {'node_id': 5, 'latitude': 40.7831, 'longitude': -73.9712, 'node_name': 'Customer 5'}
    ]
    
    demands = [
        {'node_id': 1, 'demand': 15},
        {'node_id': 2, 'demand': 8},
        {'node_id': 3, 'demand': 22},
        {'node_id': 4, 'demand': 12},
        {'node_id': 5, 'demand': 6}
    ]
    
    vehicles = [
        {'vehicle_type': 'truck', 'capacity': 100, 'speed': 50, 'cost_per_km': 1.0},
        {'vehicle_type': 'drone', 'capacity': 10, 'speed': 80, 'cost_per_km': 0.5, 'energy_capacity': 100}
    ]
    
    metro_data = [
        {'direction': 'up', 'departure_time': 360, 'arrival_time': 420, 'from_station': 0, 'to_station': 5},
        {'direction': 'down', 'departure_time': 480, 'arrival_time': 540, 'from_station': 5, 'to_station': 0}
    ]
    
    params = {
        'max_iterations': 100,
        'tolerance': 1e-6,
        'time_limit': 3600,
        'metro_capacity': 200
    }
    
    return {
        'nodes': nodes,
        'demands': demands,
        'vehicles': vehicles,
        'metro_data': metro_data,
        'params': params
    }

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)