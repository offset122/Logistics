import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataLoader:
    """Handles loading and normalizing data from Excel files"""
    
    def __init__(self):
        self.header_mappings = {
            # Chinese to English header mappings
            '节点ID': 'node_id',
            '节点名称': 'node_name',
            '经度': 'longitude',
            '纬度': 'latitude',
            '需求量': 'demand',
            '时间窗开始': 'time_window_start',
            '时间窗结束': 'time_window_end',
            '车辆类型': 'vehicle_type',
            '容量': 'capacity',
            '速度': 'speed',
            '成本': 'cost',
            '出发时间': 'departure_time',
            '到达时间': 'arrival_time',
            '方向': 'direction',
            '载重': 'load',
            # Add more mappings as needed
        }
    
    def normalize_headers(self, df):
        """Normalize column headers from Chinese to English"""
        df_copy = df.copy()
        df_copy.columns = [self.header_mappings.get(col, col.lower().replace(' ', '_')) for col in df_copy.columns]
        return df_copy
    
    def parse_time_to_minutes(self, time_str):
        """Convert time string to minutes from 00:00"""
        if pd.isna(time_str):
            return 0
        
        if isinstance(time_str, str):
            try:
                # Handle various time formats
                if ':' in time_str:
                    parts = time_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes
                else:
                    # Assume it's just hours
                    return int(time_str) * 60
            except:
                return 0
        elif isinstance(time_str, (int, float)):
            # Assume it's already in minutes or hours
            return int(time_str) if time_str < 1440 else int(time_str / 60)
        
        return 0
    
    def load_nodes_data(self, filepath):
        """Load and process nodes data"""
        df = pd.read_excel(filepath)
        df = self.normalize_headers(df)
        
        # Ensure required columns exist
        required_cols = ['node_id', 'latitude', 'longitude']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Process time windows if they exist
        if 'time_window_start' in df.columns:
            df['time_window_start'] = df['time_window_start'].apply(self.parse_time_to_minutes)
        if 'time_window_end' in df.columns:
            df['time_window_end'] = df['time_window_end'].apply(self.parse_time_to_minutes)
        
        return df.to_dict('records')
    
    def load_demands_data(self, filepath):
        """Load and process demand data"""
        df = pd.read_excel(filepath)
        df = self.normalize_headers(df)
        
        required_cols = ['node_id', 'demand']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df.to_dict('records')
    
    def load_vehicles_data(self, filepath):
        """Load and process vehicle data"""
        df = pd.read_excel(filepath)
        df = self.normalize_headers(df)
        
        required_cols = ['vehicle_type', 'capacity', 'speed']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df.to_dict('records')
    
    def load_metro_data(self, filepath):
        """Load and process metro timetable data"""
        df = pd.read_excel(filepath)
        df = self.normalize_headers(df)
        
        # Convert time columns to minutes
        time_columns = ['departure_time', 'arrival_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.parse_time_to_minutes)
        
        return df.to_dict('records')
    
    def load_params_data(self, filepath):
        """Load and process parameters data"""
        df = pd.read_excel(filepath)
        df = self.normalize_headers(df)
        
        # Convert to key-value pairs
        params = {}
        for _, row in df.iterrows():
            if 'parameter' in df.columns and 'value' in df.columns:
                params[row['parameter']] = row['value']
            elif len(df.columns) >= 2:
                params[row.iloc[0]] = row.iloc[1]
        
        return params
    
    def build_distance_matrix(self, nodes):
        """Build distance matrix from node coordinates"""
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = nodes[i]['latitude'], nodes[i]['longitude']
                    lat2, lon2 = nodes[j]['latitude'], nodes[j]['longitude']
                    
                    # Haversine distance formula
                    R = 6371  # Earth's radius in km
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    
                    a = (np.sin(dlat/2)**2 + 
                         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                         np.sin(dlon/2)**2)
                    
                    distance = 2 * R * np.arcsin(np.sqrt(a))
                    distance_matrix[i][j] = distance
        
        return distance_matrix
    
    def build_time_matrix(self, nodes, vehicles, distance_matrix):
        """Build time matrix for different vehicle types"""
        time_matrices = {}
        
        for vehicle in vehicles:
            vehicle_type = vehicle['vehicle_type']
            speed = vehicle['speed']  # km/h
            
            # Convert distance to time (in minutes)
            time_matrix = distance_matrix * 60 / speed
            time_matrices[vehicle_type] = time_matrix
        
        return time_matrices
    
    def build_metro_lookup(self, metro_data):
        """Build metro timetable lookup structures"""
        metro_lookup = {
            'up_line': [],
            'down_line': []
        }
        
        for entry in metro_data:
            direction = entry.get('direction', 'up').lower()
            if 'up' in direction:
                metro_lookup['up_line'].append(entry)
            else:
                metro_lookup['down_line'].append(entry)
        
        # Sort by departure time
        metro_lookup['up_line'].sort(key=lambda x: x.get('departure_time', 0))
        metro_lookup['down_line'].sort(key=lambda x: x.get('departure_time', 0))
        
        return metro_lookup
    
    def load_all_data(self, data_dir):
        """Load all data files and build necessary structures"""
        # Find the most recent files for each type
        file_types = ['nodes', 'demands', 'vehicles', 'metro', 'params']
        filepaths = {}
        
        for file_type in file_types:
            matching_files = [f for f in os.listdir(data_dir) if f.startswith(file_type)]
            if matching_files:
                # Get the most recent file
                matching_files.sort(reverse=True)
                filepaths[file_type] = os.path.join(data_dir, matching_files[0])
            else:
                raise FileNotFoundError(f"No {file_type} file found in {data_dir}")
        
        # Load data
        nodes = self.load_nodes_data(filepaths['nodes'])
        demands = self.load_demands_data(filepaths['demands'])
        vehicles = self.load_vehicles_data(filepaths['vehicles'])
        metro_data = self.load_metro_data(filepaths['metro'])
        params = self.load_params_data(filepaths['params'])
        
        # Build matrices and lookups
        distance_matrix = self.build_distance_matrix(nodes)
        time_matrices = self.build_time_matrix(nodes, vehicles, distance_matrix)
        metro_lookup = self.build_metro_lookup(metro_data)
        
        return {
            'nodes': nodes,
            'demands': demands,
            'vehicles': vehicles,
            'metro_data': metro_data,
            'params': params,
            'distance_matrix': distance_matrix,
            'time_matrices': time_matrices,
            'metro_lookup': metro_lookup
        }