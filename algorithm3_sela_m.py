from typing import List, Dict, Any
import numpy as np
from data_structures import Column, MetroLabel, LabelSet

class MetroSELASolver:
    """Algorithm 3: Stepwise Exact Label-Setting Algorithm for Metro (SELA-M)"""
    
    def __init__(self, nodes, metro_data, params):
        self.nodes = nodes
        self.metro_data = metro_data
        self.params = params
        
        # Metro system parameters
        self.metro_capacity = params.get('metro_capacity', 200)
        self.cost_per_shipment = params.get('metro_cost_per_shipment', 10.0)
        self.transfer_time = params.get('metro_transfer_time', 5)  # minutes
        
        # Build metro network
        self.metro_network = self._build_metro_network()
        self.station_mapping = self._build_station_mapping()
        
        # Process metro timetables
        self.up_timetable = []
        self.down_timetable = []
        self._process_timetables()
    
    def _build_metro_network(self):
        """Build metro network structure"""
        # Simplified metro network - assumes linear line with stations
        stations = []
        for i, node in enumerate(self.nodes):
            if node.get('is_metro_station', False):
                stations.append({
                    'station_id': i,
                    'node_id': node['node_id'],
                    'name': node.get('node_name', f'Station_{i}'),
                    'coordinates': (node['latitude'], node['longitude'])
                })
        
        return stations
    
    def _build_station_mapping(self):
        """Build mapping between node IDs and station indices"""
        mapping = {}
        for i, station in enumerate(self.metro_network):
            mapping[station['node_id']] = i
        return mapping
    
    def _process_timetables(self):
        """Process raw metro data into structured timetables"""
        for entry in self.metro_data:
            direction = entry.get('direction', 'up').lower()
            schedule_entry = {
                'departure_time': entry.get('departure_time', 0),
                'arrival_time': entry.get('arrival_time', 0),
                'from_station': entry.get('from_station', 0),
                'to_station': entry.get('to_station', 1),
                'capacity': entry.get('capacity', self.metro_capacity),
                'run_id': entry.get('run_id', 'default')
            }
            
            if 'up' in direction:
                self.up_timetable.append(schedule_entry)
            else:
                self.down_timetable.append(schedule_entry)
        
        # Sort timetables by departure time
        self.up_timetable.sort(key=lambda x: x['departure_time'])
        self.down_timetable.sort(key=lambda x: x['departure_time'])
    
    def solve(self, duals: Dict[str, float]) -> List[Column]:
        """Solve metro scheduling subproblem using SELA-M"""
        columns = []
        
        # Extract dual values for inventory balance (psi_up, psi_down)
        inventory_duals = self._extract_inventory_duals(duals)
        
        # Extract dual values for metro run constraints
        metro_run_duals = self._extract_metro_run_duals(duals)
        
        # Solve for up-line schedules
        up_columns = self._solve_direction('up', self.up_timetable, 
                                         inventory_duals, metro_run_duals)
        columns.extend(up_columns)
        
        # Solve for down-line schedules
        down_columns = self._solve_direction('down', self.down_timetable, 
                                           inventory_duals, metro_run_duals)
        columns.extend(down_columns)
        
        return columns
    
    def _solve_direction(self, direction: str, timetable: List[Dict], 
                        inventory_duals: Dict, metro_run_duals: Dict) -> List[Column]:
        """Solve metro scheduling for one direction using label-setting"""
        columns = []
        
        if not timetable:
            return columns
        
        # Initialize label sets for each station
        num_stations = len(self.metro_network)
        label_sets = {i: LabelSet() for i in range(num_stations)}
        
        # Initialize at first station
        initial_label = MetroLabel(
            station=0,
            time=0,
            load=0,
            cost=0.0,
            schedule_path=[]
        )
        label_sets[0].add_label(initial_label)
        
        # Process each time step in the timetable
        for schedule_entry in timetable:
            departure_time = schedule_entry['departure_time']
            arrival_time = schedule_entry['arrival_time']
            from_station = schedule_entry['from_station']
            to_station = schedule_entry['to_station']
            capacity = schedule_entry['capacity']
            run_id = schedule_entry['run_id']
            
            # Check if we have labels at the departure station
            if from_station in label_sets:
                current_labels = label_sets[from_station].get_best_labels()
                
                for label in current_labels:
                    # Check if train departure is feasible from current label time
                    if label.time <= departure_time:
                        # Calculate cargo operations
                        cargo_revenue = self._calculate_cargo_revenue(
                            from_station, to_station, direction, inventory_duals
                        )
                        
                        # Calculate transport cost
                        transport_cost = self.cost_per_shipment
                        
                        # Add metro run cost
                        run_cost = metro_run_duals.get(f'metro_run_{run_id}', 0.0)
                        
                        # Calculate load change (simplified)
                        load_change = min(capacity, 
                                        inventory_duals.get(f'{direction}_demand_{from_station}', 0))
                        
                        # Calculate reduced cost
                        reduced_cost = transport_cost + run_cost - cargo_revenue
                        
                        # Create new label
                        new_label = label.extend_to(
                            to_station, departure_time, arrival_time, 
                            reduced_cost, load_change
                        )
                        
                        # Add to destination station
                        if to_station not in label_sets:
                            label_sets[to_station] = LabelSet()
                        
                        label_sets[to_station].add_label(new_label)
        
        # Generate columns from final labels
        for station_id, label_set in label_sets.items():
            for label in label_set.labels:
                if len(label.schedule_path) > 0:  # Only non-empty schedules
                    
                    # Calculate total reduced cost
                    total_reduced_cost = label.cost
                    
                    # Only add columns with negative reduced cost
                    if total_reduced_cost < -1e-6:
                        schedule_details = self._calculate_schedule_details(
                            label, direction
                        )
                        
                        column = Column(
                            type='metro',
                            id=f"metro_{direction}_{len(columns)}",
                            direct_cost=label.cost,
                            reduced_cost=total_reduced_cost,
                            details=schedule_details
                        )
                        columns.append(column)
        
        return columns
    
    def _extract_inventory_duals(self, duals: Dict[str, float]) -> Dict[str, float]:
        """Extract dual values for inventory balance constraints"""
        inventory_duals = {}
        
        for key, value in duals.items():
            if key.startswith('inventory_balance_'):
                station_id = key.split('_')[-1]
                inventory_duals[f'up_station_{station_id}'] = value
                inventory_duals[f'down_station_{station_id}'] = value
        
        return inventory_duals
    
    def _extract_metro_run_duals(self, duals: Dict[str, float]) -> Dict[str, float]:
        """Extract dual values for metro run constraints"""
        metro_duals = {}
        
        for key, value in duals.items():
            if key.startswith('metro_run_'):
                metro_duals[key] = value
        
        return metro_duals
    
    def _calculate_cargo_revenue(self, from_station: int, to_station: int, 
                               direction: str, inventory_duals: Dict) -> float:
        """Calculate revenue from cargo operations"""
        # Simplified cargo revenue calculation
        from_key = f'{direction}_station_{from_station}'
        to_key = f'{direction}_station_{to_station}'
        
        pickup_revenue = inventory_duals.get(from_key, 0.0)
        delivery_revenue = inventory_duals.get(to_key, 0.0)
        
        return pickup_revenue + delivery_revenue
    
    def _calculate_schedule_details(self, label: MetroLabel, direction: str) -> Dict[str, Any]:
        """Calculate detailed schedule information"""
        total_load = 0
        total_distance = 0.0
        stops = []
        
        for i, schedule_step in enumerate(label.schedule_path):
            from_station = schedule_step['from_station']
            to_station = schedule_step['to_station']
            departure_time = schedule_step['departure_time']
            arrival_time = schedule_step['arrival_time']
            load = schedule_step['load']
            
            # Calculate distance between stations
            if (from_station < len(self.metro_network) and 
                to_station < len(self.metro_network)):
                
                from_coords = self.metro_network[from_station]['coordinates']
                to_coords = self.metro_network[to_station]['coordinates']
                
                # Simple distance calculation
                distance = np.sqrt((to_coords[0] - from_coords[0])**2 + 
                                 (to_coords[1] - from_coords[1])**2) * 111  # Rough km conversion
                total_distance += distance
            
            total_load += load
            
            stops.append({
                'from_station': from_station,
                'to_station': to_station,
                'departure_time': departure_time,
                'arrival_time': arrival_time,
                'load': load,
                'distance': distance if 'distance' in locals() else 0.0
            })
        
        return {
            'direction': direction,
            'schedule_path': label.schedule_path,
            'stops': stops,
            'total_load': total_load,
            'total_distance': total_distance,
            'start_time': label.schedule_path[0]['departure_time'] if label.schedule_path else 0,
            'end_time': label.schedule_path[-1]['arrival_time'] if label.schedule_path else 0,
            'num_stops': len(stops)
        }