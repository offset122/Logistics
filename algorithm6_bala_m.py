from typing import List, Dict, Any
import numpy as np
from data_structures import Column, MetroBALabel, LabelSet

class MetroBALASolver:
    """Algorithm 6: Big-Arc Label-Setting Algorithm for Metro (BALA-M)"""
    
    def __init__(self, nodes, metro_data, params):
        self.nodes = nodes
        self.metro_data = metro_data
        self.params = params
        
        # Metro system parameters
        self.metro_capacity = params.get('metro_capacity', 200)
        self.cost_per_shipment = params.get('metro_cost_per_shipment', 10.0)
        self.transfer_time = params.get('metro_transfer_time', 5)
        
        # Build metro network with big-arcs
        self.metro_network = self._build_metro_network()
        self.big_arcs = self._build_big_arcs()
        self.station_mapping = self._build_station_mapping()
        
        # Process timetables
        self.up_timetable = []
        self.down_timetable = []
        self._process_timetables()
    
    def _build_metro_network(self):
        """Build metro network structure"""
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
    
    def _build_big_arcs(self):
        """Build big-arcs for optimized metro scheduling"""
        big_arcs = []
        
        # Create big-arcs that span multiple stations
        for i in range(len(self.metro_network)):
            for j in range(i + 2, min(i + 5, len(self.metro_network))):  # Skip adjacent stations
                
                # Calculate total distance and time for big-arc
                total_distance = 0.0
                total_time = 0.0
                intermediate_stations = []
                
                for k in range(i, j):
                    if k + 1 < len(self.metro_network):
                        from_coords = self.metro_network[k]['coordinates']
                        to_coords = self.metro_network[k + 1]['coordinates']
                        
                        # Calculate distance between consecutive stations
                        distance = np.sqrt((to_coords[0] - from_coords[0])**2 + 
                                         (to_coords[1] - from_coords[1])**2) * 111  # Rough km conversion
                        time = distance / 40 * 60  # Assume 40 km/h metro speed, convert to minutes
                        
                        total_distance += distance
                        total_time += time
                        intermediate_stations.append(k + 1)
                
                big_arc = {
                    'from_station': i,
                    'to_station': j,
                    'distance': total_distance,
                    'time': total_time,
                    'intermediate_stations': intermediate_stations,
                    'capacity': self.metro_capacity
                }
                big_arcs.append(big_arc)
        
        return big_arcs
    
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
        """Solve metro scheduling subproblem using big-arc approach"""
        columns = []
        
        # Extract dual values
        inventory_duals = self._extract_inventory_duals(duals)
        metro_run_duals = self._extract_metro_run_duals(duals)
        
        # Solve using big-arc labels for both directions
        up_columns = self._solve_big_arc_direction('up', inventory_duals, metro_run_duals)
        down_columns = self._solve_big_arc_direction('down', inventory_duals, metro_run_duals)
        
        columns.extend(up_columns)
        columns.extend(down_columns)
        
        return columns
    
    def _solve_big_arc_direction(self, direction: str, inventory_duals: Dict, 
                                metro_run_duals: Dict) -> List[Column]:
        """Solve using big-arc label-setting algorithm"""
        columns = []
        
        if not self.metro_network:
            return columns
        
        # Initialize label sets for big-arc endpoints
        label_sets = {}
        for arc in self.big_arcs:
            from_station = arc['from_station']
            to_station = arc['to_station']
            
            if from_station not in label_sets:
                label_sets[from_station] = LabelSet()
            if to_station not in label_sets:
                label_sets[to_station] = LabelSet()
        
        # Initialize at first station
        if self.big_arcs:
            initial_label = MetroBALabel(
                from_station=0,
                to_station=0,
                departure_time=0,
                arrival_time=0,
                load=0,
                cost=0.0,
                big_arc_path=[]
            )
            label_sets[0].add_label(initial_label)
        
        # Process big-arcs in time-ordered fashion
        time_ordered_arcs = self._get_time_ordered_big_arcs(direction)
        
        for time_slot, available_arcs in time_ordered_arcs.items():
            for arc in available_arcs:
                from_station = arc['from_station']
                to_station = arc['to_station']
                
                if from_station in label_sets:
                    current_labels = label_sets[from_station].get_best_labels(10)
                    
                    for label in current_labels:
                        # Check if big-arc is feasible from current label
                        if self._is_big_arc_feasible(label, arc, time_slot):
                            new_label = self._extend_big_arc_label(
                                label, arc, time_slot, inventory_duals, metro_run_duals
                            )
                            
                            if to_station not in label_sets:
                                label_sets[to_station] = LabelSet()
                            
                            label_sets[to_station].add_label(new_label)
        
        # Generate columns from final labels
        for station_id, label_set in label_sets.items():
            for label in label_set.labels:
                if len(label.big_arc_path) > 0:
                    
                    # Calculate total reduced cost
                    total_reduced_cost = label.cost
                    
                    # Only add columns with negative reduced cost
                    if total_reduced_cost < -1e-6:
                        schedule_details = self._calculate_big_arc_schedule_details(
                            label, direction
                        )
                        
                        column = Column(
                            type='metro',
                            id=f"metro_ba_{direction}_{len(columns)}",
                            direct_cost=label.cost,
                            reduced_cost=total_reduced_cost,
                            details=schedule_details
                        )
                        columns.append(column)
        
        return columns
    
    def _get_time_ordered_big_arcs(self, direction: str) -> Dict[int, List[Dict]]:
        """Get big-arcs ordered by time slots"""
        time_ordered = {}
        
        # Create time slots based on timetable
        timetable = self.up_timetable if direction == 'up' else self.down_timetable
        
        for i, schedule_entry in enumerate(timetable):
            time_slot = schedule_entry['departure_time'] // 15  # 15-minute time slots
            
            if time_slot not in time_ordered:
                time_ordered[time_slot] = []
            
            # Find relevant big-arcs for this time slot
            for arc in self.big_arcs:
                if (arc['from_station'] == schedule_entry['from_station'] or
                    arc['to_station'] == schedule_entry['to_station']):
                    
                    # Add timing information to arc
                    arc_with_time = arc.copy()
                    arc_with_time['departure_time'] = schedule_entry['departure_time']
                    arc_with_time['arrival_time'] = schedule_entry['departure_time'] + arc['time']
                    arc_with_time['run_id'] = schedule_entry.get('run_id', 'default')
                    
                    time_ordered[time_slot].append(arc_with_time)
        
        return time_ordered
    
    def _is_big_arc_feasible(self, label: MetroBALabel, arc: Dict, time_slot: int) -> bool:
        """Check if big-arc extension is feasible"""
        # Check timing constraints
        if label.arrival_time > arc.get('departure_time', 0):
            return False
        
        # Check capacity constraints
        if label.load > arc.get('capacity', self.metro_capacity):
            return False
        
        # Check for cycles in big-arc path
        for prev_arc in label.big_arc_path:
            if (prev_arc.get('from_station') == arc['from_station'] and 
                prev_arc.get('to_station') == arc['to_station']):
                return False
        
        return True
    
    def _extend_big_arc_label(self, label: MetroBALabel, arc: Dict, time_slot: int,
                            inventory_duals: Dict, metro_run_duals: Dict) -> MetroBALabel:
        """Extend label using big-arc"""
        # Calculate cargo operations along the big-arc
        cargo_revenue = 0.0
        total_load_change = 0
        
        # Revenue from intermediate stations
        for intermediate_station in arc.get('intermediate_stations', []):
            station_key = f"up_station_{intermediate_station}" if 'up' in str(time_slot) else f"down_station_{intermediate_station}"
            cargo_revenue += inventory_duals.get(station_key, 0.0)
            total_load_change += min(arc['capacity'] - label.load, 50)  # Simplified load calculation
        
        # Transport cost for big-arc
        transport_cost = self.cost_per_shipment * (len(arc.get('intermediate_stations', [])) + 1)
        
        # Metro run cost
        run_cost = metro_run_duals.get(f"metro_run_{arc.get('run_id', 'default')}", 0.0)
        
        # Calculate reduced cost
        reduced_cost = transport_cost + run_cost - cargo_revenue
        
        # Create new big-arc path entry
        new_arc_entry = {
            'from_station': arc['from_station'],
            'to_station': arc['to_station'],
            'departure_time': arc.get('departure_time', 0),
            'arrival_time': arc.get('arrival_time', 0),
            'load_change': total_load_change,
            'intermediate_stations': arc.get('intermediate_stations', []),
            'distance': arc.get('distance', 0.0)
        }
        
        return MetroBALabel(
            from_station=label.to_station,
            to_station=arc['to_station'],
            departure_time=arc.get('departure_time', 0),
            arrival_time=arc.get('arrival_time', 0),
            load=label.load + total_load_change,
            cost=label.cost + reduced_cost,
            big_arc_path=label.big_arc_path + [new_arc_entry],
            parent=label
        )
    
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
    
    def _calculate_big_arc_schedule_details(self, label: MetroBALabel, direction: str) -> Dict[str, Any]:
        """Calculate detailed schedule information for big-arc solution"""
        total_load = 0
        total_distance = 0.0
        stops = []
        
        for arc_entry in label.big_arc_path:
            from_station = arc_entry['from_station']
            to_station = arc_entry['to_station']
            departure_time = arc_entry['departure_time']
            arrival_time = arc_entry['arrival_time']
            load_change = arc_entry['load_change']
            distance = arc_entry.get('distance', 0.0)
            
            total_load += load_change
            total_distance += distance
            
            # Add stops for intermediate stations
            for intermediate_station in arc_entry.get('intermediate_stations', []):
                stops.append({
                    'station': intermediate_station,
                    'time': departure_time + (arrival_time - departure_time) * 0.5,  # Simplified
                    'load': load_change // len(arc_entry.get('intermediate_stations', [1])),
                    'type': 'intermediate'
                })
            
            # Add final stop
            stops.append({
                'station': to_station,
                'time': arrival_time,
                'load': load_change,
                'type': 'final'
            })
        
        return {
            'direction': direction,
            'big_arc_path': label.big_arc_path,
            'stops': stops,
            'total_load': total_load,
            'total_distance': total_distance,
            'start_time': label.big_arc_path[0]['departure_time'] if label.big_arc_path else 0,
            'end_time': label.big_arc_path[-1]['arrival_time'] if label.big_arc_path else 0,
            'num_big_arcs': len(label.big_arc_path),
            'optimization_type': 'big_arc'
        }