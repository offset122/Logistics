from typing import List, Dict, Any
import numpy as np
from data_structures import Column, DroneLabel, LabelSet

class DroneULASolver:
    """Algorithm 2: Unidirectional Label-Setting Algorithm for Drone (ULA-D)"""
    
    def __init__(self, nodes, demands, vehicles, params):
        self.nodes = nodes
        self.demands = demands
        self.vehicles = vehicles
        self.params = params
        
        # Get drone-specific parameters
        self.drone_params = next((v for v in vehicles if v['vehicle_type'] == 'drone'), {})
        self.capacity = self.drone_params.get('capacity', 10)
        self.speed = self.drone_params.get('speed', 80)  # km/h
        self.max_flight_time = self.drone_params.get('max_flight_time', 60)  # minutes
        self.energy_capacity = self.drone_params.get('energy_capacity', 100)  # battery units
        self.energy_per_km = self.drone_params.get('energy_per_km', 2.0)
        self.cost_per_km = self.drone_params.get('cost_per_km', 0.5)
        
        # Build distance and time matrices
        self.distance_matrix = self._build_distance_matrix()
        self.time_matrix = self.distance_matrix * 60 / self.speed  # Convert to minutes
        
        # Demand lookup
        self.demand_lookup = {d['node_id']: d['demand'] for d in demands}
    
    def _build_distance_matrix(self):
        """Build distance matrix between nodes (straight-line for drones)"""
        n = len(self.nodes)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = self.nodes[i]['latitude'], self.nodes[i]['longitude']
                    lat2, lon2 = self.nodes[j]['latitude'], self.nodes[j]['longitude']
                    
                    # Haversine distance (straight-line for drones)
                    R = 6371  # Earth radius in km
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    
                    a = (np.sin(dlat/2)**2 + 
                         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                         np.sin(dlon/2)**2)
                    
                    distance = 2 * R * np.arcsin(np.sqrt(a))
                    distance_matrix[i][j] = distance
        
        return distance_matrix
    
    def solve(self, duals: Dict[str, float]) -> List[Column]:
        """Solve drone routing subproblem using ULA-D"""
        columns = []
        
        # Extract dual values for demand constraints
        demand_duals = {}
        for key, value in duals.items():
            if key.startswith('demand_'):
                node_id = int(key.split('_')[1])
                demand_duals[node_id] = value
        
        # Extract dual values for resource constraints
        resource_duals = self._extract_resource_duals(duals)
        
        # Initialize label sets for each node
        label_sets = {i: LabelSet() for i in range(len(self.nodes))}
        
        # Initialize at depot (node 0)
        initial_label = DroneLabel(
            node=0,
            time=0,
            load=self.capacity,  # Start with full capacity
            energy=self.energy_capacity,  # Start with full battery
            cost=0.0,
            path=[0]
        )
        label_sets[0].add_label(initial_label)
        
        # Forward pass - extend labels
        changed = True
        iteration = 0
        max_iterations = 100
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for current_node in range(len(self.nodes)):
                current_labels = label_sets[current_node].get_best_labels()
                
                for label in current_labels:
                    # Try extending to all other nodes
                    for next_node in range(len(self.nodes)):
                        if next_node == current_node or next_node in label.path:
                            continue
                        
                        # Check feasibility
                        travel_distance = self.distance_matrix[current_node][next_node]
                        travel_time = self.time_matrix[current_node][next_node]
                        energy_consumption = travel_distance * self.energy_per_km
                        travel_cost = travel_distance * self.cost_per_km
                        
                        # Check energy constraint
                        if label.energy < energy_consumption:
                            continue
                        
                        # Check flight time constraint
                        arrival_time = label.time + travel_time
                        if arrival_time > self.max_flight_time:
                            continue
                        
                        # Check time window constraints
                        node_data = self.nodes[next_node]
                        if 'time_window_end' in node_data:
                            if arrival_time > node_data['time_window_end']:
                                continue
                        
                        # Calculate pickup/delivery
                        pickup_load = 0
                        delivery_revenue = 0
                        
                        if next_node in self.demand_lookup:
                            demand_qty = self.demand_lookup[next_node]
                            
                            # Check if we can deliver
                            if label.load >= demand_qty:
                                pickup_load = -demand_qty  # Negative because it's delivery
                                delivery_revenue = demand_duals.get(next_node, 0) * demand_qty
                        
                        # Check capacity constraint
                        new_load = label.load + pickup_load
                        if new_load < 0:  # Can't deliver more than we have
                            continue
                        
                        # Calculate reduced cost
                        reduced_cost = travel_cost - delivery_revenue
                        
                        # Add resource cost (drone and pilot usage)
                        resource_cost = self._calculate_resource_cost(
                            label.time, arrival_time, resource_duals
                        )
                        reduced_cost += resource_cost
                        
                        # Create new label
                        new_label = label.extend_to(
                            next_node, travel_time, energy_consumption, 
                            travel_cost + reduced_cost, -pickup_load
                        )
                        
                        # Add to label set with dominance check
                        if label_sets[next_node].add_label(new_label):
                            changed = True
        
        # Generate columns from labels that can return to depot
        for node_id in range(1, len(self.nodes)):  # Skip depot
            for label in label_sets[node_id].labels:
                # Check if can return to depot
                return_distance = self.distance_matrix[node_id][0]
                return_time = self.time_matrix[node_id][0]
                return_energy = return_distance * self.energy_per_km
                return_cost = return_distance * self.cost_per_km
                
                # Check feasibility of return
                if (label.energy >= return_energy and 
                    label.time + return_time <= self.max_flight_time):
                    
                    final_time = label.time + return_time
                    final_cost = label.cost + return_cost
                    final_path = label.path + [0]
                    
                    # Calculate total reduced cost
                    total_reduced_cost = final_cost
                    
                    # Only add columns with negative reduced cost
                    if total_reduced_cost < -1e-6:
                        # Calculate route details
                        route_details = self._calculate_route_details(final_path, label, final_time)
                        
                        column = Column(
                            type='drone',
                            id=f"drone_route_{len(columns)}",
                            direct_cost=final_cost,
                            reduced_cost=total_reduced_cost,
                            details=route_details
                        )
                        columns.append(column)
        
        return columns
    
    def _extract_resource_duals(self, duals: Dict[str, float]) -> Dict[str, List[float]]:
        """Extract dual values for resource constraints by time slice"""
        resource_duals = {
            'drone': [],
            'pilot': []
        }
        
        # Assume 96 time slices (15-minute intervals over 24 hours)
        for t in range(96):
            drone_key = f"drone_cap_{t}"
            pilot_key = f"pilot_cap_{t}"
            
            resource_duals['drone'].append(duals.get(drone_key, 0.0))
            resource_duals['pilot'].append(duals.get(pilot_key, 0.0))
        
        return resource_duals
    
    def _calculate_resource_cost(self, start_time: int, end_time: int, 
                               resource_duals: Dict[str, List[float]]) -> float:
        """Calculate resource cost for using drone and pilot in time window"""
        cost = 0.0
        
        # Convert times to time slice indices (15-minute resolution)
        start_slice = start_time // 15
        end_slice = end_time // 15
        
        for t in range(start_slice, min(end_slice + 1, len(resource_duals['drone']))):
            cost += resource_duals['drone'][t] + resource_duals['pilot'][t]
        
        return cost
    
    def _calculate_route_details(self, path: List[int], final_label: DroneLabel, 
                               final_time: float) -> Dict[str, Any]:
        """Calculate detailed route information"""
        total_distance = 0.0
        total_energy = 0.0
        deliveries = []
        current_time = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            segment_distance = self.distance_matrix[from_node][to_node]
            segment_time = self.time_matrix[from_node][to_node]
            segment_energy = segment_distance * self.energy_per_km
            
            total_distance += segment_distance
            total_energy += segment_energy
            current_time += segment_time
            
            # Record delivery if this node has demand
            if to_node in self.demand_lookup and to_node != 0:
                deliveries.append({
                    'node_id': to_node,
                    'demand': self.demand_lookup[to_node],
                    'arrival_time': current_time,
                    'energy_remaining': self.energy_capacity - total_energy
                })
        
        return {
            'route': path,
            'total_distance': total_distance,
            'total_energy': total_energy,
            'flight_time': final_time,
            'deliveries': deliveries,
            'start_time': 0,
            'end_time': final_time,
            'load': sum(d['demand'] for d in deliveries),
            'energy_consumption': total_energy
        }