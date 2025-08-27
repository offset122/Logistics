from typing import List, Dict, Any
import numpy as np
from data_structures import Column, DroneLabel, LabelSet

class DroneBLASolver:
    """Algorithm 5: Bidirectional Label-Setting Algorithm for Drone (BLA-D)"""
    
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
        self.energy_capacity = self.drone_params.get('energy_capacity', 100)
        self.energy_per_km = self.drone_params.get('energy_per_km', 2.0)
        self.cost_per_km = self.drone_params.get('cost_per_km', 0.5)
        
        # Build distance and time matrices
        self.distance_matrix = self._build_distance_matrix()
        self.time_matrix = self.distance_matrix * 60 / self.speed
        
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
                    
                    # Haversine distance
                    R = 6371
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    
                    a = (np.sin(dlat/2)**2 + 
                         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                         np.sin(dlon/2)**2)
                    
                    distance = 2 * R * np.arcsin(np.sqrt(a))
                    distance_matrix[i][j] = distance
        
        return distance_matrix
    
    def solve(self, duals: Dict[str, float]) -> List[Column]:
        """Solve drone routing subproblem using BLA-D (bidirectional)"""
        columns = []
        
        # Extract dual values
        demand_duals = {}
        for key, value in duals.items():
            if key.startswith('demand_'):
                node_id = int(key.split('_')[1])
                demand_duals[node_id] = value
        
        resource_duals = self._extract_resource_duals(duals)
        
        # Run bidirectional search
        forward_labels, backward_labels = self._bidirectional_drone_search(demand_duals, resource_duals)
        
        # Join forward and backward labels
        complete_routes = self._join_drone_labels(forward_labels, backward_labels, demand_duals)
        
        # Convert routes to columns
        for route in complete_routes:
            if route['reduced_cost'] < -1e-6:
                column = Column(
                    type='drone',
                    id=f"drone_bla_route_{len(columns)}",
                    direct_cost=route['cost'],
                    reduced_cost=route['reduced_cost'],
                    details=route['details']
                )
                columns.append(column)
        
        return columns
    
    def _bidirectional_drone_search(self, demand_duals: Dict, resource_duals: Dict):
        """Perform bidirectional search for drones"""
        n = len(self.nodes)
        
        # Forward search from depot
        forward_label_sets = {i: LabelSet() for i in range(n)}
        forward_label_sets[0].add_label(DroneLabel(
            node=0, time=0, load=self.capacity, energy=self.energy_capacity, cost=0.0, path=[0]
        ))
        
        # Backward search to depot
        backward_label_sets = {i: LabelSet() for i in range(n)}
        backward_label_sets[0].add_label(DroneLabel(
            node=0, time=0, load=0, energy=self.energy_capacity, cost=0.0, path=[0]
        ))
        
        # Forward expansion
        for iteration in range(30):
            changed = False
            for current_node in range(n):
                for label in forward_label_sets[current_node].get_best_labels(5):
                    for next_node in range(n):
                        if next_node == current_node or next_node in label.path:
                            continue
                        
                        if self._is_drone_forward_feasible(label, current_node, next_node):
                            new_label = self._extend_drone_forward_label(
                                label, current_node, next_node, demand_duals, resource_duals
                            )
                            if forward_label_sets[next_node].add_label(new_label):
                                changed = True
            
            if not changed:
                break
        
        # Backward expansion
        for iteration in range(30):
            changed = False
            for current_node in range(n):
                for label in backward_label_sets[current_node].get_best_labels(5):
                    for prev_node in range(n):
                        if prev_node == current_node or prev_node in label.path:
                            continue
                        
                        if self._is_drone_backward_feasible(label, current_node, prev_node):
                            new_label = self._extend_drone_backward_label(
                                label, current_node, prev_node, demand_duals, resource_duals
                            )
                            if backward_label_sets[prev_node].add_label(new_label):
                                changed = True
            
            if not changed:
                break
        
        return forward_label_sets, backward_label_sets
    
    def _is_drone_forward_feasible(self, label: DroneLabel, current_node: int, next_node: int) -> bool:
        """Check drone forward feasibility"""
        # Energy constraint
        travel_distance = self.distance_matrix[current_node][next_node]
        energy_consumption = travel_distance * self.energy_per_km
        if label.energy < energy_consumption:
            return False
        
        # Flight time constraint
        travel_time = self.time_matrix[current_node][next_node]
        if label.time + travel_time > self.max_flight_time:
            return False
        
        # Capacity constraint for delivery
        if next_node in self.demand_lookup:
            demand_qty = self.demand_lookup[next_node]
            if label.load < demand_qty:
                return False
        
        return True
    
    def _is_drone_backward_feasible(self, label: DroneLabel, current_node: int, prev_node: int) -> bool:
        """Check drone backward feasibility"""
        # Energy constraint
        travel_distance = self.distance_matrix[prev_node][current_node]
        energy_consumption = travel_distance * self.energy_per_km
        if label.energy < energy_consumption:
            return False
        
        # Capacity constraint for pickup
        if current_node in self.demand_lookup:
            demand_qty = self.demand_lookup[current_node]
            if label.load + demand_qty > self.capacity:
                return False
        
        return True
    
    def _extend_drone_forward_label(self, label: DroneLabel, current_node: int, 
                                  next_node: int, demand_duals: Dict, resource_duals: Dict) -> DroneLabel:
        """Extend drone label in forward direction"""
        travel_distance = self.distance_matrix[current_node][next_node]
        travel_time = self.time_matrix[current_node][next_node]
        energy_consumption = travel_distance * self.energy_per_km
        travel_cost = travel_distance * self.cost_per_km
        
        # Calculate delivery
        pickup_load = 0
        delivery_revenue = 0
        if next_node in self.demand_lookup:
            demand_qty = self.demand_lookup[next_node]
            pickup_load = -demand_qty
            delivery_revenue = demand_duals.get(next_node, 0) * demand_qty
        
        # Resource cost
        arrival_time = label.time + travel_time
        resource_cost = self._calculate_drone_resource_cost(
            label.time, arrival_time, resource_duals
        )
        
        reduced_cost = travel_cost + resource_cost - delivery_revenue
        
        return label.extend_to(next_node, travel_time, energy_consumption, reduced_cost, -pickup_load)
    
    def _extend_drone_backward_label(self, label: DroneLabel, current_node: int, 
                                   prev_node: int, demand_duals: Dict, resource_duals: Dict) -> DroneLabel:
        """Extend drone label in backward direction"""
        travel_distance = self.distance_matrix[prev_node][current_node]
        travel_time = self.time_matrix[prev_node][current_node]
        energy_consumption = travel_distance * self.energy_per_km
        travel_cost = travel_distance * self.cost_per_km
        
        # In backward search, we pick up demand at current node
        pickup_load = 0
        pickup_revenue = 0
        if current_node in self.demand_lookup:
            demand_qty = self.demand_lookup[current_node]
            pickup_load = demand_qty
            pickup_revenue = demand_duals.get(current_node, 0) * demand_qty
        
        reduced_cost = travel_cost - pickup_revenue
        
        return DroneLabel(
            node=prev_node,
            time=label.time + travel_time,
            load=label.load + pickup_load,
            energy=label.energy - energy_consumption,
            cost=label.cost + reduced_cost,
            path=[prev_node] + label.path
        )
    
    def _join_drone_labels(self, forward_labels: Dict, backward_labels: Dict, 
                          demand_duals: Dict) -> List[Dict]:
        """Join forward and backward drone labels"""
        complete_routes = []
        
        for join_node in range(1, len(self.nodes)):
            for forward_label in forward_labels[join_node].get_best_labels(3):
                for backward_label in backward_labels[join_node].get_best_labels(3):
                    
                    if self._are_drone_labels_compatible(forward_label, backward_label):
                        route = self._create_complete_drone_route(forward_label, backward_label, join_node)
                        if route:
                            complete_routes.append(route)
        
        return complete_routes
    
    def _are_drone_labels_compatible(self, forward_label: DroneLabel, backward_label: DroneLabel) -> bool:
        """Check if drone labels can be joined"""
        # Check load compatibility
        if forward_label.load + backward_label.load > self.capacity:
            return False
        
        # Check energy compatibility
        if forward_label.energy + backward_label.energy > self.energy_capacity * 1.5:  # Some buffer
            return False
        
        # Check path intersection
        forward_path_set = set(forward_label.path)
        backward_path_set = set(backward_label.path)
        common_nodes = forward_path_set.intersection(backward_path_set)
        
        return len(common_nodes) <= 1  # Only depot should be common
    
    def _create_complete_drone_route(self, forward_label: DroneLabel, backward_label: DroneLabel, 
                                   join_node: int) -> Dict:
        """Create complete drone route from labels"""
        # Construct complete path
        forward_path = forward_label.path
        backward_path = backward_label.path[1:]
        backward_path.reverse()
        
        complete_path = forward_path + backward_path
        
        # Calculate totals
        total_cost = forward_label.cost + backward_label.cost
        total_time = forward_label.time + backward_label.time
        total_energy = (self.energy_capacity - forward_label.energy) + (self.energy_capacity - backward_label.energy)
        total_load = sum(self.demand_lookup.get(node, 0) for node in complete_path if node != 0)
        
        return {
            'path': complete_path,
            'cost': total_cost,
            'reduced_cost': total_cost,
            'details': {
                'route': complete_path,
                'total_time': total_time,
                'total_distance': self._calculate_path_distance(complete_path),
                'total_energy': total_energy,
                'load': total_load,
                'join_node': join_node
            }
        }
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calculate total distance for a path"""
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i]][path[i + 1]]
        return total_distance
    
    def _extract_resource_duals(self, duals: Dict[str, float]) -> Dict[str, List[float]]:
        """Extract dual values for drone resource constraints"""
        resource_duals = {'drone': [], 'pilot': []}
        
        for t in range(96):
            drone_key = f"drone_cap_{t}"
            pilot_key = f"pilot_cap_{t}"
            
            resource_duals['drone'].append(duals.get(drone_key, 0.0))
            resource_duals['pilot'].append(duals.get(pilot_key, 0.0))
        
        return resource_duals
    
    def _calculate_drone_resource_cost(self, start_time: int, end_time: int, 
                                     resource_duals: Dict[str, List[float]]) -> float:
        """Calculate drone resource cost"""
        cost = 0.0
        start_slice = start_time // 15
        end_slice = end_time // 15
        
        for t in range(start_slice, min(end_slice + 1, len(resource_duals['drone']))):
            cost += resource_duals['drone'][t] + resource_duals['pilot'][t]
        
        return cost