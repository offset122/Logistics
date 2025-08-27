from typing import List, Dict, Any, Tuple
import numpy as np
from data_structures import Column, TruckLabel, LabelSet

class TruckBLASolver:
    """Algorithm 4: Bidirectional Label-Setting Algorithm for Truck (BLA-T)"""
    
    def __init__(self, nodes, demands, vehicles, params):
        self.nodes = nodes
        self.demands = demands
        self.vehicles = vehicles
        self.params = params
        
        # Get truck-specific parameters
        self.truck_params = next((v for v in vehicles if v['vehicle_type'] == 'truck'), {})
        self.capacity = self.truck_params.get('capacity', 100)
        self.speed = self.truck_params.get('speed', 50)  # km/h
        self.cost_per_km = self.truck_params.get('cost_per_km', 1.0)
        
        # Build distance and time matrices
        self.distance_matrix = self._build_distance_matrix()
        self.time_matrix = self.distance_matrix * 60 / self.speed  # Convert to minutes
        
        # Demand lookup
        self.demand_lookup = {d['node_id']: d['demand'] for d in demands}
    
    def _build_distance_matrix(self):
        """Build distance matrix between nodes"""
        n = len(self.nodes)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = self.nodes[i]['latitude'], self.nodes[i]['longitude']
                    lat2, lon2 = self.nodes[j]['latitude'], self.nodes[j]['longitude']
                    
                    # Haversine distance
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
        """Solve truck routing subproblem using BLA-T (bidirectional)"""
        columns = []
        
        # Extract dual values
        demand_duals = {}
        for key, value in duals.items():
            if key.startswith('demand_'):
                node_id = int(key.split('_')[1])
                demand_duals[node_id] = value
        
        resource_duals = self._extract_resource_duals(duals)
        
        # Run bidirectional search
        forward_labels, backward_labels = self._bidirectional_search(demand_duals, resource_duals)
        
        # Join forward and backward labels to create complete routes
        complete_routes = self._join_labels(forward_labels, backward_labels, demand_duals)
        
        # Convert routes to columns
        for route in complete_routes:
            if route['reduced_cost'] < -1e-6:
                column = Column(
                    type='truck',
                    id=f"truck_bla_route_{len(columns)}",
                    direct_cost=route['cost'],
                    reduced_cost=route['reduced_cost'],
                    details=route['details']
                )
                columns.append(column)
        
        return columns
    
    def _bidirectional_search(self, demand_duals: Dict, resource_duals: Dict):
        """Perform bidirectional label-setting search"""
        n = len(self.nodes)
        
        # Forward search from depot (node 0)
        forward_label_sets = {i: LabelSet() for i in range(n)}
        forward_label_sets[0].add_label(TruckLabel(
            node=0, time=0, load=self.capacity, cost=0.0, path=[0]
        ))
        
        # Backward search to depot (node 0)
        backward_label_sets = {i: LabelSet() for i in range(n)}
        backward_label_sets[0].add_label(TruckLabel(
            node=0, time=0, load=0, cost=0.0, path=[0]  # Backward: start with empty load
        ))
        
        # Forward expansion
        for iteration in range(50):  # Limit iterations
            changed = False
            for current_node in range(n):
                for label in forward_label_sets[current_node].get_best_labels():
                    for next_node in range(n):
                        if next_node == current_node or next_node in label.path:
                            continue
                        
                        # Check forward feasibility
                        if self._is_forward_feasible(label, current_node, next_node, demand_duals):
                            new_label = self._extend_forward_label(
                                label, current_node, next_node, demand_duals, resource_duals
                            )
                            if forward_label_sets[next_node].add_label(new_label):
                                changed = True
            
            if not changed:
                break
        
        # Backward expansion
        for iteration in range(50):  # Limit iterations
            changed = False
            for current_node in range(n):
                for label in backward_label_sets[current_node].get_best_labels():
                    for prev_node in range(n):
                        if prev_node == current_node or prev_node in label.path:
                            continue
                        
                        # Check backward feasibility
                        if self._is_backward_feasible(label, current_node, prev_node, demand_duals):
                            new_label = self._extend_backward_label(
                                label, current_node, prev_node, demand_duals, resource_duals
                            )
                            if backward_label_sets[prev_node].add_label(new_label):
                                changed = True
            
            if not changed:
                break
        
        return forward_label_sets, backward_label_sets
    
    def _join_labels(self, forward_labels: Dict, backward_labels: Dict, 
                    demand_duals: Dict) -> List[Dict]:
        """Join forward and backward labels to create complete routes"""
        complete_routes = []
        
        # Try joining labels at each intermediate node
        for join_node in range(1, len(self.nodes)):  # Skip depot
            for forward_label in forward_labels[join_node].get_best_labels(10):  # Top 10
                for backward_label in backward_labels[join_node].get_best_labels(10):  # Top 10
                    
                    # Check compatibility
                    if self._are_labels_compatible(forward_label, backward_label):
                        # Create complete route
                        route = self._create_complete_route(forward_label, backward_label, join_node)
                        if route:
                            complete_routes.append(route)
        
        return complete_routes
    
    def _are_labels_compatible(self, forward_label: TruckLabel, backward_label: TruckLabel) -> bool:
        """Check if forward and backward labels can be joined"""
        # Check load compatibility
        if forward_label.load + backward_label.load > self.capacity:
            return False
        
        # Check path intersection (no cycles)
        forward_path_set = set(forward_label.path)
        backward_path_set = set(backward_label.path)
        
        # Only depot should be common
        common_nodes = forward_path_set.intersection(backward_path_set)
        if len(common_nodes) > 1:  # More than just the depot
            return False
        
        return True
    
    def _create_complete_route(self, forward_label: TruckLabel, backward_label: TruckLabel, 
                             join_node: int) -> Dict:
        """Create complete route from forward and backward labels"""
        # Construct complete path
        forward_path = forward_label.path
        backward_path = backward_label.path[1:]  # Remove depot from backward path
        backward_path.reverse()  # Reverse backward path
        
        complete_path = forward_path + backward_path
        
        # Calculate total cost and details
        total_cost = forward_label.cost + backward_label.cost
        total_time = forward_label.time + backward_label.time
        total_load = sum(self.demand_lookup.get(node, 0) for node in complete_path if node != 0)
        
        return {
            'path': complete_path,
            'cost': total_cost,
            'reduced_cost': total_cost,  # Simplified
            'details': {
                'route': complete_path,
                'total_time': total_time,
                'total_distance': self._calculate_path_distance(complete_path),
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
    
    def _is_forward_feasible(self, label: TruckLabel, current_node: int, 
                           next_node: int, demand_duals: Dict) -> bool:
        """Check if forward extension is feasible"""
        travel_time = self.time_matrix[current_node][next_node]
        arrival_time = label.time + travel_time
        
        # Time window check
        node_data = self.nodes[next_node]
        if 'time_window_end' in node_data:
            if arrival_time > node_data['time_window_end']:
                return False
        
        # Capacity check for delivery
        if next_node in self.demand_lookup:
            demand_qty = self.demand_lookup[next_node]
            if label.load < demand_qty:
                return False
        
        return True
    
    def _is_backward_feasible(self, label: TruckLabel, current_node: int, 
                            prev_node: int, demand_duals: Dict) -> bool:
        """Check if backward extension is feasible"""
        # In backward search, we need to ensure we don't exceed capacity
        if current_node in self.demand_lookup:
            demand_qty = self.demand_lookup[current_node]
            if label.load + demand_qty > self.capacity:
                return False
        
        return True
    
    def _extend_forward_label(self, label: TruckLabel, current_node: int, 
                            next_node: int, demand_duals: Dict, resource_duals: Dict) -> TruckLabel:
        """Extend label in forward direction"""
        travel_distance = self.distance_matrix[current_node][next_node]
        travel_time = self.time_matrix[current_node][next_node]
        travel_cost = travel_distance * self.cost_per_km
        
        # Calculate delivery
        pickup_load = 0
        delivery_revenue = 0
        if next_node in self.demand_lookup:
            demand_qty = self.demand_lookup[next_node]
            pickup_load = -demand_qty  # Delivery reduces load
            delivery_revenue = demand_duals.get(next_node, 0) * demand_qty
        
        # Calculate resource cost
        arrival_time = label.time + travel_time
        resource_cost = self._calculate_resource_cost(
            label.time, arrival_time, resource_duals
        )
        
        reduced_cost = travel_cost + resource_cost - delivery_revenue
        
        return label.extend_to(next_node, travel_time, reduced_cost, -pickup_load)
    
    def _extend_backward_label(self, label: TruckLabel, current_node: int, 
                             prev_node: int, demand_duals: Dict, resource_duals: Dict) -> TruckLabel:
        """Extend label in backward direction"""
        travel_distance = self.distance_matrix[prev_node][current_node]
        travel_time = self.time_matrix[prev_node][current_node]
        travel_cost = travel_distance * self.cost_per_km
        
        # In backward search, we pick up demand at current node
        pickup_load = 0
        pickup_revenue = 0
        if current_node in self.demand_lookup:
            demand_qty = self.demand_lookup[current_node]
            pickup_load = demand_qty  # Pickup increases load in backward search
            pickup_revenue = demand_duals.get(current_node, 0) * demand_qty
        
        reduced_cost = travel_cost - pickup_revenue
        
        return TruckLabel(
            node=prev_node,
            time=label.time + travel_time,
            load=label.load + pickup_load,
            cost=label.cost + reduced_cost,
            path=[prev_node] + label.path
        )
    
    def _extract_resource_duals(self, duals: Dict[str, float]) -> Dict[str, List[float]]:
        """Extract dual values for resource constraints by time slice"""
        resource_duals = {'truck': [], 'pilot': []}
        
        for t in range(96):  # 96 time slices
            truck_key = f"truck_cap_{t}"
            pilot_key = f"pilot_cap_{t}"
            
            resource_duals['truck'].append(duals.get(truck_key, 0.0))
            resource_duals['pilot'].append(duals.get(pilot_key, 0.0))
        
        return resource_duals
    
    def _calculate_resource_cost(self, start_time: int, end_time: int, 
                               resource_duals: Dict[str, List[float]]) -> float:
        """Calculate resource cost for using truck in time window"""
        cost = 0.0
        start_slice = start_time // 15
        end_slice = end_time // 15
        
        for t in range(start_slice, min(end_slice + 1, len(resource_duals['truck']))):
            cost += resource_duals['truck'][t]
        
        return cost