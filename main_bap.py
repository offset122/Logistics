import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
import logging

from data_structures import Column, BapNode, ResourceTracker
from algorithm1_ula_t import TruckULASolver
from algorithm2_ula_d import DroneULASolver  
from algorithm3_sela_m import MetroSELASolver
from algorithm4_bla_t import TruckBLASolver
from algorithm5_bla_d import DroneBLASolver
from algorithm6_bala_m import MetroBALASolver

class BranchAndPriceOptimizer:
    """Main Branch-and-Price optimizer implementing Algorithm 7"""
    
    def __init__(self, nodes, demands, vehicles, metro_data, params):
        self.nodes = nodes
        self.demands = demands
        self.vehicles = vehicles
        self.metro_data = metro_data
        self.params = params
        
        # Initialize data structures
        self.columns = []
        self.bap_tree = []
        self.resource_tracker = ResourceTracker()
        
        # Algorithm solvers
        self.algorithms = {
            1: TruckULASolver(nodes, demands, vehicles, params),
            2: DroneULASolver(nodes, demands, vehicles, params),
            3: MetroSELASolver(nodes, metro_data, params),
            4: TruckBLASolver(nodes, demands, vehicles, params),
            5: DroneBLASolver(nodes, demands, vehicles, params),
            6: MetroBALASolver(nodes, metro_data, params)
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.max_iterations = params.get('max_iterations', 100)
        self.tolerance = params.get('tolerance', 1e-6)
        self.time_limit = params.get('time_limit', 3600)  # 1 hour
        
        # Results tracking
        self.iteration_log = []
        self.runtime_comparison = {}
        
    def build_initial_columns(self) -> List[Column]:
        """Build initial feasible columns for warm start"""
        initial_columns = []
        
        # Create simple direct routes for each demand
        for demand in self.demands:
            node_id = demand['node_id']
            demand_qty = demand['demand']
            
            # Truck direct route
            truck_column = Column(
                type='truck',
                id=f"truck_direct_{node_id}",
                direct_cost=100.0,  # Base cost
                reduced_cost=0.0,
                details={
                    'route': [0, node_id, 0],  # Depot -> Node -> Depot
                    'load': demand_qty,
                    'time_windows': [],
                    'distance': 0.0
                }
            )
            initial_columns.append(truck_column)
            
            # Drone direct route (if capacity allows)
            drone_capacity = next((v['capacity'] for v in self.vehicles if v['vehicle_type'] == 'drone'), 10)
            if demand_qty <= drone_capacity:
                drone_column = Column(
                    type='drone',
                    id=f"drone_direct_{node_id}",
                    direct_cost=80.0,  # Generally cheaper than truck
                    reduced_cost=0.0,
                    details={
                        'route': [0, node_id, 0],
                        'load': demand_qty,
                        'energy_consumption': 50.0,
                        'flight_time': 30
                    }
                )
                initial_columns.append(drone_column)
        
        return initial_columns
    
    def build_rmp_model(self, columns: List[Column], constraints: List[Dict] = None) -> gp.Model:
        """Build Restricted Master Problem using Gurobi"""
        model = gp.Model("RMP")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output
        
        # Decision variables - lambda for route selection, mu for metro schedules
        lambda_vars = {}
        mu_vars = {}
        
        for col in columns:
            if col.type in ['truck', 'drone']:
                lambda_vars[col.id] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    obj=col.direct_cost,
                    name=f"lambda_{col.id}"
                )
            elif col.type == 'metro':
                mu_vars[col.id] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    obj=col.direct_cost,
                    name=f"mu_{col.id}"
                )
        
        # Constraints
        
        # 1. Demand satisfaction constraints
        for demand in self.demands:
            node_id = demand['node_id']
            demand_qty = demand['demand']
            
            expr = gp.LinExpr()
            for col in columns:
                if col.type in ['truck', 'drone']:
                    if node_id in col.details.get('route', []):
                        expr += lambda_vars[col.id] * col.details.get('load', 0)
                elif col.type == 'metro':
                    # Check if metro schedule serves this demand
                    if self._metro_serves_demand(col, node_id):
                        expr += mu_vars[col.id] * col.details.get('load', 0)
            
            model.addConstr(expr >= demand_qty, name=f"demand_{node_id}")
        
        # 2. Resource capacity constraints (per time slice)
        time_slices = range(self.resource_tracker.num_slices)
        
        for t in time_slices:
            # Drone capacity constraint
            drone_expr = gp.LinExpr()
            for col in columns:
                if col.type == 'drone' and self._column_uses_time_slice(col, t):
                    drone_expr += lambda_vars[col.id]
            
            model.addConstr(
                drone_expr <= self.resource_tracker.drone_capacity[t],
                name=f"drone_cap_{t}"
            )
            
            # Truck capacity constraint
            truck_expr = gp.LinExpr()
            for col in columns:
                if col.type == 'truck' and self._column_uses_time_slice(col, t):
                    truck_expr += lambda_vars[col.id]
            
            model.addConstr(
                truck_expr <= self.resource_tracker.truck_capacity[t],
                name=f"truck_cap_{t}"
            )
            
            # Pilot capacity constraint
            pilot_expr = gp.LinExpr()
            for col in columns:
                if col.type == 'drone' and self._column_uses_time_slice(col, t):
                    pilot_expr += lambda_vars[col.id]
            
            model.addConstr(
                pilot_expr <= self.resource_tracker.pilot_capacity[t],
                name=f"pilot_cap_{t}"
            )
        
        # 3. Metro schedule constraints (at most one schedule per run)
        metro_runs = self._get_metro_runs()
        for run_id in metro_runs:
            metro_expr = gp.LinExpr()
            for col in columns:
                if col.type == 'metro' and col.details.get('run_id') == run_id:
                    metro_expr += mu_vars[col.id]
            
            model.addConstr(metro_expr <= 1, name=f"metro_run_{run_id}")
        
        # 4. Inventory balance constraints
        self._add_inventory_constraints(model, columns, lambda_vars, mu_vars)
        
        # 5. Branching constraints (if any)
        if constraints:
            self._add_branching_constraints(model, constraints, lambda_vars, mu_vars)
        
        model.update()
        return model, lambda_vars, mu_vars
    
    def solve_rmp(self, columns: List[Column], constraints: List[Dict] = None) -> Dict[str, Any]:
        """Solve Restricted Master Problem and extract dual values"""
        model, lambda_vars, mu_vars = self.build_rmp_model(columns, constraints)
        
        # Solve the model
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            return {
                'status': 'infeasible',
                'objective': float('inf'),
                'duals': {},
                'solution': {}
            }
        
        # Extract solution
        solution = {}
        for var_id, var in lambda_vars.items():
            solution[var_id] = var.x
        for var_id, var in mu_vars.items():
            solution[var_id] = var.x
        
        # Extract dual values
        duals = {}
        for constr in model.getConstrs():
            duals[constr.ConstrName] = constr.Pi
        
        return {
            'status': 'optimal',
            'objective': model.objVal,
            'duals': duals,
            'solution': solution
        }
    
    def run_pricing_algorithms(self, duals: Dict[str, float], algorithms: List[int]) -> List[Column]:
        """Run selected pricing algorithms to generate new columns"""
        new_columns = []
        
        for alg_id in algorithms:
            if alg_id in self.algorithms:
                self.logger.info(f"Running pricing algorithm {alg_id}")
                start_time = time.time()
                
                try:
                    # Run the pricing algorithm
                    columns = self.algorithms[alg_id].solve(duals)
                    
                    # Filter columns with negative reduced cost
                    negative_columns = [col for col in columns if col.reduced_cost < -self.tolerance]
                    new_columns.extend(negative_columns)
                    
                    # Track runtime
                    runtime = time.time() - start_time
                    if alg_id not in self.runtime_comparison:
                        self.runtime_comparison[alg_id] = []
                    self.runtime_comparison[alg_id].append(runtime)
                    
                    self.logger.info(f"Algorithm {alg_id}: {len(negative_columns)} columns, {runtime:.3f}s")
                    
                except Exception as e:
                    self.logger.error(f"Error in algorithm {alg_id}: {str(e)}")
                    continue
        
        return new_columns
    
    def branch_and_bound(self, root_node: BapNode) -> Dict[str, Any]:
        """Branch-and-bound procedure"""
        node_queue = [root_node]
        best_solution = None
        best_objective = float('inf')
        
        while node_queue:
            current_node = node_queue.pop(0)  # BFS
            
            # Solve LP relaxation at current node
            rmp_result = self.solve_rmp(current_node.columns, current_node.constraints)
            
            if rmp_result['status'] == 'infeasible':
                current_node.is_feasible = False
                continue
            
            current_node.lower_bound = rmp_result['objective']
            
            # Pruning
            if current_node.lower_bound >= best_objective:
                continue
            
            # Check if solution is integer
            if self._is_integer_solution(rmp_result['solution']):
                current_node.is_integer = True
                if current_node.lower_bound < best_objective:
                    best_objective = current_node.lower_bound
                    best_solution = rmp_result['solution']
                continue
            
            # Branch on fractional variables
            branching_candidates = self._get_branching_candidates(rmp_result['solution'])
            
            if branching_candidates:
                var_name, var_value = branching_candidates[0]
                
                # Create two child nodes
                # Left child: variable = 0
                left_child = BapNode(
                    id=f"{current_node.id}_L",
                    parent_id=current_node.id,
                    columns=current_node.columns.copy(),
                    constraints=current_node.constraints.copy(),
                    lower_bound=0.0,
                    depth=current_node.depth + 1
                )
                left_child.add_branching_constraint(var_name, '=', 0.0)
                
                # Right child: variable = 1
                right_child = BapNode(
                    id=f"{current_node.id}_R",
                    parent_id=current_node.id,
                    columns=current_node.columns.copy(),
                    constraints=current_node.constraints.copy(),
                    lower_bound=0.0,
                    depth=current_node.depth + 1
                )
                right_child.add_branching_constraint(var_name, '=', 1.0)
                
                node_queue.extend([left_child, right_child])
        
        return {
            'best_objective': best_objective,
            'best_solution': best_solution,
            'nodes_explored': len(self.bap_tree)
        }
    
    def solve(self, algorithms: List[int] = None) -> Dict[str, Any]:
        """Main Branch-and-Price solving procedure (Algorithm 7)"""
        if algorithms is None:
            algorithms = [1, 2, 3, 4, 5, 6]  # Run all algorithms
        
        start_time = time.time()
        self.logger.info("Starting Branch-and-Price optimization")
        
        # Initialize with basic columns
        self.columns = self.build_initial_columns()
        
        # Create root node
        root_node = BapNode(
            id="root",
            parent_id=None,
            columns=self.columns,
            constraints=[],
            lower_bound=0.0
        )
        
        # Column generation loop
        iteration = 0
        converged = False
        
        while not converged and iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Iteration {iteration}")
            
            # Solve RMP
            rmp_result = self.solve_rmp(root_node.columns)
            
            if rmp_result['status'] == 'infeasible':
                return {
                    'status': 'infeasible',
                    'message': 'Master problem is infeasible'
                }
            
            current_objective = rmp_result['objective']
            
            # Run pricing algorithms
            new_columns = self.run_pricing_algorithms(rmp_result['duals'], algorithms)
            
            if not new_columns:
                self.logger.info("No columns with negative reduced cost found - converged")
                converged = True
            else:
                # Add new columns to master problem
                root_node.columns.extend(new_columns)
                self.logger.info(f"Added {len(new_columns)} new columns")
            
            # Log iteration results
            self.iteration_log.append({
                'iteration': iteration,
                'objective': current_objective,
                'columns_added': len(new_columns),
                'total_columns': len(root_node.columns)
            })
            
            # Check time limit
            if time.time() - start_time > self.time_limit:
                self.logger.warning("Time limit reached")
                break
        
        # Check if LP solution is integer
        if self._is_integer_solution(rmp_result['solution']):
            final_result = {
                'status': 'optimal',
                'objective_value': rmp_result['objective'],
                'solution': rmp_result['solution']
            }
        else:
            # Run branch-and-bound
            self.logger.info("LP solution is fractional - starting branch-and-bound")
            bb_result = self.branch_and_bound(root_node)
            final_result = {
                'status': 'optimal' if bb_result['best_solution'] else 'no_integer_solution',
                'objective_value': bb_result['best_objective'],
                'solution': bb_result['best_solution']
            }
        
        # Process final results
        runtime = time.time() - start_time
        routes, metro_schedules = self._extract_routes_and_schedules(final_result['solution'])
        
        return {
            'status': final_result['status'],
            'objective_value': final_result['objective_value'],
            'runtime': runtime,
            'iterations': iteration,
            'routes': routes,
            'metro_schedules': metro_schedules,
            'iteration_log': self.iteration_log,
            'runtime_comparison': self.runtime_comparison,
            'columns_generated': len(root_node.columns)
        }
    
    # Helper methods
    
    def _metro_serves_demand(self, metro_column: Column, node_id: int) -> bool:
        """Check if metro schedule serves specific demand"""
        schedule_path = metro_column.details.get('schedule_path', [])
        return any(stop.get('station') == node_id for stop in schedule_path)
    
    def _column_uses_time_slice(self, column: Column, time_slice: int) -> bool:
        """Check if column uses specific time slice"""
        if column.type in ['truck', 'drone']:
            start_time = column.details.get('start_time', 0)
            end_time = column.details.get('end_time', 0)
            start_slice = self.resource_tracker.get_time_slice(start_time)
            end_slice = self.resource_tracker.get_time_slice(end_time)
            return start_slice <= time_slice <= end_slice
        return False
    
    def _get_metro_runs(self) -> List[str]:
        """Get list of metro run IDs"""
        runs = set()
        for entry in self.metro_data:
            runs.add(entry.get('run_id', 'default'))
        return list(runs)
    
    def _add_inventory_constraints(self, model, columns, lambda_vars, mu_vars):
        """Add inventory balance constraints for metro system"""
        # Simplified inventory constraints - would be more complex in practice
        for station_id in range(len(self.nodes)):
            up_expr = gp.LinExpr()
            down_expr = gp.LinExpr()
            
            for col in columns:
                if col.type == 'metro':
                    schedule = col.details.get('schedule_path', [])
                    for stop in schedule:
                        if stop.get('station') == station_id:
                            if stop.get('direction') == 'up':
                                up_expr += mu_vars[col.id] * stop.get('load', 0)
                            else:
                                down_expr += mu_vars[col.id] * stop.get('load', 0)
            
            # Balance constraint
            model.addConstr(up_expr == down_expr, name=f"inventory_balance_{station_id}")
    
    def _add_branching_constraints(self, model, constraints, lambda_vars, mu_vars):
        """Add branching constraints to RMP"""
        for constraint in constraints:
            var_name = constraint['variable']
            constraint_type = constraint['type']
            value = constraint['value']
            
            if var_name in lambda_vars:
                var = lambda_vars[var_name]
            elif var_name in mu_vars:
                var = mu_vars[var_name]
            else:
                continue
            
            if constraint_type == '=':
                model.addConstr(var == value, name=f"branch_{var_name}")
            elif constraint_type == '<=':
                model.addConstr(var <= value, name=f"branch_{var_name}")
            elif constraint_type == '>=':
                model.addConstr(var >= value, name=f"branch_{var_name}")
    
    def _is_integer_solution(self, solution: Dict[str, float]) -> bool:
        """Check if solution is integer within tolerance"""
        for var_name, value in solution.items():
            if abs(value - round(value)) > self.tolerance:
                return False
        return True
    
    def _get_branching_candidates(self, solution: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get variables for branching (most fractional first)"""
        candidates = []
        for var_name, value in solution.items():
            if abs(value - round(value)) > self.tolerance:
                fractionality = min(value - int(value), int(value) + 1 - value)
                candidates.append((var_name, fractionality))
        
        # Sort by fractionality (most fractional first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _extract_routes_and_schedules(self, solution: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """Extract routes and schedules from solution"""
        routes = []
        metro_schedules = []
        
        for col in self.columns:
            var_name = f"lambda_{col.id}" if col.type in ['truck', 'drone'] else f"mu_{col.id}"
            
            if var_name in solution and solution[var_name] > self.tolerance:
                if col.type in ['truck', 'drone']:
                    routes.append({
                        'type': col.type,
                        'id': col.id,
                        'cost': col.direct_cost,
                        'details': col.details,
                        'selection_value': solution[var_name]
                    })
                elif col.type == 'metro':
                    metro_schedules.append({
                        'id': col.id,
                        'cost': col.direct_cost,
                        'direction': col.details.get('direction', 'up'),
                        'departure': col.details.get('departure_time', 0),
                        'arrival': col.details.get('arrival_time', 0),
                        'load': col.details.get('load', 0),
                        'selection_value': solution[var_name]
                    })
        
        return routes, metro_schedules