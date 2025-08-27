from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid

@dataclass
class Column:
    """Represents a column (route/schedule) in the master problem"""
    type: str  # 'drone', 'truck', 'metro'
    id: str
    direct_cost: float
    reduced_cost: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class DroneLabel:
    """Label for drone route in RCSPP"""
    node: int
    time: int
    load: int
    energy: float
    cost: float
    path: List[int] = field(default_factory=list)
    parent: Optional['DroneLabel'] = None
    
    def is_dominated_by(self, other: 'DroneLabel') -> bool:
        """Check if this label is dominated by another"""
        if (self.node != other.node or 
            self.time > other.time or 
            self.load > other.load or 
            self.energy < other.energy or 
            self.cost > other.cost):
            return False
        return True
    
    def extend_to(self, next_node: int, travel_time: int, energy_consumption: float, 
                  travel_cost: float, pickup_load: int = 0) -> 'DroneLabel':
        """Extend label to next node"""
        new_time = self.time + travel_time
        new_load = max(0, self.load - pickup_load)  # Delivery reduces load
        new_energy = self.energy - energy_consumption
        new_cost = self.cost + travel_cost
        new_path = self.path + [next_node]
        
        return DroneLabel(
            node=next_node,
            time=new_time,
            load=new_load,
            energy=new_energy,
            cost=new_cost,
            path=new_path,
            parent=self
        )

@dataclass
class TruckLabel:
    """Label for truck route in RCSPP"""
    node: int
    time: int
    load: int
    cost: float
    path: List[int] = field(default_factory=list)
    parent: Optional['TruckLabel'] = None
    
    def is_dominated_by(self, other: 'TruckLabel') -> bool:
        """Check if this label is dominated by another"""
        if (self.node != other.node or 
            self.time > other.time or 
            self.load > other.load or 
            self.cost > other.cost):
            return False
        return True
    
    def extend_to(self, next_node: int, travel_time: int, travel_cost: float, 
                  pickup_load: int = 0) -> 'TruckLabel':
        """Extend label to next node"""
        new_time = self.time + travel_time
        new_load = max(0, self.load - pickup_load)  # Delivery reduces load
        new_cost = self.cost + travel_cost
        new_path = self.path + [next_node]
        
        return TruckLabel(
            node=next_node,
            time=new_time,
            load=new_load,
            cost=new_cost,
            path=new_path,
            parent=self
        )

@dataclass
class MetroLabel:
    """Label for metro schedule in RCSPP"""
    station: int
    time: int
    load: int
    cost: float
    schedule_path: List[Dict] = field(default_factory=list)
    parent: Optional['MetroLabel'] = None
    
    def is_dominated_by(self, other: 'MetroLabel') -> bool:
        """Check if this label is dominated by another"""
        if (self.station != other.station or 
            self.time > other.time or 
            self.load > other.load or 
            self.cost > other.cost):
            return False
        return True
    
    def extend_to(self, next_station: int, departure_time: int, arrival_time: int, 
                  transport_cost: float, pickup_load: int = 0) -> 'MetroLabel':
        """Extend label to next station via metro"""
        new_load = max(0, self.load - pickup_load)
        new_cost = self.cost + transport_cost
        new_schedule = self.schedule_path + [{
            'from_station': self.station,
            'to_station': next_station,
            'departure_time': departure_time,
            'arrival_time': arrival_time,
            'load': new_load
        }]
        
        return MetroLabel(
            station=next_station,
            time=arrival_time,
            load=new_load,
            cost=new_cost,
            schedule_path=new_schedule,
            parent=self
        )

@dataclass
class MetroBALabel:
    """Big-Arc Metro Label for optimized metro scheduling"""
    from_station: int
    to_station: int
    departure_time: int
    arrival_time: int
    load: int
    cost: float
    big_arc_path: List[Dict] = field(default_factory=list)
    parent: Optional['MetroBALabel'] = None
    
    def is_dominated_by(self, other: 'MetroBALabel') -> bool:
        """Check if this label is dominated by another"""
        if (self.from_station != other.from_station or
            self.to_station != other.to_station or
            self.departure_time > other.departure_time or 
            self.arrival_time > other.arrival_time or
            self.load > other.load or 
            self.cost > other.cost):
            return False
        return True

@dataclass 
class BapNode:
    """Branch-and-Price tree node"""
    id: str
    parent_id: Optional[str]
    columns: List[Column]
    constraints: List[Dict[str, Any]]
    lower_bound: float
    upper_bound: float = float('inf')
    is_integer: bool = False
    is_feasible: bool = True
    branching_variable: Optional[str] = None
    branching_value: Optional[float] = None
    depth: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_branching_constraint(self, variable: str, constraint_type: str, value: float):
        """Add branching constraint to node"""
        constraint = {
            'variable': variable,
            'type': constraint_type,  # '=' or '<=' or '>='
            'value': value
        }
        self.constraints.append(constraint)
        self.branching_variable = variable
        self.branching_value = value
    
    def is_integer_solution(self, tolerance: float = 1e-6) -> bool:
        """Check if current solution is integer"""
        # This would be implemented based on the solution values
        return self.is_integer
    
    def can_branch(self) -> bool:
        """Check if node can be branched"""
        return (self.is_feasible and 
                not self.is_integer and 
                self.lower_bound < self.upper_bound)

class LabelSet:
    """Container for managing labels in RCSPP algorithms"""
    
    def __init__(self):
        self.labels = []
        self.dominated_count = 0
    
    def add_label(self, label):
        """Add label with dominance checking"""
        # Check if new label is dominated
        for existing_label in self.labels:
            if label.is_dominated_by(existing_label):
                self.dominated_count += 1
                return False
        
        # Remove labels dominated by new label
        self.labels = [l for l in self.labels if not l.is_dominated_by(label)]
        
        # Add new label
        self.labels.append(label)
        return True
    
    def get_best_labels(self, count: int = None):
        """Get best labels sorted by cost"""
        sorted_labels = sorted(self.labels, key=lambda x: x.cost)
        return sorted_labels[:count] if count else sorted_labels
    
    def clear(self):
        """Clear all labels"""
        self.labels = []
        self.dominated_count = 0
    
    def size(self):
        """Get number of non-dominated labels"""
        return len(self.labels)

class ResourceTracker:
    """Track resource consumption and constraints"""
    
    def __init__(self, time_horizon: int = 1440, time_resolution: int = 15):
        self.time_horizon = time_horizon  # Total time in minutes (24 hours)
        self.time_resolution = time_resolution  # Time slice in minutes
        self.num_slices = time_horizon // time_resolution
        
        # Resource usage per time slice
        self.drone_usage = [0] * self.num_slices
        self.truck_usage = [0] * self.num_slices
        self.pilot_usage = [0] * self.num_slices
        
        # Resource capacities per time slice
        self.drone_capacity = [10] * self.num_slices  # Default capacities
        self.truck_capacity = [5] * self.num_slices
        self.pilot_capacity = [8] * self.num_slices
    
    def get_time_slice(self, time_minutes: int) -> int:
        """Convert time in minutes to time slice index"""
        return min(time_minutes // self.time_resolution, self.num_slices - 1)
    
    def is_resource_available(self, vehicle_type: str, start_time: int, end_time: int) -> bool:
        """Check if resource is available in time window"""
        start_slice = self.get_time_slice(start_time)
        end_slice = self.get_time_slice(end_time)
        
        for slice_idx in range(start_slice, min(end_slice + 1, self.num_slices)):
            if vehicle_type == 'drone':
                if self.drone_usage[slice_idx] >= self.drone_capacity[slice_idx]:
                    return False
            elif vehicle_type == 'truck':
                if self.truck_usage[slice_idx] >= self.truck_capacity[slice_idx]:
                    return False
            elif vehicle_type == 'pilot':
                if self.pilot_usage[slice_idx] >= self.pilot_capacity[slice_idx]:
                    return False
        
        return True
    
    def allocate_resource(self, vehicle_type: str, start_time: int, end_time: int):
        """Allocate resource for time window"""
        start_slice = self.get_time_slice(start_time)
        end_slice = self.get_time_slice(end_time)
        
        for slice_idx in range(start_slice, min(end_slice + 1, self.num_slices)):
            if vehicle_type == 'drone':
                self.drone_usage[slice_idx] += 1
            elif vehicle_type == 'truck':
                self.truck_usage[slice_idx] += 1
            elif vehicle_type == 'pilot':
                self.pilot_usage[slice_idx] += 1
    
    def reset(self):
        """Reset all resource usage"""
        self.drone_usage = [0] * self.num_slices
        self.truck_usage = [0] * self.num_slices  
        self.pilot_usage = [0] * self.num_slices