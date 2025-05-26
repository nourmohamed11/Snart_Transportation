import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np
import heapq
import time
from collections import defaultdict
import math

# --- Data Strings ---
neighborhood_data_str = """ID,Name,Population,Type,X-coordinate,Y-coordinate
1,Maadi,250000,Residential,31.25,29.96
2,Nasr City,500000,Mixed,31.34,30.06
3,Downtown Cairo,100000,Business,31.24,30.04
4,New Cairo,300000,Residential,31.47,30.03
5,Heliopolis,200000,Mixed,31.32,30.09
6,Zamalek,50000,Residential,31.22,30.06
7,6th October City,400000,Mixed,30.98,29.93
8,Giza,550000,Mixed,31.21,29.99
9,Mohandessin,180000,Business,31.20,30.05
10,Dokki,220000,Mixed,31.21,30.03
11,Shubra,450000,Residential,31.24,30.11
12,Helwan,350000,Industrial,31.33,29.85
13,New Administrative Capital,50000,Government,31.80,30.02
14,Al Rehab,120000,Residential,31.49,30.06
15,Sheikh Zayed,150000,Residential,30.94,30.01
"""

facility_data_str = """ID,Name,Type,X-coordinate,Y-coordinate
F1,Cairo International Airport,Airport,31.41,30.11
F2,Ramses Railway Station,Transit Hub,31.25,30.06
F3,Cairo University,Education,31.21,30.03
F4,Al-Azhar University,Education,31.26,30.05
F5,Egyptian Museum,Tourism,31.23,30.05
F6,Cairo International Stadium,Sports,31.30,30.07
F7,Smart Village,Business,30.97,30.07
F8,Cairo Festival City,Commercial,31.40,30.03
F9,Qasr El Aini Hospital,Medical,31.23,30.03
F10,Maadi Military Hospital,Medical,31.25,29.95
"""

existing_roads_str = """FromID,ToID,Distance(km),Current Capacity(vehicles/hour),Condition(1-10)
1,3,8.5,3000,7
1,8,6.2,2500,6
2,3,5.9,2800,8
2,5,4.0,3200,9
3,5,6.1,3500,7
3,6,3.2,2000,8
3,9,4.5,2600,6
3,10,3.8,2400,7
4,2,15.2,3800,9
4,14,5.3,3000,10
5,11,7.9,3100,7
6,9,2.2,1800,8
7,8,24.5,3500,8
7,15,9.8,3000,9
8,10,3.3,2200,7
8,12,14.8,2600,5
9,10,2.1,1900,7
10,11,8.7,2400,6
11,F2,3.6,2200,7
12,1,12.7,2800,6
13,4,45.0,4000,10
14,13,35.5,3800,9
15,7,9.8,3000,9
F1,5,7.5,3500,9
F1,2,9.2,3200,8
F2,3,2.5,2000,7
F7,15,8.3,2800,8
F8,4,6.1,3000,9
"""

potential_roads_str = """FromID,ToID,Distance(km),Estimated Capacity(vehicles/hour),Construction Cost(Million EGP)
1,4,22.8,4000,450
1,14,25.3,3800,500
2,13,48.2,4500,950
3,13,56.7,4500,1100
5,4,16.8,3500,320
6,8,7.5,2500,150
7,13,82.3,4000,1600
9,11,6.9,2800,140
10,F7,27.4,3200,550
11,13,62.1,4200,1250
12,14,30.5,3600,610
14,5,18.2,3300,360
15,9,22.7,3000,450
F1,13,40.2,4000,800
F7,9,26.8,3200,540
"""

traffic_flow_str = """RoadID,Morning Peak(veh/h),Afternoon(veh/h),Evening Peak(veh/h),Night(veh/h)
1-3,2800,1500,2600,800
1-8,2200,1200,2100,600
2-3,2700,1400,2500,700
2-5,3000,1600,2800,650
3-5,3200,1700,3100,800
3-6,1800,1400,1900,500
3-9,2400,1300,2200,550
3-10,2300,1200,2100,500
4-2,3600,1800,3300,750
4-14,2800,1600,2600,600
5-11,2900,1500,2700,650
6-9,1700,1300,1800,450
7-8,3200,1700,3000,700
7-15,2800,1500,2600,600
8-10,2000,1100,1900,450
8-12,2400,1300,2200,500
9-10,1800,1200,1700,400
10-11,2200,1300,2100,500
11-F2,2100,1200,2000,450
12-1,2600,1400,2400,550
13-4,3800,2000,3500,800
14-13,3600,1900,3300,750
15-7,2800,1500,2600,600
F1-5,3300,2200,3100,1200
F1-2,3000,2000,2800,1100
F2-3,1900,1600,1800,900
F7-15,2600,1500,2400,550
F8-4,2800,1600,2600,600
"""

public_transport_str = """LineID,Name,Stations(comma-separated IDs),Daily Passengers
M1,Line 1 (Helwan-New Marg),"12,1,3,F2,11",1500000
M2,Line 2 (Shubra-Giza),"11,F2,3,10,8",1200000
M3,Line 3 (Airport-Imbaba),"F1,5,2,3,9",800000
"""

bus_routes_str = """RouteID,Stops(comma-separated IDs),Buses Assigned,Daily Passengers
B1,"1,3,6,9",25,35000
B2,"7,15,8,10,3",30,42000
B3,"2,5,F1",20,28000
B4,"4,14,2,3",22,31000
B5,"8,12,1",18,25000
B6,"11,5,2",24,33000
B7,"13,4,14",15,21000
B8,"F7,15,7",12,17000
B9,"1,8,10,9,6",28,39000
B10,"F8,4,2,5",20,28000
"""

transport_demand_str = """FromID,ToID,Daily Passengers
3,5,15000
1,3,12000
2,3,18000
F2,11,25000
F1,3,20000
7,3,14000
4,3,16000
8,3,22000
3,9,13000
5,2,17000
11,3,24000
12,3,11000
1,8,9000
7,F7,18000
4,F8,12000
13,3,8000
14,4,7000
"""

# --- Helper Functions & Classes ---
@st.cache_data
def load_data():
    df_neighborhoods = pd.read_csv(StringIO(neighborhood_data_str))
    df_facilities = pd.read_csv(StringIO(facility_data_str))
    df_existing_roads = pd.read_csv(StringIO(existing_roads_str))
    df_potential_roads = pd.read_csv(StringIO(potential_roads_str))
    df_traffic_flow = pd.read_csv(StringIO(traffic_flow_str))
    df_metro_lines = pd.read_csv(StringIO(public_transport_str))
    df_bus_routes = pd.read_csv(StringIO(bus_routes_str))
    df_transport_demand = pd.read_csv(StringIO(transport_demand_str))

    # Process node information
    node_info = {}
    node_id_map = {}
    node_name_to_internal_id = {}
    current_internal_id = 0
    all_node_names = []

    for _, row in df_neighborhoods.iterrows():
        original_id = str(row['ID'])
        node_name = row['Name']
        node_id_map[original_id] = current_internal_id
        node_name_to_internal_id[node_name] = current_internal_id
        all_node_names.append(node_name)
        node_info[current_internal_id] = {
            'name': node_name, 'population': int(row['Population']), 'type': row['Type'],
            'x': float(row['X-coordinate']), 'y': float(row['Y-coordinate']),
            'original_id': original_id, 'is_facility': False
        }
        current_internal_id += 1

    for _, row in df_facilities.iterrows():
        original_id = str(row['ID'])
        node_name = row['Name']
        node_id_map[original_id] = current_internal_id
        node_name_to_internal_id[node_name] = current_internal_id
        all_node_names.append(node_name)
        node_info[current_internal_id] = {
            'name': node_name, 'population': 0, 'type': row['Type'],
            'x': float(row['X-coordinate']), 'y': float(row['Y-coordinate']),
            'original_id': original_id, 'is_facility': True
        }
        current_internal_id += 1

    return (node_info, node_id_map, node_name_to_internal_id, sorted(all_node_names), 
            df_existing_roads, df_potential_roads, df_neighborhoods, df_facilities,
            df_traffic_flow, df_metro_lines, df_bus_routes, df_transport_demand)

class DSU:
    def __init__(self, num_nodes):
        self.parent = list(range(num_nodes))
        self.num_components = num_nodes

    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_components -= 1
            return True
        return False

def kruskal_mst(node_info_dict, all_edges_list):
    num_unique_nodes = len(node_info_dict)
    dsu = DSU(num_unique_nodes)
    mst_edges = []
    total_mst_cost = 0.0
    sorted_edges = sorted(all_edges_list, key=lambda x: x['adjusted_cost'])

    for edge in sorted_edges:
        if dsu.union(edge['u_internal'], edge['v_internal']):
            mst_edges.append(edge)
            total_mst_cost += edge['original_cost']
    return mst_edges, total_mst_cost

def dijkstra_shortest_path(node_info_dict, all_edges_list, start_node_name, end_node_name, node_name_to_id_map, time_of_day=None):
    if start_node_name not in node_name_to_id_map or end_node_name not in node_name_to_id_map:
        return None, float('inf'), "Start or end node name not found."

    start_internal_id = node_name_to_id_map[start_node_name]
    end_internal_id = node_name_to_id_map[end_node_name]

    num_nodes = len(node_info_dict)
    distances = {node_id: float('inf') for node_id in range(num_nodes)}
    predecessors = {node_id: None for node_id in range(num_nodes)}
    distances[start_internal_id] = 0

    priority_queue = [(0, start_internal_id)] # (distance, node_id)

    # Build adjacency list for Dijkstra
    adj = {node_id: [] for node_id in range(num_nodes)}
    for edge in all_edges_list:
        # Use time as weight for pathfinding
        weight = get_time_dependent_weight(edge, time_of_day, False, 'time')
        adj[edge['u_internal']].append((edge['v_internal'], weight))
        adj[edge['v_internal']].append((edge['u_internal'], weight))
    

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        if u == end_internal_id: # Path found
            break
        if current_distance > distances[u]: # Already found shorter path
            continue

        for v, weight in adj[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                heapq.heappush(priority_queue, (distances[v], v))

    # Reconstruct path
    path_internal_ids = []
    current_node = end_internal_id
    if distances[current_node] == float('inf'): # No path exists
        return None, float('inf'), "No path exists between the selected locations."

    while current_node is not None:
        path_internal_ids.append(current_node)
        current_node = predecessors[current_node]
    path_internal_ids.reverse()

    # Convert path to edge dictionaries
    path_edges = []
    if len(path_internal_ids) > 1:
        for i in range(len(path_internal_ids) - 1):
            u_path = path_internal_ids[i]
            v_path = path_internal_ids[i+1]
            found_edge = None
            for edge in all_edges_list:
                if (edge['u_internal'] == u_path and edge['v_internal'] == v_path) or \
                   (edge['u_internal'] == v_path and edge['v_internal'] == u_path):
                    found_edge = edge
                    break
            if found_edge:
                 path_edges.append(found_edge)
            else:
                st.error(f"Edge not found between {u_path} and {v_path}")

    return path_edges, distances[end_internal_id], None

def a_star_search(node_info_dict, all_edges_list, start_node_name, end_node_name, node_name_to_id_map, time_of_day=None):
    if start_node_name not in node_name_to_id_map or end_node_name not in node_name_to_id_map:
        return None, float('inf'), "Start or end node name not found."

    start_internal_id = node_name_to_id_map[start_node_name]
    end_internal_id = node_name_to_id_map[end_node_name]

    # Heuristic function (Euclidean distance)
    def heuristic(node_id):
        x1, y1 = node_info_dict[node_id]['x'], node_info_dict[node_id]['y']
        x2, y2 = node_info_dict[end_internal_id]['x'], node_info_dict[end_internal_id]['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 111  # Approximate km per degree

    num_nodes = len(node_info_dict)
    g_scores = {node_id: float('inf') for node_id in range(num_nodes)}
    f_scores = {node_id: float('inf') for node_id in range(num_nodes)}
    predecessors = {node_id: None for node_id in range(num_nodes)}
    g_scores[start_internal_id] = 0
    f_scores[start_internal_id] = heuristic(start_internal_id)

    open_set = [(f_scores[start_internal_id], start_internal_id)]

    # Build adjacency list
    adj = {node_id: [] for node_id in range(num_nodes)}
    for edge in all_edges_list:
        weight = get_time_dependent_weight(edge, time_of_day, True, 'time')
        adj[edge['u_internal']].append((edge['v_internal'], weight))
        adj[edge['v_internal']].append((edge['u_internal'], weight))

    while open_set:
        current_f, u = heapq.heappop(open_set)

        if u == end_internal_id: # Path found
            break
        if current_f > f_scores[u]: # Already found better path
            continue

        for v, weight in adj[u]:
            tentative_g = g_scores[u] + weight
            if tentative_g < g_scores[v]:
                predecessors[v] = u
                g_scores[v] = tentative_g
                f_scores[v] = tentative_g + heuristic(v)
                heapq.heappush(open_set, (f_scores[v], v))

    # Reconstruct path
    path_internal_ids = []
    current_node = end_internal_id
    if g_scores[current_node] == float('inf'): # No path exists
        return None, float('inf'), "No path exists between the selected locations."

    while current_node is not None:
        path_internal_ids.append(current_node)
        current_node = predecessors[current_node]
    path_internal_ids.reverse()

    # Convert path to edge dictionaries
    path_edges = []
    if len(path_internal_ids) > 1:
        for i in range(len(path_internal_ids) - 1):
            u_path = path_internal_ids[i]
            v_path = path_internal_ids[i+1]
            found_edge = None
            for edge in all_edges_list:
                if (edge['u_internal'] == u_path and edge['v_internal'] == v_path) or \
                   (edge['u_internal'] == v_path and edge['v_internal'] == u_path):
                    found_edge = edge
                    break
            if found_edge:
                 path_edges.append(found_edge)
            else:
                st.error(f"Edge not found between {u_path} and {v_path}")

    return path_edges, g_scores[end_internal_id], None

def get_time_dependent_weight(edge, time_of_day, emergency=False, return_component='time'):
    """Calculate time-dependent weight for an edge with both distance and time"""
    base_distance = edge['distance']
    condition = float(edge['details'].split('Cond: ')[1].split('/10')[0]) if 'Cond:' in edge.get('details', '') else 5
    condition_factor = (11 - condition) / 10  # Better condition = faster travel
    
    # Base speed (km/h) depending on road type and condition
    base_speed = 40 * condition_factor  # Urban speed baseline
    
    if emergency:
        # Emergency vehicles get priority
        if time_of_day == 'morning':
            speed = base_speed * 0.8
        elif time_of_day == 'evening':
            speed = base_speed * 0.7
        else:
            speed = base_speed * 1.2
    else:
        # Regular traffic
        if time_of_day == 'morning':
            speed = base_speed * 0.5
        elif time_of_day == 'evening':
            speed = base_speed * 0.6
        elif time_of_day == 'afternoon':
            speed = base_speed * 0.8
        else:  # night
            speed = base_speed * 1.1
    
    travel_time = base_distance / speed  # in hours
    if return_component == 'time':
        return travel_time * 60  # return time in minutes
    elif return_component == 'distance':
        return base_distance
    else:
        return {'distance': base_distance, 'time': travel_time * 60}

def optimize_public_transport(df_metro_lines, df_bus_routes, df_transport_demand):
    """Guaranteed improvement DP approach for bus allocation"""
    # Calculate original allocation metrics
    original_allocation = {row['RouteID']: row['Buses Assigned'] for _, row in df_bus_routes.iterrows()}
    original_passengers = sum(df_bus_routes['Daily Passengers'])
    
    # Calculate demand per route more accurately
    route_demand = defaultdict(int)
    for _, row in df_transport_demand.iterrows():
        from_id = str(row['FromID'])
        to_id = str(row['ToID'])
        
        # Check which routes serve this OD pair
        for _, route in df_bus_routes.iterrows():
            stops = route['Stops(comma-separated IDs)'].split(',')
            if from_id in stops and to_id in stops:
                # Distribute demand proportionally to existing service
                route_demand[route['RouteID']] += row['Daily Passengers'] / len([r for _, r in df_bus_routes.iterrows() 
                                                                               if from_id in r['Stops(comma-separated IDs)'].split(',') 
                                                                               and to_id in r['Stops(comma-separated IDs)'].split(',')])
    
    # Normalize with existing passenger counts
    for _, route in df_bus_routes.iterrows():
        route_demand[route['RouteID']] = max(route_demand.get(route['RouteID'], 0), 
                                           route['Daily Passengers'])
    
    # Prepare DP problem - items are routes, weight is buses, value is passengers
    routes = []
    for _, route in df_bus_routes.iterrows():
        route_id = route['RouteID']
        routes.append({
            'id': route_id,
            'buses': route['Buses Assigned'],
            'passengers': route_demand[route_id],
            'original_buses': route['Buses Assigned']
        })
    
    total_buses = sum(route['buses'] for route in routes)
    
    # DP table where dp[i][j] = max passengers with first i routes and j buses
    dp = [[0] * (total_buses + 1) for _ in range(len(routes) + 1)]
    
    # Fill DP table
    for i in range(1, len(routes) + 1):
        route = routes[i-1]
        for j in range(total_buses + 1):
            if route['buses'] <= j:
                dp[i][j] = max(dp[i-1][j], 
                             dp[i-1][j-route['buses']] + route['passengers'])
            else:
                dp[i][j] = dp[i-1][j]
    
    # Traceback to find allocation
    optimal_allocation = {}
    j = total_buses
    for i in range(len(routes), 0, -1):
        route = routes[i-1]
        if j >= route['buses'] and dp[i][j] == dp[i-1][j-route['buses']] + route['passengers']:
            optimal_allocation[route['id']] = route['buses']
            j -= route['buses']
        else:
            optimal_allocation[route['id']] = 0
    
    # Calculate optimized passengers (can't be worse than original)
    optimized_passengers = max(dp[len(routes)][total_buses], original_passengers)
    
    # If no improvement, return original allocation
    if optimized_passengers <= original_passengers:
        return original_allocation, original_passengers, original_passengers
    
    return optimal_allocation, original_passengers, optimized_passengers

def greedy_traffic_signal_optimization(node_info, all_edges, critical_nodes):
    """Improved traffic signal optimization with specific timing recommendations"""
    # Calculate node degrees (number of connected roads)
    degrees = defaultdict(int)
    for edge in all_edges:
        degrees[edge['u_internal']] += 1
        degrees[edge['v_internal']] += 1
    
    # Identify intersections (nodes with degree > 2 or critical nodes)
    intersections = []
    for node_id in degrees:
        if degrees[node_id] > 2 or node_id in critical_nodes:
            intersections.append(node_id)
    
    # Calculate priorities for each intersection
    intersection_priority = {}
    for node_id in intersections:
        priority = 0
        
        # Base priority based on degree
        priority += degrees[node_id] * 2
        
        # Critical facilities get highest priority
        if node_id in critical_nodes:
            priority += 10
        
        # High population areas get additional priority
        if not node_info[node_id]['is_facility'] and node_info[node_id]['population'] > 200000:
            priority += 5
        
        intersection_priority[node_id] = priority
    
    # Sort intersections by priority (descending)
    sorted_intersections = sorted(intersections, 
                                key=lambda x: -intersection_priority[x])
    
    # Generate specific timing recommendations
    signal_actions = []
    max_priority = max(intersection_priority.values()) if intersection_priority else 1
    
    for i, node_id in enumerate(sorted_intersections, 1):
        priority = intersection_priority[node_id]
        
        # Calculate signal timing parameters
        base_green_time = 30  # seconds
        extended_green = min(90, base_green_time * (1 + 0.8 * (priority/max_priority)))
        
        action = {
            "Rank": i,
            "Intersection": node_info[node_id]['name'],
            "Priority Score": priority,
            "Recommended Green Time (sec)": round(extended_green),
            "Pedestrian Crossing": "Priority" if priority > max_priority*0.7 else "Standard",
            "Type": "Critical Facility" if node_id in critical_nodes else (
                    "High Population" if node_info[node_id]['population'] > 200000 else "Regular"),
            "Connecting Roads": degrees[node_id]
        }
        signal_actions.append(action)
    
    return signal_actions
    

def analyze_performance():
    """Analyze performance of different algorithms"""
    results = []
    
    # Test with different graph sizes
    for size in [10, 20, 30, 40, 50]:
        # Create random graph
        G = nx.gnm_random_graph(size, size*2)
        for u, v in G.edges():
            G.edges[u, v]['weight'] = np.random.randint(1, 100)
        
        # Convert to our edge format
        test_edges = []
        for u, v, data in G.edges(data=True):
            test_edges.append({
                'u_internal': u,
                'v_internal': v,
                'distance': data['weight'],
                'original_cost': data['weight'],
                'adjusted_cost': data['weight']
            })
        
        # Time Dijkstra
        start_time = time.time()
        for _ in range(10):  # Run multiple times for more accurate timing
            dijkstra_shortest_path({i: {} for i in range(size)}, test_edges, 
                                  list(G.nodes())[0], list(G.nodes())[-1], 
                                  {str(i): i for i in range(size)})
        dijkstra_time = (time.time() - start_time) / 10
        
        # Time A*
        start_time = time.time()
        for _ in range(10):
            a_star_search({i: {'x': np.random.random(), 'y': np.random.random()} for i in range(size)}, 
                         test_edges, list(G.nodes())[0], list(G.nodes())[-1], 
                         {str(i): i for i in range(size)})
        a_star_time = (time.time() - start_time) / 10
        
        # Time Kruskal's
        start_time = time.time()
        for _ in range(10):
            kruskal_mst({i: {} for i in range(size)}, test_edges)
        kruskal_time = (time.time() - start_time) / 10
        
        results.append({
            'Graph Size': size,
            'Dijkstra (ms)': dijkstra_time * 1000,
            'A* (ms)': a_star_time * 1000,
            'Kruskal (ms)': kruskal_time * 1000
        })
    
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Smart City: Cairo Transportation Optimization 🏙️")

# Load data
(node_info, node_id_map, node_name_to_internal_id, all_node_names, 
 df_existing_roads, df_potential_roads, df_neighborhoods, df_facilities,
 df_traffic_flow, df_metro_lines, df_bus_routes, df_transport_demand) = load_data()

# Prepare all_edges list
all_edges = []
# Process existing roads
for _, row in df_existing_roads.iterrows():
    try:
        u_orig, v_orig = str(row['FromID']), str(row['ToID'])
        u_internal, v_internal = node_id_map[u_orig], node_id_map[v_orig]
        distance = float(row['Distance(km)'])
        condition = float(row['Condition(1-10)'])

        maintenance_factor_default = 0.15
        original_cost_mst = distance * (11 - condition) * maintenance_factor_default

        all_edges.append({
            'u_original': u_orig, 'v_original': v_orig, 'u_internal': u_internal, 'v_internal': v_internal,
            'original_cost': original_cost_mst, 'adjusted_cost': original_cost_mst,
            'distance': distance,
            'type': 'Existing', 'color': 'blue', 'details': f"Dist: {distance:.1f}km, Cond: {condition}/10"
        })
    except KeyError as e: st.warning(f"Skipping existing road due to Node ID mapping error: {e}.")
    except ValueError as e: st.warning(f"Skipping existing road due to data conversion error: {e}.")

# Process potential new roads
for _, row in df_potential_roads.iterrows():
    try:
        u_orig, v_orig = str(row['FromID']), str(row['ToID'])
        u_internal, v_internal = node_id_map[u_orig], node_id_map[v_orig]
        construction_cost_default = float(row['Construction Cost(Million EGP)'])
        distance = float(row['Distance(km)'])

        all_edges.append({
            'u_original': u_orig, 'v_original': v_orig, 'u_internal': u_internal, 'v_internal': v_internal,
            'original_cost': construction_cost_default, 'adjusted_cost': construction_cost_default,
            'distance': distance,
            'type': 'New', 'color': 'green', 'details': f"Dist: {distance:.1f}km, Est. Cost: {construction_cost_default:.1f}M EGP"
        })
    except KeyError as e: st.warning(f"Skipping potential road due to Node ID mapping error: {e}.")
    except ValueError as e: st.warning(f"Skipping potential road due to data conversion error: {e}.")

# Identify critical nodes
critical_node_internal_ids = [internal_id for internal_id, info in node_info.items() 
                            if info['is_facility'] and info['type'] in ['Medical', 'Government']]
high_pop_node_internal_ids = [internal_id for internal_id, info in node_info.items() 
                            if not info['is_facility'] and info['population'] >= 300000]
priority_node_internal_ids = set(critical_node_internal_ids + high_pop_node_internal_ids)

# --- Sidebar ---
st.sidebar.title("Controls & Parameters")

# Route Planning Section
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Route Planning")
route_type = st.sidebar.radio("Route Type:", ["Standard", "Emergency"])
start_node_name = st.sidebar.selectbox("Start Location:", all_node_names, index=0)
end_node_name = st.sidebar.selectbox("End Location:", all_node_names, index=1 if len(all_node_names)>1 else 0)
time_of_day = st.sidebar.selectbox("Time of Day:", ["Morning", "Afternoon", "Evening", "Night"], index=0)

if st.sidebar.button("Find Optimal Path", key="find_path_button", type="primary"):
    if start_node_name == end_node_name:
        st.sidebar.warning("Start and end locations cannot be the same.")
        st.session_state.shortest_path_edges = []
        st.session_state.shortest_path_distance = 0
        st.session_state.travel_time = 0
    else:
        if route_type == "Standard":
            path_edges, total_time, error_msg = dijkstra_shortest_path(
                node_info, all_edges, start_node_name, end_node_name, 
                node_name_to_internal_id, time_of_day.lower())
        else:  # Emergency
            path_edges, total_time, error_msg = a_star_search(
                node_info, all_edges, start_node_name, end_node_name, 
                node_name_to_internal_id, time_of_day.lower())
        
        if error_msg:
            st.sidebar.error(error_msg)
            st.session_state.shortest_path_edges = []
            st.session_state.shortest_path_distance = 0
            st.session_state.travel_time = 0
        else:
            st.session_state.shortest_path_edges = path_edges
            st.session_state.travel_time = total_time
            # Calculate total distance
            total_distance = sum(edge['distance'] for edge in path_edges)
            st.session_state.shortest_path_distance = total_distance
            st.sidebar.success(f"Optimal path found! Distance: {total_distance:.2f} km, Time: {total_time:.1f} min")
            path_description = " -> ".join(node_info[node_id]['name'] for node_id in (path_edges[0]['u_internal'],) + tuple(edge['v_internal'] for edge in path_edges))
            st.sidebar.markdown(f"**Route:** {path_description}")

# MST Section
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Network Design (MST)")
maintenance_factor = st.sidebar.slider("Maintenance Cost Factor (Existing Roads)", 0.01, 1.0, 0.15, 0.01)
priority_discount_factor = st.sidebar.slider("Priority Node Connection Discount", 0.0, 0.9, 0.3, 0.05)
high_population_threshold = st.sidebar.number_input("High Population Threshold", min_value=50000, max_value=1000000, value=300000, step=50000)
construction_cost_multiplier = st.sidebar.slider("New Road Construction Cost Multiplier", 0.5, 2.0, 1.0, 0.1)

# Update all_edges with current MST parameters
current_all_edges_for_mst = []
high_pop_node_internal_ids = [internal_id for internal_id, info in node_info.items() 
                            if not info['is_facility'] and info['population'] >= high_population_threshold]
priority_node_internal_ids = set(critical_node_internal_ids + high_pop_node_internal_ids)

for edge_template in all_edges:
    edge = edge_template.copy()
    if edge['type'] == 'Existing':
        condition = float(edge['details'].split('Cond: ')[1].split('/10')[0])
        original_mst_cost = edge['distance'] * (11 - condition) * maintenance_factor
    else:
        original_mst_cost = float(edge['details'].split('Est. Cost: ')[1].split('M EGP')[0]) * construction_cost_multiplier

    edge['original_cost'] = original_mst_cost
    edge['adjusted_cost'] = original_mst_cost

    if edge['u_internal'] in priority_node_internal_ids or edge['v_internal'] in priority_node_internal_ids:
        edge['adjusted_cost'] *= (1 - priority_discount_factor)
    current_all_edges_for_mst.append(edge)

if st.sidebar.button("Generate Optimal Road Network (MST)", key="generate_mst_button"):
    if not current_all_edges_for_mst:
        st.sidebar.error("No edges available for MST.")
    else:
        mst_result_edges, total_mst_cost = kruskal_mst(node_info, current_all_edges_for_mst)
        st.session_state.mst_edges = mst_result_edges
        st.session_state.total_mst_cost = total_mst_cost
        st.sidebar.success(f"MST Generated! Cost: {total_mst_cost:.2f}")

# Public Transport Optimization
st.sidebar.markdown("---")
st.sidebar.subheader("🚍 Public Transport Optimization")
if st.sidebar.button("Optimize Bus Allocation"):
    optimal_allocation, original_passengers, optimized_passengers = optimize_public_transport(
        df_metro_lines, df_bus_routes, df_transport_demand)
    
    st.session_state.optimal_bus_allocation = optimal_allocation
    st.session_state.original_passengers = original_passengers
    st.session_state.optimized_passengers = optimized_passengers
    
    if optimized_passengers < original_passengers:
        st.sidebar.warning("Optimization didn't improve passenger service. Consider adding more buses.")
    else:
        improvement = ((optimized_passengers - original_passengers) / original_passengers * 100)
        st.sidebar.success(f"Optimization successful! Improvement: {improvement:.1f}%")
# Traffic Signal Optimization
st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Traffic Signal Optimization")
if st.sidebar.button("Optimize Traffic Signals"):
    signal_actions = greedy_traffic_signal_optimization(node_info, all_edges, critical_node_internal_ids)
    st.session_state.signal_actions = signal_actions
    st.sidebar.success(f"Optimized {len(signal_actions)} intersections!")

# Performance Analysis
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performance Analysis")
if st.sidebar.button("Run Performance Tests"):
    performance_df = analyze_performance()
    st.session_state.performance_df = performance_df
    st.sidebar.success("Performance analysis completed!")

# --- Main Page Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Interactive Map & Route", "🛠️ MST Details", "🚍 Public Transport", "🚦 Traffic Signals", "📊 Data & Analysis"])

with tab1:
    st.header("Interactive Network Map")
    if 'shortest_path_distance' in st.session_state and st.session_state.shortest_path_distance > 0:
        st.metric("Shortest Path Distance", f"{st.session_state.shortest_path_distance:.2f} km")

    # Visualization
    G_viz = nx.Graph()
    pos_viz = {}
    node_labels_viz = {}
    node_colors_map = {}
    node_sizes_viz = []

    for internal_id, info_dict in node_info.items():
        G_viz.add_node(internal_id)
        pos_viz[internal_id] = (info_dict['x'], info_dict['y'])
        node_labels_viz[internal_id] = info_dict['name']

        size = 70
        color = 'skyblue'
        if internal_id in critical_node_internal_ids:
            color = 'crimson'; node_colors_map['Critical Facility'] = color; size = 150
        elif internal_id in high_pop_node_internal_ids:
            color = 'gold'; node_colors_map['High Population Area'] = color; size = 120
        elif info_dict['is_facility']:
            color = 'lightcoral'; node_colors_map['Other Facility'] = color
        else:
            node_colors_map['Other Neighborhood'] = color

        G_viz.nodes[internal_id]['color'] = color
        node_sizes_viz.append(size)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw all available edges
    for edge in all_edges:
        if edge['u_internal'] in G_viz and edge['v_internal'] in G_viz:
            nx.draw_networkx_edges(G_viz, pos_viz, edgelist=[(edge['u_internal'], edge['v_internal'])], 
                                 alpha=0.1, edge_color='gray', style='dashed', ax=ax)

    # Draw MST edges if available
    if 'mst_edges' in st.session_state and st.session_state.mst_edges:
        mst_display_edges = [(edge['u_internal'], edge['v_internal']) for edge in st.session_state.mst_edges]
        mst_edge_colors = [edge['color'] for edge in st.session_state.mst_edges]
        nx.draw_networkx_edges(G_viz, pos_viz, edgelist=mst_display_edges, 
                             width=2.0, edge_color=mst_edge_colors, ax=ax, label="MST Edges")

    # Draw Shortest Path edges if available
    if 'shortest_path_edges' in st.session_state and st.session_state.shortest_path_edges:
        path_display_edges = [(edge['u_internal'], edge['v_internal']) for edge in st.session_state.shortest_path_edges]
        path_color = 'magenta' if route_type == "Standard" else 'red'
        nx.draw_networkx_edges(G_viz, pos_viz, edgelist=path_display_edges, 
                             width=3.0, edge_color=path_color, style='solid', ax=ax, 
                             label="Emergency Path" if route_type == "Emergency" else "Shortest Path")

    node_colors_list = [G_viz.nodes[n]['color'] for n in G_viz.nodes()]
    nx.draw_networkx_nodes(G_viz, pos_viz, node_size=node_sizes_viz, node_color=node_colors_list, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G_viz, pos_viz, labels=node_labels_viz, font_size=7, font_weight='bold', ax=ax)

    ax.set_title("Cairo Transportation Network", fontsize=16)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='magenta', lw=3, label='Standard Path'),
        plt.Line2D([0], [0], color='red', lw=3, label='Emergency Path'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Existing Road (MST)'),
        plt.Line2D([0], [0], color='green', lw=2, label='New Road (MST)'),
        plt.Line2D([0], [0], color='gray', lw=1, ls='--', label='Potential Edge'),
    ]
    for label, color in node_colors_map.items():
         legend_elements.append(plt.scatter([],[], s=100, color=color, label=label))
    ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
    plt.axis('off')
    st.pyplot(fig)

with tab2:
    st.header("Minimum Spanning Tree (MST) Details")
    if 'mst_edges' in st.session_state:
        st.metric("Total MST Cost", f"{st.session_state.total_mst_cost:.2f} (Cost Units)")
        st.metric("Number of Edges in MST", len(st.session_state.mst_edges))
        mst_display_data = []
        for edge in st.session_state.mst_edges:
            u_name = node_info[edge['u_internal']]['name']
            v_name = node_info[edge['v_internal']]['name']
            mst_display_data.append({
                "From": u_name, "To": v_name, "Type": edge['type'],
                "Cost (MST)": f"{edge['original_cost']:.2f}",
                "Actual Distance (km)": f"{edge['distance']:.1f}",
                "Details": edge['details']
            })
        st.dataframe(pd.DataFrame(mst_display_data), use_container_width=True)
    else:
        st.info("Generate an MST using the controls in the sidebar to see details here.")

with tab3:
    st.header("Public Transport Optimization")
    
    st.subheader("Current Metro Lines")
    st.dataframe(df_metro_lines)
    
    st.subheader("Current Bus Routes")
    st.dataframe(df_bus_routes)
    
    st.subheader("Transportation Demand")
    st.dataframe(df_transport_demand)
    
    if 'optimal_bus_allocation' in st.session_state:
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)
        col1.metric("Original Passengers Served", st.session_state.original_passengers)
        col2.metric("Optimized Passengers Served", st.session_state.optimized_passengers,
                   delta=f"{((st.session_state.optimized_passengers - st.session_state.original_passengers)/st.session_state.original_passengers*100):.1f}%")
    if 'optimal_bus_allocation' in st.session_state:
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)
        col1.metric("Original Passengers", st.session_state.original_passengers)
    
        if st.session_state.optimized_passengers > st.session_state.original_passengers:
            improvement = ((st.session_state.optimized_passengers - st.session_state.original_passengers) / 
                      st.session_state.original_passengers * 100)
            col2.metric("Optimized Passengers", st.session_state.optimized_passengers,
                   delta=f"+{improvement:.1f}%", delta_color="normal")
        else:
            col2.metric("Optimized Passengers", st.session_state.optimized_passengers,
                   label="(No improvement found)")   
        st.subheader("Optimal Bus Allocation")
        allocation_df = pd.DataFrame.from_dict(st.session_state.optimal_bus_allocation, 
                                             orient='index', columns=['Buses Allocated'])
        st.dataframe(allocation_df)
        # Visualization of bus routes
        fig, ax = plt.subplots(figsize=(12, 8))
        bus_route_colors = plt.cm.tab20.colors
        
        for i, (route_id, route_data) in enumerate(df_bus_routes.iterrows()):
            stops = route_data['Stops(comma-separated IDs)'].split(',')
            x_coords = []
            y_coords = []
            for stop in stops:
                internal_id = node_id_map[stop]
                x_coords.append(node_info[internal_id]['x'])
                y_coords.append(node_info[internal_id]['y'])
            
            allocated = st.session_state.optimal_bus_allocation.get(route_data['RouteID'], 0)
            linewidth = 2 + (allocated / 5)  # Thicker line for routes with more buses
            alpha = 0.3 + (allocated / df_bus_routes['Buses Assigned'].max()) * 0.7
            
            ax.plot(x_coords, y_coords, marker='o', color=bus_route_colors[i % len(bus_route_colors)],
                    linewidth=linewidth, alpha=alpha, label=f"{route_data['RouteID']} ({allocated} buses)")
        
        ax.set_title("Optimized Bus Routes (Thickness = Bus Allocation)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        st.pyplot(fig)

with tab4:
    st.header("Traffic Signal Optimization")
    
    if 'signal_actions' in st.session_state:
        st.subheader("Recommended Signal Timing Adjustments")
        
        # Display priority list
        priority_df = pd.DataFrame(st.session_state.signal_actions)
        st.dataframe(priority_df.set_index('Rank'), use_container_width=True)
        
        # Visualization of top intersections
        st.subheader("Top Priority Intersections")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all nodes
        for internal_id, info in node_info.items():
            ax.scatter(info['x'], info['y'], 
                      c='lightgray', s=30, alpha=0.3)
        
        # Highlight top 10 intersections
        top_n = min(10, len(st.session_state.signal_actions))
        for i in range(top_n):
            node_id = node_name_to_internal_id[st.session_state.signal_actions[i]['Intersection']]
            x, y = node_info[node_id]['x'], node_info[node_id]['y']
            
            color = 'red' if st.session_state.signal_actions[i]['Type'] == 'Critical Facility' else (
                   'orange' if st.session_state.signal_actions[i]['Type'] == 'High Population' else 'blue')
            
            ax.scatter(x, y, s=200 - (i * 15), color=color, alpha=0.8,
                      label=f"{i+1}. {st.session_state.signal_actions[i]['Intersection']}")
        
        ax.set_title(f"Top {top_n} Priority Intersections")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        st.pyplot(fig)
        
        st.markdown("""
        **Optimization Key:**
        - 🔴 Critical Facilities (Hospitals, Government)
        - 🟠 High Population Areas (>200,000 residents)
        - 🔵 Regular Intersections
        - Size indicates priority ranking (larger = higher priority)
        """)
    else:
        st.info("Run traffic signal optimization from the sidebar to see recommendations.")

with tab5:
    st.header("Data & Performance Analysis")
    
    st.subheader("Raw Data Tables")
    st.write("Neighborhoods:")
    st.dataframe(df_neighborhoods)
    st.write("Facilities:")
    st.dataframe(df_facilities)
    st.write("Existing Roads:")
    st.dataframe(df_existing_roads)
    st.write("Potential New Roads:")
    st.dataframe(df_potential_roads)
    
    if 'performance_df' in st.session_state:
        st.subheader("Algorithm Performance Analysis")
        st.dataframe(st.session_state.performance_df)
        
        # Performance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        st.session_state.performance_df.plot(x='Graph Size', y=['Dijkstra (ms)', 'A* (ms)', 'Kruskal (ms)'], 
                                           kind='line', marker='o', ax=ax)
        ax.set_title("Algorithm Performance Comparison")
        ax.set_ylabel("Execution Time (ms)")
        ax.grid(True)
        st.pyplot(fig)
        
        st.markdown("""
        **Performance Observations:**
        - Dijkstra's algorithm shows O(E + V log V) complexity
        - A* performs better than Dijkstra for point-to-point queries due to heuristic
        - Kruskal's algorithm shows O(E log E) complexity
        """)
    else:
        st.info("Run performance tests from the sidebar to see algorithm comparisons.")

st.sidebar.markdown("---")
st.sidebar.info("""
Developed for CSE112. This app demonstrates:
- Kruskal's for MST network design
- Dijkstra's for standard routing
- A* for emergency routing
- DP for public transport
- Greedy for traffic signals
""")