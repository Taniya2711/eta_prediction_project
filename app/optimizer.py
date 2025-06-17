import pandas as pd
import numpy as np
import pickle
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def haversine_distance(loc1, loc2):
    return geodesic(loc1, loc2).km

def compute_distance_matrix(locations):
    matrix = []
    for a in locations:
        row = []
        for b in locations:
            km = haversine_distance(a, b)
            minutes = int((km / 30) * 60)  # Assuming 30km/h
            row.append(minutes)
        matrix.append(row)
    return matrix

def optimize_route(input_df, model_type="lgbm", buffer=25):
    df = input_df.copy()

    # Get coordinates and distance matrix
    coords = list(zip(df['delivery_lat'], df['delivery_lon']))
    distance_matrix = compute_distance_matrix(coords)

    # Time windows from predicted ETAs
    if model_type == "lgbm":
        df['tw_start'] = (df['eta_50'] - buffer).clip(lower=0).astype(int)
        df['tw_end'] = (df['eta_50'] + buffer).astype(int)
    elif model_type == "ngb":
        df['tw_start'] = (df['eta_mean'] - 1.28 * df['eta_std'] - buffer).clip(lower=0).astype(int)
        df['tw_end'] = (df['eta_mean'] + 1.28 * df['eta_std'] + buffer).astype(int)
    else:
        raise ValueError("model_type must be 'lgbm' or 'ngb'")

    time_windows = list(zip(df['tw_start'], df['tw_end']))

    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index,
        30, 1000, False, "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for i, (start, end) in enumerate(time_windows):
        index = manager.NodeToIndex(i)
        time_dimension.CumulVar(index).SetRange(start, end)

    # Solve
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)

    if not solution:
        return {
            "success": False,
            "message": "❌ No feasible route found. Try increasing buffer or reducing stops.",
            "route": [],
            "summary": pd.DataFrame()
        }

    # Extract route
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    # ETA summary
    summary = []
    for stop in route:
        if stop == 0:
            continue
        if model_type == "lgbm":
            eta = df.loc[stop, 'eta_50']
            low = df.loc[stop, 'eta_10']
            high = df.loc[stop, 'eta_90']
            risk = (high - low) / 2
        else:
            eta = df.loc[stop, 'eta_mean']
            std = df.loc[stop, 'eta_std']
            risk = std * 1.28

        risk_level = "Low" if risk < 5 else "Medium" if risk < 10 else "High"

        summary.append({
            "Stop": stop,
            "ETA": f"{eta:.1f} min",
            "Risk ±": f"{risk:.1f} min",
            "Risk Level": risk_level
        })

    return {
        "success": True,
        "message": "✅ Feasible route computed.",
        "route": route,
        "summary": pd.DataFrame(summary)
    }
