#########################################################
#                                                       #
# Created on: 14/01/2025                                #
# Created by: Frank                                     #
#                                                       #
# Updated on: 23/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

#Import
import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxRuntime
import matplotlib.pyplot as plt
import math

class VRPSolver:
    #Create model to solve VRP
    def build_model(self, input_df, chosen_company, truck_capacity, distance_matrix, chosen_candidate = None):
        COORDS = []
        current_names = []
        demands = []

        for name, lat, lon in zip(input_df.name, input_df.lat, input_df.lon):
            if chosen_candidate:  #Collaborative VRP
                if chosen_company in name or chosen_candidate in name or name == "Depot":
                    COORDS.append((lat, lon))
                    current_names.append(name)
                    demands.append(1)  #We assume each location has demand of 1
            else:  #Single Company VRP
                if chosen_company in name or name == "Depot":
                    COORDS.append((lat, lon))
                    current_names.append(name)
                    demands.append(1)  #We assume each location has demand of 1

        total_demand = sum(demands)  #Amount of locations
        num_vehicles = max(1, math.ceil(total_demand / truck_capacity))  #Amount of needed trucks (ceil)

        #Setup model
        model = Model()
        model.add_vehicle_type(num_vehicles, capacity=truck_capacity)
        depot = model.add_depot(x=COORDS[0][0], y=COORDS[0][1], name=current_names[0])
        clients = [model.add_client(x=COORDS[idx][0], y=COORDS[idx][1], name=current_names[idx], delivery=demands[idx]) for idx in range(1, len(COORDS))]
        locations = [depot] + clients

        added_edges = set()
        for frm in locations:
            for to in locations:
                if frm.name != to.name:
                    edge = (frm.name, to.name)

                    #Check if the edge (or its reverse) has already been added
                    if edge not in added_edges and (to.name, frm.name) not in added_edges:
                        #Get the distance from matrix
                        distance = distance_matrix.loc[frm.name, to.name]
                        if isinstance(distance, pd.Series):
                            distance = distance.iloc[0]
                        #Add the edge to the set
                        added_edges.add(edge)
                        added_edges.add((to.name, frm.name))

                        #Add the edge
                        model.add_edge(frm, to, distance=distance)
                        model.add_edge(to, frm, distance=distance)

        return model, current_names

    def solve(self, model, max_runtime, display, current_names, print_routes=False):

        #Solve VRP
        res = model.solve(stop=MaxRuntime(max_runtime), display=display)

        solution = res.best
        routes = []
        for vehicle_route in solution.routes():
            route = [current_names[i] for i in vehicle_route.visits()]
            route.insert(0, current_names[0])  #Adding the depot at the start
            route.append(current_names[0])  #Adding the depot at the end
            routes.append(route)

        if print_routes:
            for idx, route in enumerate(routes):
                print(f"Route for Vehicle {idx + 1}: {route}")

        return solution, routes

    def calculate_distance_per_order(self, routes, distance_matrix):
        total_distance = 0
        total_orders = 0
        for route in routes:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += distance_matrix.loc[route[i], route[i + 1]]
            total_distance += route_distance
            total_orders += len(route) - 2  #Exclude depot at start and end

        distance_per_order = total_distance / total_orders if total_orders > 0 else 0
        print(f"Total Distance: {total_distance}, Total Orders: {total_orders}, Distance per Order: {distance_per_order}")
        return total_distance, distance_per_order

    def plotRoute(self, routes, input_df):
        plt.figure(figsize=(12, 8))

        #Plot each route
        for idx, route in enumerate(routes):
            route_df = input_df[input_df['name'].isin(route)]
            depot = route_df[route_df['name'] == 'Depot']
            other_locations = route_df[route_df['name'] != 'Depot']

            #Plot the depot
            plt.scatter(depot['lon'], depot['lat'], color='red', label=f'Depot', alpha=1, marker='D', s=100)

            #Plot other locations
            plt.scatter(other_locations['lon'], other_locations['lat'], label=f'Route {idx + 1}', alpha=0.6, marker='o')

            #Plot the route
            for i in range(1, len(route)):
                start = route_df[route_df.name == route[i - 1]].iloc[0]
                end = route_df[route_df.name == route[i]].iloc[0]

                plt.arrow(start['lon'], start['lat'], end['lon'] - start['lon'], end['lat'] - start['lat'],
                          head_width=0.025, head_length=0.025, fc='green', ec='green', alpha=0.6)

        #Labels and Title
        plt.title('Optimal Routes with Locations and Depot', fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()

