#########################################################
#                                                       #
# Created on: 22/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 28/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

# Import necessary libraries
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Candidate_Ranking.ranking_methods import CandidateRanking
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np


class PrepareInput:
    """
    Class to prepare input data for different types of routing and clustering algorithms.
    """

    def __init__(self):
        """
        Initializes the PrepareInput class without any attributes.
        """
        pass

    def prep_greedy(self, input_df_with_depot, chosen_company, chosen_candidates, method,
                    distance_matrix_ranking, truck_capacity, greedy_ranking):
        """
        Prepares input for the greedy algorithm by calculating distances and organizing stops.

        Args:
            input_df_with_depot (pd.DataFrame): DataFrame including the depot.
            chosen_company (str): The company selected for the routing.
            chosen_candidates (list): List of candidate locations.
            method (str): Method used to calculate distances.
            distance_matrix_ranking (pd.DataFrame): Precomputed distance matrix.
            truck_capacity (int): Capacity of the vehicle.
            greedy_ranking (pd.DataFrame): DataFrame with ranking for greedy algorithm.

        Returns:
            tuple: A tuple containing the results DataFrame and the distance matrix.
        """
        results = []
        for chosen_candidate in chosen_candidates:
            distance_calc = RoadDistanceCalculator()
            total_greedy_dist = greedy_ranking.loc[chosen_candidate].iloc[0]
            distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                input_df_with_depot, chosen_company=chosen_company, candidate_name=chosen_candidate,
                method=method, computed_distances_df=distance_matrix_ranking
            )
            route_sequence = distance_matrix_vrp.index.tolist()
            total_route_distance = sum(
                distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
                for i in range(len(route_sequence) - 1)
            )
            num_stops = len(distance_matrix_vrp.columns) - 1
            depot_distances = distance_matrix_vrp.loc['Depot'].drop('Depot')
            min_distance = depot_distances.min()
            max_distance = depot_distances.max()
            results.append({
                "totalsum": total_route_distance,
                "number_stops": num_stops,
                "Max_to_depot": max_distance,
                "Min_to_depot": min_distance,
                "vehicle_cap": truck_capacity,
                "greedy_total_sum": total_greedy_dist
            })
        results_df = pd.DataFrame(results)
        return results_df, distance_matrix_vrp

    def prep_bounding_circle(self, df_input, df_input_modified, sliced_distance_matrix_ranking,
                             max_truck_cap, chosen_company, chosen_candidates, method, bounding_circle_ranking):
        """
        Prepares input for the bounding circle algorithm by analyzing geographical data and clustering.

        Args:
            df_input (pd.DataFrame): Original input DataFrame.
            df_input_modified (pd.DataFrame): Modified version of the input DataFrame for analysis.
            sliced_distance_matrix_ranking (pd.DataFrame): Sliced distance matrix for selected candidates.
            max_truck_cap (int): Maximum capacity of the truck.
            chosen_company (str): The company selected for the analysis.
            chosen_candidates (list): List of candidate locations.
            method (str): Method used for distance calculations.
            bounding_circle_ranking (pd.DataFrame): DataFrame with ranking based on bounding circle metrics.

        Returns:
            tuple: A tuple containing the results DataFrame and the distance matrix.
        """
        results = []
        ranking = CandidateRanking()
        for chosen_candidate in chosen_candidates:
            company_df = df_input[df_input['name'] == chosen_company]
            middle_point_circle = (company_df['lat'].mean(), company_df['lon'].mean())
            distances = company_df.apply(
                lambda row: ranking._euclidean_distance(middle_point_circle[0], middle_point_circle[1],
                                                        row['lat'], row['lon']), axis=1)
            radius = distances.mean()
            average_deliveries_in_circle = bounding_circle_ranking[['Percentage Overlap']].mean()
            mean_distance_within_circle = distances[distances <= radius].mean()
            area_of_circle = np.pi * (radius ** 2)
            delivery_density = len(distances[distances <= radius]) / area_of_circle
            mean_distance_to_midpoint = distances.mean()
            distance_calc = RoadDistanceCalculator()
            distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                df_input_modified, chosen_company=chosen_company, candidate_name=chosen_candidate,
                method=method, computed_distances_df=sliced_distance_matrix_ranking
            )
            route_sequence = distance_matrix_vrp.index.tolist()
            total_route_distance = sum(
                distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
                for i in range(len(route_sequence) - 1)
            )
            num_stops = len(distance_matrix_vrp.columns) - 1
            depot_distances = distance_matrix_vrp.loc['Depot'].copy().drop('Depot')
            min_distance = depot_distances.min()
            max_distance = depot_distances.max()
            results.append({
                "totalsum": total_route_distance,
                "number_stops": num_stops,
                "Max_to_depot": max_distance,
                "Min_to_depot": min_distance,
                "vehicle_cap": max_truck_cap,
                "average_percentage_in_circle": average_deliveries_in_circle.values[0],
                "mean_distance_within_circle": mean_distance_within_circle,
                "delivery_density": delivery_density,
                "radius": mean_distance_to_midpoint
            })
        results_df = pd.DataFrame(results)
        return results_df, distance_matrix_vrp

    def prep_k_means(self, df_input, df_input_modified, sliced_distance_matrix_ranking, squared_matrix, max_truck_cap,
                     chosen_company, chosen_candidates, method, k_means_ranking):
        """
        Prepares input for the k-means clustering algorithm by analyzing geographical data and optimizing cluster sizes.

        Args:
            df_input (pd.DataFrame): Original input DataFrame.
            df_input_modified (pd.DataFrame): Modified version of the input DataFrame for analysis.
            sliced_distance_matrix_ranking (pd.DataFrame): Sliced distance matrix for selected candidates.
            squared_matrix (pd.DataFrame): Squared distance matrix used for clustering.
            max_truck_cap (int): Maximum capacity of the truck.
            chosen_company (str): The company selected for the analysis.
            chosen_candidates (list): List of candidate locations.
            method (str): Method used for distance calculations.
            k_means_ranking (pd.DataFrame): DataFrame with ranking based on k-means metrics.

        Returns:
            tuple: A tuple containing the results DataFrame and the distance matrix.
        """
        results = []
        ranking = CandidateRanking()
        for chosen_candidate in chosen_candidates:
            cleaned_matrix = ranking.clean_column_and_index_labels(squared_matrix)

            # Prepare calculation
            minimal_clusters = len(df_input[df_input['name'] == chosen_company])
            scaler = StandardScaler()
            df_input[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_input[['lat', 'lon']])

            # Determine number of clusters and compute silhouette scores
            silhouette_scores_kmeans = {}
            b_s = {}
            a_s = {}
            for i in range(minimal_clusters, minimal_clusters + 6):
                kmeans = KMeans(n_clusters=i)
                df_new = df_input_modified.copy()
                df_new['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
                b_s[i] = ranking._average_distance_between_clusters(df_new, squared_matrix)
                silhouette_scores_kmeans[i], a_s[i] = ranking._mean_intra_cluster_distance(df_new, squared_matrix,
                                                                                           b_s[i], i)

            # Determine optimal clusters
            optimal_clusters = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get)
            kmeans = KMeans(n_clusters=optimal_clusters)
            df_input['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
            average_overlap = k_means_ranking[['Percentage Overlap']].mean()

            # Compute metrics for clusters
            average_cluster_size = df_input['cluster'].value_counts().mean()
            centroid_distances = []
            for cluster in df_input['cluster'].unique():
                cluster_points = df_input[df_input['cluster'] == cluster][['lat_scaled', 'lon_scaled']]
                centroid = kmeans.cluster_centers_[cluster]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                centroid_distances.extend(distances)
            average_centroid_distance = np.mean(centroid_distances)

            # Compute minimal and maximal distances within clusters
            min_distances = []
            max_distances = []
            for cluster in df_input['cluster'].unique():
                cluster_points = df_input[df_input['cluster'] == cluster][['lat_scaled', 'lon_scaled']]
                if len(cluster_points) > 1:
                    pairwise_dist = pdist(cluster_points)
                    min_distances.append(np.min(pairwise_dist))
                    max_distances.append(np.max(pairwise_dist))

            average_minimal_distance = np.mean(min_distances) if min_distances else 0
            average_maximal_distance = np.mean(max_distances) if max_distances else 0

            distance_calc = RoadDistanceCalculator()
            distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                df_input_modified, chosen_company=chosen_company, candidate_name=chosen_candidate,
                method=method, computed_distances_df=sliced_distance_matrix_ranking
            )
            route_sequence = distance_matrix_vrp.index.tolist()
            total_route_distance = sum(
                distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
                for i in range(len(route_sequence) - 1)
            )
            num_stops = len(distance_matrix_vrp.columns) - 1
            depot_distances = distance_matrix_vrp.loc['Depot'].drop('Depot')
            min_distance = depot_distances.min()
            max_distance = depot_distances.max()

            # Append results to list
            results.append({
                "totalsum": total_route_distance,
                "number_stops": num_stops,
                "Max_to_depot": max_distance,
                "Min_to_depot": min_distance,
                "average_overlap_percentage": average_overlap.values[0],
                "average_cluster_size": average_cluster_size,
                "average_centroid_distance": average_centroid_distance,
                "average_distance_within_clusters": a_s[optimal_clusters],
                "average_distance_between_clusters": b_s[optimal_clusters],
                "average_minimal_distance": average_minimal_distance,
                "average_maximal_distance": average_maximal_distance,
                "vehicle_cap": max_truck_cap,
            })

        # Convert results list to DataFrame
        results_df = pd.DataFrame(results)

        return results_df, distance_matrix_vrp