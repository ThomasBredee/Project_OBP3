#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester                          #
#                                                       #
# Updated on: 26/01/2025                                #
# Updated by: Lukas and Dennis and Wester               #
#                                                       #
#########################################################

#Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import euclidean_distances

class CandidateRanking:
    def _init_(self):
        pass

    #Remove everything after '_' in column names and index
    def clean_column_and_index_labels(self, matrix):
        matrix_copy = matrix.copy()  #Create a copy to avoid modifying the original
        matrix_copy.columns = [col.split('_')[0] for col in matrix_copy.columns]
        matrix_copy.index = [idx.split('_')[0] for idx in matrix_copy.index]
        return matrix_copy

    #Calculate Euclidean distance between two points
    def _euclidean_distance(self, lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    #First algorithm to solve ranking
    def greedy(self, partial_dist_matrix, chosen_company):
        #Clean column names and index
        cleaned_matrix = self.clean_column_and_index_labels(partial_dist_matrix)

        #Sums the total distances of each potential partner
        grouped_df = cleaned_matrix.T.groupby(cleaned_matrix.columns).sum().T

        #Create ranked df
        ranking = grouped_df.sum(axis=0).sort_values(ascending=True).to_frame()
        ranking.columns = ['Total Distance']

        return ranking[ranking.index!=chosen_company]

    #Second algorithm to solve ranking (based on bounding box idea)
    def bounding_circle(self, df_input, partial_dist_matrix, chosen_company):

        #Get correct format for matrix
        cleaned_matrix = self.clean_column_and_index_labels(partial_dist_matrix)

        #Get sliced dfs for partner and company
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        company_df = df_input[df_input['name'] == chosen_company]
        partner_df = df_input[df_input['name'] != chosen_company]

        #Get middel point circle
        middel_point_circle = (company_df['lat'].mean(), company_df['lon'].mean())
        distances = company_df.apply(lambda row: self._euclidean_distance(middel_point_circle[0], middel_point_circle[1], row['lat'], row['lon']), axis=1)
        radius = distances.mean()

        #Creating ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names)

        #Calculate overlap percentages
        for partner in partner_names:
            df_temp = partner_df[partner_df['name'] == partner]
            count = 0
            for _, row in df_temp.iterrows():
                distance = self._euclidean_distance(middel_point_circle[0], middel_point_circle[1], row['lat'],row['lon'])
                if distance <= radius:
                    count += 1
            ranking_df.loc[partner, 'Percentage Overlap'] = count / len(df_temp)

        return ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

    #Third ranking algorithm
    def k_means(self, df_input, df_input_modified, full_dist_matrix, chosen_company):

        #Get correct format for matrix
        cleaned_matrix = self.clean_column_and_index_labels(full_dist_matrix)

        #Prep calculation
        minimal_clusters = len(df_input[df_input['name'] == chosen_company])
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        scaler = StandardScaler()
        df_input[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_input[['lat', 'lon']])

        #Determine number of clusters
        silhouette_scores_kmeans = {}
        b_s = {}
        a_s = {}
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i, random_state=42)
            df_new = df_input_modified.copy()
            df_new['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
            b_s[i] = self._average_distance_between_clusters(df_new, full_dist_matrix)
            silhouette_scores_kmeans[i], a_s[i] = self._mean_intra_cluster_distance(df_new, full_dist_matrix, b_s[i], i)

        #Determine optimal clusters
        optimal_clusters = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get) + minimal_clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df_input['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
        clusters_company = df_input[df_input['name'] == chosen_company]['cluster'].unique()

        #Creating ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names)

        #Get percentage overlap of clusters
        for partner in partner_names:
            partner_df = df_input[df_input['name'] == partner]
            partner_data = partner_df[partner_df['name'] == partner]
            is_in_cluster = partner_data['cluster'].isin(clusters_company)

            #Calculate and assign percentage overlap directly to ranking_df
            ranking_df.loc[partner, 'Percentage Overlap'] = is_in_cluster.mean()

        return ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

    #fourth ranking algorithm
    def dbscan(self, input_df, full_matrix, chosen_company, eps, ms):
        df_copy = input_df.copy()

        # Make DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=ms, metric='precomputed')

        # Assign clusters
        clusters = dbscan.fit_predict(full_matrix)
        df_copy['cluster'] = clusters

        # Handle noise
        df_noise_assign = self._assign_noise_points(df_copy, chosen_company, True)
        df_filter_noise = df_noise_assign[df_noise_assign['cluster'] != -1]

        # Get percentage results
        percentages = self._get_percentages(df_filter_noise, chosen_company)
        percentages.index = percentages["name"]
        percentages = percentages.drop(columns=["name"])
        percentages.index.name = None

        return percentages

    def _mean_intra_cluster_distance(self,df_input_modified, distance_matrix, average_distance_between_clusters, k):
        #Obtain correct df format
        grouped = df_input_modified.groupby('cluster')
        silhouette_scores = np.zeros(k)
        #Calculate the distance within clusters
        for cluster, group in grouped:
            #Obtain each cluster
            names = df_input_modified.loc[group.index, 'name'].tolist()
            sub_matrix = distance_matrix.loc[names, names]
            #Because of symmetry in distance matrix use upper right triangle
            mask = np.triu(np.ones(sub_matrix.shape, dtype=bool), k=1)
            upper_triangle_values = sub_matrix.values[mask]
            mean_distance_in_cluster = upper_triangle_values.mean() if len(names) > 1 else 0
            silhouette_scores[cluster]= (average_distance_between_clusters-mean_distance_in_cluster)/max(average_distance_between_clusters,mean_distance_in_cluster)
        return np.mean(silhouette_scores), mean_distance_in_cluster

    def _average_distance_between_clusters(self,df_input_modified , distance_matrix):
        #Obtain right df format
        grouped = df_input_modified.groupby('cluster')
        cluster_distances = {}
        cluster_names = [cluster for cluster, _ in grouped]
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names):
                #Obtain different clusters and calculate the average distance between them
                    names1 = df_input_modified.loc[grouped.get_group(cluster1).index, 'name'].tolist()
                    names2 = df_input_modified.loc[grouped.get_group(cluster2).index, 'name'].tolist()
                    if names1 != names2:
                        sub_matrix = distance_matrix.loc[names1, names2]
                        mask = np.triu(np.ones(sub_matrix.shape, dtype=bool), k=1)
                        upper_triangle_values = sub_matrix.values[mask]
                        mean_distance_between_clusters = upper_triangle_values.mean() if len(names1) & len(names2) > 1 else 0
                        cluster_distances[(cluster1, cluster2)] = mean_distance_between_clusters
                    else:
                        continue

        return np.mean(list(cluster_distances.values()))

    def _assign_noise_points(self, df, chosen_company, company):
        # Decide if all noise points are assigned or only company noise
        if company is True:
            noise_df = df[(df['name'] == chosen_company) & (df['cluster'] == -1)].reset_index(drop=True)
        else:
            noise_df = df[df['cluster'] == -1].reset_index(drop=True)

        if noise_df.empty:
            return df

        # Get all unique clusters
        clusters = df['cluster'].unique()
        clusters = sorted(clusters)
        centroids = []

        # Get centroids of all clusters
        for cluster in clusters:
            if cluster != -1:  # Ignore noise
                cluster_points = df[df['cluster'] == cluster]
                centroid = (cluster_points['lat'].mean(), cluster_points['lon'].mean())
                centroids.append(centroid)
        centroids = np.array(centroids)

        # Locate nearest cluster for noise points
        for i in range(len(noise_df)):
            point = (noise_df['lat'][i], noise_df['lon'][i])
            distances = euclidean_distances([point], centroids)
            nearest_cluster = np.argmin(distances)
            noise_df.loc[i, 'cluster'] = clusters[nearest_cluster]

        # Assign new clusters to df
        for idx, row in noise_df.iterrows():
            # Match rows in df based on name, lat, and lon
            mask = (
                    (df['name'] == row['name']) &
                    (df['lat'] == row['lat']) &
                    (df['lon'] == row['lon'])
            )
            df.loc[mask, 'cluster'] = row['cluster']

        return df

    def _get_percentages(self, df_new, chosen_company):
        # Get unique clusters of the chosen company
        unique_clusters = df_new.loc[df_new['name'] == chosen_company, 'cluster'].unique()
        filtered_df = df_new[df_new['cluster'].isin(unique_clusters)]

        matching_counts = filtered_df.groupby('name').size()  # Amount matching
        total_counts = df_new.groupby('name').size()  # Total amount
        percentages = (matching_counts / total_counts).reset_index()  # Amount matching / total amount

        # Clean percentage output
        percentages.rename(columns={0: 'Percentage'}, inplace=True)  # Rename
        percentages = percentages.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)  # Sort
        percentages['Percentage'] = percentages['Percentage'].fillna(0)
        percentages = percentages[percentages['name'] != chosen_company]  # Drop chosen company
        return percentages
