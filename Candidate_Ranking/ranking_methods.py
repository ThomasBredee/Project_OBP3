#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester and Dennis               #
#                                                       #
# Updated on: 30/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

#Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import euclidean_distances
from Machine_Learning_Predict.predictor import ModelPredictor

class CandidateRanking:
    def _init_(self):
        pass

    #First algorithm
    def greedy(self, partial_dist_matrix, chosen_company):
        """
        Do a greedy selection to find the potential partner with the least total distance,
        excluding the chosen company

        input:
        partial_dist_matrix (DF): A DF containing distances between companies,
                                         where all companies are in the columns and on the rows only the chosen comapny
        chosen_company (str): The name of the company to exclude from the final ranking

        output:
        DF: A DF with companies ranked by their total distance to the chosen
                   company, without the chosen company
        """

        #Clean column names and index of the input matrix
        cleaned_matrix = self.clean_column_and_index_labels(partial_dist_matrix)

        #Get distances and rank them
        grouped_df = cleaned_matrix.T.groupby(cleaned_matrix.columns).sum().T
        ranking = grouped_df.sum(axis=0).sort_values(ascending=True).to_frame()
        ranking.columns = ['Total Distance']

        return ranking[ranking.index != chosen_company]

    # Second algorithm to solve ranking (based on bounding box idea)
    def bounding_circle(self, df_input, partial_dist_matrix, chosen_company):
        """
        Calculates a ranking based on how many partners fall within a bounding circle
        defined by the average location of the chosen company

        input:
        df_input (DF): A DF with columns for 'name', 'lat', and 'lon' of companies
        partial_dist_matrix (DF): A DF containing distances between companies,
                                         where companies are both in rows and columns
        chosen_company (str): The name of the company to base the bounding circle on

        output:
        DataFrame: A DF with companies ranked by the percentage of their locations that
                   fall within the bounding circle around the chosen company, without including
                   the chosen company itself
        """

        #Clean column names and index of the input matrix
        cleaned_matrix = self.clean_column_and_index_labels(partial_dist_matrix)

        #Identify unique partner names, excluding the chosen company and get dfs
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        company_df = df_input[df_input['name'] == chosen_company]
        partner_df = df_input[df_input['name'] != chosen_company]

        #Calculate the middle point of the bounding circle using the chosen company's locations and get radius
        middle_point_circle = (company_df['lat'].mean(), company_df['lon'].mean())
        distances = company_df.apply(
            lambda row: self._euclidean_distance(middle_point_circle[0], middle_point_circle[1], row['lat'],row['lon']), axis=1)
        radius = distances.mean()

        #Get ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names, dtype='float')
        for partner in partner_names:
            df_temp = partner_df[partner_df['name'] == partner]
            count = 0
            for _, row in df_temp.iterrows():
                distance = self._euclidean_distance(middle_point_circle[0], middle_point_circle[1], row['lat'],
                                                    row['lon'])
                if distance <= radius:
                    count += 1
            ranking_df.loc[partner, 'Percentage Overlap'] = count / len(df_temp) if len(df_temp) > 0 else 0

        return ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

    #Third ranking algorithm
    def k_means(self, df_input, df_input_modified, full_dist_matrix, chosen_company):
        """
        Calculates a ranking based on how many partners fall within the optimal clusters determined
        by the k-means clustering algorithm centered around the locations of the chosen company

        input:
        df_input (DF): A DataFrame with columns for 'name', 'lat', and 'lon' of companies
        df_input_modified (DF): A modified version of df_input that may be used for clustering
        full_dist_matrix (DF): A DataFrame containing distances between companies, where companies are both
                               in rows and columns
        chosen_company (str): The name of the company to base the clusters on

        output:
        DataFrame: A DF with companies ranked by the percentage of their locations that fall within the
                   same clusters as the chosen company
        """

        #Clean column names and index of the input matrix
        cleaned_matrix = self.clean_column_and_index_labels(full_dist_matrix)

        #Prepare calculation
        minimal_clusters = len(df_input[df_input['name'] == chosen_company])
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        scaler = StandardScaler()
        df_input[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_input[['lat', 'lon']])

        #Determine number of clusters using silhouette scoring and distances
        silhouette_scores_kmeans = {}
        b_s = {}  # between cluster distances
        a_s = {}  # average intra cluster distances
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i)
            df_new = df_input_modified.copy()
            df_new['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
            b_s[i] = self._average_distance_between_clusters(df_new, full_dist_matrix)
            silhouette_scores_kmeans[i], a_s[i] = self._mean_intra_cluster_distance(df_new, full_dist_matrix, b_s[i], i)

        #Determine the optimal number of clusters
        optimal_clusters = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get) + minimal_clusters
        kmeans = KMeans(n_clusters=optimal_clusters)
        df_input['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
        clusters_company = df_input[df_input['name'] == chosen_company]['cluster'].unique()

        #Get ranking
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names, dtype='float')
        for partner in partner_names:
            partner_df = df_input[df_input['name'] == partner]
            partner_data = partner_df[partner_df['name'] == partner]
            is_in_cluster = partner_data['cluster'].isin(clusters_company)
            ranking_df.loc[partner, 'Percentage Overlap'] = is_in_cluster.mean()

        return ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

    # Fourth ranking algorithm
    def dbscan(self, input_df, full_matrix, chosen_company, eps=17, ms=2):
        """
        Calculates a ranking based on how many partners fall within the clusters determined
        by the DBSCAN clustering algorithm, excluding noise points and focusing on the proximity
        to the chosen company

        input:
        input_df (DF): A DF with columns for 'name' and other attributes of companies
        full_matrix (DF): A precomputed distance matrix used by DBSCAN, where companies are both
                          in rows and columns
        chosen_company (str): The name of the company around which the clustering is analyzed
        eps (int): The maximum distance between two samples for one to be considered as in the
                   neighborhood of the other (calculated by found equation)
        ms (int): The number of samples in a neighborhood for a point to be considered as a core
                  point (default=2)

        output:
        DataFrame: A DF with companies ranked by the percentage of their locations that fall within
                   the same clusters as the chosen company, excluding noise points
        """

        df_copy = input_df.copy()

        #Initialize DBSCAN with the given parameters and compute clustering
        epsilon = int(round(19 * (len(input_df) ** -0.1) + 5, 0))
        dbscan = DBSCAN(eps=epsilon, min_samples=ms, metric='precomputed')
        clusters = dbscan.fit_predict(full_matrix)
        df_copy['cluster'] = clusters  # Assign cluster labels to the dataframe

        #Assign noise points to nearest cluster or filter them out
        df_noise_assign = self._assign_noise_points(df_copy, chosen_company, True)
        df_filter_noise = df_noise_assign[df_noise_assign['cluster'] != -1]

        #Calculate percentage of points in the same cluster as the chosen company
        percentages = self._get_percentages(df_filter_noise, chosen_company)
        percentages.index = percentages["name"]
        percentages = percentages.drop(columns=["name"])
        percentages.index.name = None

        return percentages

    #Remove everything after '_' in column names and index
    def clean_column_and_index_labels(self, matrix):
        matrix_copy = matrix.copy()  # Create a copy to avoid modifying the original
        matrix_copy.columns = [col.split('_')[0] for col in matrix_copy.columns]
        matrix_copy.index = [idx.split('_')[0] for idx in matrix_copy.index]
        return matrix_copy

    #Calculate Euclidean distance between two points
    def _euclidean_distance(self, lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    def _mean_intra_cluster_distance(self,df_input_modified, distance_matrix, average_distance_between_clusters, k):
        #Get correct df format
        grouped = df_input_modified.groupby('cluster')
        silhouette_scores = np.zeros(k)
        #Calculate the distance within clusters
        for cluster, group in grouped:
            names = df_input_modified.loc[group.index, 'name'].tolist()
            sub_matrix = distance_matrix.loc[names, names]
            #Because of symmetry in distance matrix use upper right triangle
            mask = np.triu(np.ones(sub_matrix.shape, dtype=bool), k=1)
            upper_triangle_values = sub_matrix.values[mask]
            mean_distance_in_cluster = upper_triangle_values.mean() if len(names) > 1 else 0
            silhouette_scores[cluster]= (average_distance_between_clusters-mean_distance_in_cluster)/max(average_distance_between_clusters,mean_distance_in_cluster)
        return np.mean(silhouette_scores), mean_distance_in_cluster

    def _average_distance_between_clusters(self,df_input_modified , distance_matrix):
        #Get right df format
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
        #Decide if all noise points are assigned or only company noise
        if company is True:
            noise_df = df[(df['name'] == chosen_company) & (df['cluster'] == -1)].reset_index(drop=True)
        else:
            noise_df = df[df['cluster'] == -1].reset_index(drop=True)

        if noise_df.empty:
            return df

        #Get all unique clusters
        clusters = df['cluster'].unique()
        clusters = sorted(clusters)
        centroids = []

        #Get centroids of all clusters
        for cluster in clusters:
            if cluster != -1:
                cluster_points = df[df['cluster'] == cluster]
                centroid = (cluster_points['lat'].mean(), cluster_points['lon'].mean())
                centroids.append(centroid)
        centroids = np.array(centroids)

        #Locate nearest cluster for noise points
        for i in range(len(noise_df)):
            point = (noise_df['lat'][i], noise_df['lon'][i])
            distances = euclidean_distances([point], centroids)
            nearest_cluster = np.argmin(distances)
            noise_df.loc[i, 'cluster'] = clusters[nearest_cluster]

        #Assign new clusters to df
        for idx, row in noise_df.iterrows():
            mask = (
                    (df['name'] == row['name']) &
                    (df['lat'] == row['lat']) &
                    (df['lon'] == row['lon'])
            )
            df.loc[mask, 'cluster'] = row['cluster']

        return df

    def _get_percentages(self, df_new, chosen_company):
        #Get unique clusters of the chosen company
        unique_clusters = df_new.loc[df_new['name'] == chosen_company, 'cluster'].unique()
        filtered_df = df_new[df_new['cluster'].isin(unique_clusters)]
        matching_counts = filtered_df.groupby('name').size()
        total_counts = df_new.groupby('name').size()
        percentages = (matching_counts / total_counts).reset_index()

        # Clean percentage output
        percentages.rename(columns={0: 'Percentage'}, inplace=True)
        percentages = percentages.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        percentages['Percentage'] = percentages['Percentage'].fillna(0)
        percentages = percentages[percentages['name'] != chosen_company]
        return percentages

    #Machine learning-based ranking algorithm
    def ranker_ml(self, data_to_predict_on, model_file, scaler_file, CHOSEN_CANDIDATES):
        """
        Calculates a ranking of candidates based on predicted distances using a pre-trained machine
        learning model. The lower the predicted distance, the higher the rank of the candidate.

        input:
        data_to_predict_on (DataFrame): A DataFrame containing the features required by the model
                                        to make predictions.
        model_file (str): The file path to the trained machine learning model.
        scaler_file (str): The file path to the scaler used for normalizing the data.
        CHOSEN_CANDIDATES (list): A list of candidate names corresponding to the rows in
                                  'data_to_predict_on'.

        output:
        DataFrame: A DataFrame with candidates ranked by the predicted 'Total Distance' to a
                   specific target, sorted in ascending order.
        """

        # Initialize the model predictor with the provided model and scaler files
        predictor = ModelPredictor(model_file, scaler_file)

        # Predict the distances using the provided data
        kms = predictor.predict(data_to_predict_on)

        # Create a DataFrame to display the predicted distances along with candidate names
        ranking = pd.DataFrame(kms, columns=["Total Distance"], index=CHOSEN_CANDIDATES)

        # Return the DataFrame sorted by 'Total Distance' in ascending order for ranking
        return ranking.sort_values(by=["Total Distance"], ascending=True)

