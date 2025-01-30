#########################################################
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

from Input_Transformation.transforming_input import TransformInput
import pandas as pd

def main():

    #######CHECK 1
    #Check for df
    df = pd.read_csv("Data/manyLarge.csv")
    check_road_proximity = False #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    df_modified = transformer.execute_validations(df)

    #######CHECK 2
    #Check for different values on the Dutch map (also no sea/waters)
    data = {'lat': [52.548397087668235, 52.40988381501502, 52.520008, 52.62619574073854, 52.34661594123669], 'lon': [5.225299845087256, 4.941418050084637, 13.404954, 5.022999973718499, 5.576027859636472],
            'name': ['Midden_ijselmeer', 'Amsterdam', 'Berlijn', 'Hoorn_505meter', 'midden in een rivier']}
    df = pd.DataFrame(data)
    check_road_proximity = True
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    df_modified = transformer.execute_validations(df)

if __name__ == "__main__":
    main()

