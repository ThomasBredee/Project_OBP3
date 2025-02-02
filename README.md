# Project_OBP Group 1

This research is about creating a helpful tool for managers that makes it easier for logistic companies to work together. The manager can select a company and see the recommendation of the algorithms.  This results in the best partners for them to work with and show what benefits they might get from the partnership. Once the potential partner is selected, our tool will display the best delivery routes on a dashboard. This dashboard will update instantly and use really fast algorithms to make sure everything runs smoothly 

## Installation 
1. Clone this repository
2. Download Docker and give Docker full access 
3. From the website from https://download.geofabrik.de/europe/netherlands.html, download the file called 'netherlands-latest.osm.pbf'
4. Place this downloaded file in the DockerSetup directory
5. Make sure Docker is open and the engine is running
6. Make sure GIT is installed
7. Open the python editor and open the terminal of this
8. Login to Docker by running this command in the terminal: docker login -u <username> and press your password when asked
9. Run the command DockerSetup/setup_osrm.sh First time download of osrm map Netherlands takes a bit longer! This will create a Docker contain on port 5000. (If you want to change this, you can do this in the setup_osrm.sh)
10. Install requirements.txt to get all correct packages and versions to run the code, do this in e new virtual environment to make sure that no packages are conflicting with each other.
11. Command to run dashboard: streamlit run <directory>Project_OBP\main.py




