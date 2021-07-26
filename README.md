# Clustering and Visualizing Gaze

This repository contains a pipeline to cluster gaze data provided by [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

The config.py file contains several different clustering mechanisms, you can experiment with them or create your own.
For the underlying project we used KMEANS clustering, other clustering algorithms might be more appropriate depending on your use scenario.
This set of files is meant to aid testing and veryfying different approaches to gaze-clustering with Open-Face data.

# Setting up

Before using the pipeline you will need:

[Python3](https://www.python.org/downloads/)

[OpenCV](https://pypi.org/project/opencv-python/) For returning the video

[Pandas](https://pandas.pydata.org/)

[Pickle](https://docs.python.org/3/library/pickle.html)

[Numpy](https://numpy.org/)

["Matplotlib"](https://matplotlib.org/)

Additionally you will need the Videos + CSV files returned by OpenFace to run the pipeline


# Interacting with the Clusters:
In config.py set the data-location and the model name correctly. Set the number of clusters and the pre-training preparation (projections) performed on the data before clustering as well as the desired frame-rate of the output video (the video that is saved, the displayed video will compute as fast as the leftover parts)
In build_model.py you can change the specific clustering algorithm used.

Build the model using `python3 build_model.py`


Set the evaluation mode in config.py

Evaluate the model `python3 evaluateVideoModel.py`

Select a region you want to view by drawing a rectangle in the plot that shows up

The video containing the selected frames should play and be saved to the videos folder.

