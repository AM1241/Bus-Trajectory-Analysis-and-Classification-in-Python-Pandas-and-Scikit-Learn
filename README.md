# Bus-Trajectory-Analysis-and-Classification-in-Python-Pandas-and-Scikit-Learn
An application of data mining techniques, namely: collection, pre-processing / cleaning, conversion, use of data mining techniques and evaluation. The implementation is in Python using the SciKit-Learn tool. This work is related to the categorization of space-time data (bus tracks of Dublin city).

<b>Trajectory Analysis and Classification in Python (Pandas and Scikit Learn) </b>

A university project  for the postgraduate class of Data Mining.

We were given a train_set with geographical points paired with the time interval. Firstly, we cleaned the dataset and then we formed the trajectories (with the corresponding route id). The last step of this part was to filter out some trajectories based on _their total_distance and max distance (between two of their points).

The goal of this project was firstly to compute trajectory similarity between trajectories of test_set_a1/a2.csv and the train_set.csv. 

The algorithms used for that were :
1) <b>Fast Dynamic Time Warping (Fast-DTW)</b>, taken from https://github.com/slaypni/fastdtw
2) <b>Longest Common Subsequence algorithm</b>, which i implemented.

<b>(a) Data pre-processing </b>

The process of cleaning the track data of the dataset train_set.csv. The whole process is implemented in the datacleaning.py file, while individual useful functionalities called as functions from auxiliaryfunctions.py file.
First, the log file is readed and all rows with a null value in the JourneyPatternId field are removed. The resulting dataframe is limited to 1,484,821 rows (261,557 rows less than the original).
A a new field called route is created. It is the concatenated string of VehicleId and timestamp. Based on this a classification is made in order to separate the routes. The result of this transformation is saved in the Sorted.csv file,  and the df reindexing is performed.
Next, according to the information in the previous file, we get the different trajectories based on the rotation of the JoyrneyPatternId field, merging the timestamp, longitute, latitute into one called timestamp_longitute_latitute, which are stored in the TripId.df file, creating a new dataframe with three columns (TripId, JourneyPatternId, timestamp_longitute_latitute). The df type is chosen because it allows the proper storage of a variety of data structures (eg lists or lists of lists in our case). Each trajectory is identified by its TripId and is located in the timestamp_longitude_latitude field, like a list of (global) points. The number of tracks in the exported TripId file is 7,435.

<b> (b) Data clearance </b>

In the same datacleaning.py file data cleaning takes place. The distance taken into account each time, was the Havershine distance of the points, which is calculated in a function located in the file auxiliaryfunctions.py. Given an orbit as input (field timestamp_longitude_latitude ) returns a list of two items. In the first element is the total orbital distance calculated by Havershine in kilometers (considering the radius of the Earth 6371km), and in the second element is the largest distance between two points of the orbit. Tracks that either have a total distance of less than 2 km or a longer point distance of more than 2 km are deleted. The final output file is final_cleaned.df.
Initially there were 7435 tracks in the input file. With the application of the first filter, 193 tracks were removed. By applying the second filter, 858 tracks were removed. There are 6384 tracks in the aforementioned final file.
  
<b> (c) Data Visualization </b>

Finally, in the last part of the datacleaning.py file, the program selects 5 randomly different tracks from the final_cleaned.df file and designs them using the gmplot library. The plot_traj function located in the auxiliaryfunctions.py file implements the design and the final images are saved in the RandomImages folder that is created during the execution of the program.

<b>(d.1) Finding nearest neighbors </b>

DTW

The fastdtw package was used for the DTW technique, which is a dtw approach (https://pypi.python.org/pypi/fastdtw). The code that implements this query is in the fast_dtwneigbors.py file.
Finally, for each trajectory of the test file, all trajectories of the final_cleaned.df file are scanned and all the 'distances' are calculated via fastdtw, using the Havershine distance function used in the previous query.
To find the 5 nearest neighbors, the distances table containing all distances is sorted (via argsort). Since the distances table is in a 1-1 ratio with our df, the indices of the top5 distances are the indices of the top5 JourneyPatternIDs.
  
<b>(d.2) Finding nearest neighbors </b>

LCSS

An implementation (https://github.com/maikol-solis/trajectory_distance/blob/master/traj_dist/pydist/lcss.py) of the algorithm found in this link was used for LCSS. The code that implements this query is in the lcss_neigbors.py file. First, read the file created in (a), as well as the test_set_a2.csv file, using regular expressions in python to get the results correctly. Next, the program creates the LCSSresults folder where the images (in html) will be saved by gmplot.
Finally, for each trajectory of the test file, all trajectories of the final_cleaned.df file are scanned and all common points are calculated via lcss, using the Havershine distance function used in the previous query.
To find the common points, the distances table containing all distances is sorted (via argsort). Two points will be matched if the distance does not exceed 200m. Thus, the calculation time of the common 5-point points, the JourneyPatternId for each of the neighbors detected, the number of points encountered with each of the 5 neighbors, the visualization of the given path and the visualization of the five nearby in the subdivisions detected in red, and in blue the entire route of the neighbor is shown in the pictures below for each trajectory.
 

The second part of the project was to train KNN,Random Forest, and Logistic Regression classifiers and predict the routes of trajectories of the test_set.csv . The first step was to assign each trajectory to a string (composed of cell codes) via a grid representation. In the second step, 10-cross-fold-validation was used to train the classifiers with grid strings of the dataset with accuracy metric . I conducted various experiments, by changing each classifier's parameters. 

Lastly,the classifiers with the best accuracy were bunched together in the Voting Classifier. The final classifier was used to find labels for the trajectories of the test_set.csv .
  
<b>(e) Export Features for Categorization </b>

To apply the two-dimensional Grid to the coordinates of the paths, in order to represent them as a set of cells, we first locate the point "0.0" of the theoretical Cartesian plane, through the function down_left_point.py which results from the combination of the recommended latitude of the most south point of our set, and the recommended lontitute of the westernmost point of our set.
This point is (53.070450, -6.61505).
Then we create the grid_points.py file in which all the points of each path are traversed and after locating them in the Grid (calculating their distance at Haversine from each remote axis) they are replaced by the corresponding Grid cell. The result is exported to the grids.csv file, which consists of two columns (TripId, and Grids). Grids store the entire sequence of cells in the Grid of each path.
  
<b>(f) Categorization </b>

In the categorization step we used the 3 classifiers given to the pronunciation (Knearest Neigbors, Random Forest and Logistic Regression) from the scikit learn package.
As for the experiments we did, we first experimented with SVD which is responsible for reducing the dimension of the vectors exported by the vectorizers.
  
For each experiment performed, the classifytraj.py file generated an EvaluationMetricAccuracy file where the accuracy for each of the 10 folds was kept for each categorizer. Starting with 50 coordinates per orbit, the results we have had are mediocre to disappointing. Especially, in the last fold, the (trained LogisticRegression categorizer) has a yield of only 20%.
Then we performed the same experiment (keeping the default arguments of the classifiers) and increased the dimension of the vectors from 50 to 100. The conclusion was that the accuracy of the classifiers went up 5% -10%. From the first two experiments it seems that Random Forest is the best and Logistic Regression the worst 

 In the next two experiments, having a fixed SVD of 100 dimensions, we changed the arguments of the categorizers. More specifically, in the third experiment NN ran examining the 3 closest neighbors, Random Forests with 50 estimators and Logistic Regression with a tolerance of 0.000001 and solver = sag. All classifiers in all experiments ran with n_jobs = -1, using all system cores.
  
  We observe that the yield of all Random Forest increases by 5% while of the rest from minimal (KNN) to not at all. In the fourth experiment, for KNN we increased the neighbors to 7 for KNN, the estimators to 200 for Random Forest and for the Logistic Regression ton maximum number of iterations for convergence to 500. The most important observation from this experiment is that the performance of the KNN fell making the neighbors 7 out of 3.
  
Random Forest showed a slight improvement while Logistic Regression remained stable and disappointing once again in terms of performance in fold10, which is only 31%. In the last two experiments, the dimension of the vectors is now 300, so that less information is lost as the dimension decreases. The neighbors for KNN are in both experiments 5 the estimators for Random Forest in the 5th experiment are 400 and regarding Logistic Regression the tolerance is 0.0000001, the maximum number of repetitions is 1000 and in this case the solver is lbfgs.
  
The most significant improvement in this experiment was Logistc Regression, but in general all classifiers had an improvement in their performance of 1-5%. The only downside here is the time from the 8m36secs of the 4th experiment to 22m6.38secs.
In the last experiment the estimates for Random Forest became 500 and in Logistic Regression the maximum convergence repetitions became 2000 and now the solver is sag. The improvement in performance is almost non-existent. The time on the other hand increased to 26m26.131secs.
  
<b>Beat the Benchmark </b>

From the above experiments we concluded that for this dataset the best classifier was Random Forest followed by KNN and Logistic Regression. As a last resort to improve performance we decided to use the Voting Classifier, which decided according to the majority of the 3 classifiers it took into account. The first was KNN with 5 neighbors and the other two were Random Forest with 400 and 500 appraisers respectively. The pretreatment of the applied data was similar to experiments 5 and 6. The time taken for 10-fold CrossValidation was 32m and the results are shown below:
We observe that the m.o. of accuracy for folds is at 80-82%. 
  
<b>Test Set Prediction </b>

The Python source code is in the test_prediction.py file. It first reads the existing tracks from the test_set.csv file. It then runs the grid function for each of them to generate the cell string. The major drawback that affected our performance is that the test_set cells_strings generated by the grid are short (~ 40-45 cells). So, 'necessarily' the svd could not work with 300 vectors. Therefore we chose the vectors to be of space of 40 dimensions since it is not possible to train our best categorizer (Voting Classifier) in space of 300 dimensions and to examine its forecast in space of 40.
Therefore, we created two pipelines, the pipeline where he trained the aforementioned Voting Classifier with all the train_set and the pipeline_test which he gave to the categorizer to predict the total trajectories of the test_set. The forecast results can be found in the testSet_JP_IDS.csv file.
