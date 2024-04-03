# Classification with Deep Learning

## Goal
Finish a sequence classification task using deep learning. A trajectory data set with five taxi drivers' daily driving trajectories in 6 months will be provided. The primary objective is to predict which driver each 100-step sub-trajectory, extracted from the daily trajectories, belongs to. To evaluate your model, it will be tested on a separate set of data for five additional days, using the same preprocessing steps to ensure consistent data handling. This approach ensures consistency in data preparation across training and testing phases, allowing the model to accurately attribute each sub-trajectory to the correct driver. 

## Guidelines
* Implement required functions in model.py, train.py, and test.py.
* The file extract_feature.py provides a way to preprocess the data before feeding the data to the neural network, you can customize it to get more features but maintain 100 steps as standard for fair comparison.

## How to run :
training model:
* `$ python main.py train`

testing model:
* `$ python main.py test`

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|**1**   | ... | 90% |
|**2**   | ... | 86% |
|**3**   | ... | 80% |

## Project Guidelines

#### Dataset Description
| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4    |114.10437    |22.573433    |2016-07-02 0:08:45    |1|
|1    |114.179665    |22.558701    |2016-07-02 0:08:52    |1|
|0    |114.120682    |22.543751    |2016-07-02 0:08:51    |0|
|3    |113.93055    |22.545834    |2016-07-02 0:08:55    |0|
|4    |114.102051    |22.571966    |2016-07-02 0:09:01    |1|
|0    |114.12072    |22.543716    |2016-07-02 0:09:01    |0|


Above is an example of what the data looks like. In the data/ folder, each .csv file is trajectories for 5 drivers on the same day. Each trajectory step is detailed with features such as longitude, latitude, time, and status. Data can be found at [Google Drive](https://drive.google.com/open?id=1xfyxupoE1C5z7w1Bn5oPRcLgtfon6xeT)
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. The same plate means the same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means the taxi is occupied and 0 means a vacant taxi.

#### Problem Definition
Given a full-day trajectory of a taxi, you need to extract the sub-trajectories of each 100 steps and predict which taxi driver it belongs to. 

#### Evaluation 
Your model will be evaluated using 100-step sub-trajectories extracted from five days of comprehensive daily trajectories. These test trajectories are not included in the data/ folder provided for training. This ensures a fair assessment of your model's ability to generalize to unseen data.

## Some Tips

* You will need some GPU resources to train deep learning models. There are three resources below you can use:
* Open source GPU
   * [WPI Turing](https://arc.wpi.edu/cluster-documentation/build/html/index.html)
   * now you all have access to WPI Turing GPU cluster with your WPI accounts.
   * [Google Cloud](https://cloud.google.com/free)
   * [Google CoLab](https://colab.research.google.com/)
* Anaconda and virtual environment set tup
   * [Download and install anaconda](https://www.anaconda.com/download)
   * [Create a virtual environment with commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
* Deep learning package
   * [Pytorch](https://pytorch.org/tutorials/)
* **Keywords**. 
   * If you are wondering where to start, you can try to search "sequence classification", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you some insights.
   

