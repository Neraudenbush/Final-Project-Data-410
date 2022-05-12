# Final-Project-Data-410
Final Research Project in DATA 410

## Introduction 
One aspect of data science that interests me a lot is its cross over between sports. One of the fields I would really like to go into as a career is using data to model and predict to help professional baseball teams optimize their results. For this reason, I have elected to do my final research using Data from Major League Baseball (MLB). In 2015 the MLB introduced _Statcast_. 

Description of Statcast"
"Statcast is a state-of-the-art tracking technology, capable of measuring previously unquantifiable aspects of the game. Set up in all 30 Major League ballparks, Statcast collects data using a series of high-resolution optical cameras along with radar equipment. The technology precisely tracks the location and movements of the ball and every player on the field, resulting in an unparalleled amount of information covering everything from the pitcher to the batter to baserunners and defensive players"

This allows for highly accurate data in all aspects of the sport. The dataset I have selected is Pitch data from 2021. I plan on using the information provided in the dataset to see how accuratly I can predict the outcome of a given pitch (hit, strike, ball, etc.).

## Data
The data set I have selected comes from Kaggle and is titled Statcast_2021. The raw data set has 93 different variables with a little over 70,000 observations. This represents every pitch thrown over the course of the season. I aim to use the model to predict the target variable "pitch_name" which is the type of pitch the pitcher threw. There are 8 types of pitches that a pitcher can throw in this dataset: Splitter, Slider, Sinker, Knuckle Curveball, Cutter, Curveball, Changeup, and Fastball.

For the independent variables, some data can be eliminated immediately. There are many columns that have a majority of their values missing, so removing those columns from the data set would be beneficial. Additionally there are variables included that clearly  have no impact on the result of the ptich such as the date. There are also variables that are somewhat repetitive, for each pitcher and batter, it gives the name and the ID number so removing the names of the players would help narrow down the independent variables. 

Although there are many variables in the data set there are ones that I belive will be better predictors. This list includes: Pitch Type (Fastball, curveball etc.), pitch velocity, pitch angle, zone, release point, runners on base, spin direction, spin rate, and more. 

There are also some variables in the data set that I cannot tell what they mean. 

## Methods
### Preparing the data
Before getting into coding for the project, all the necessary packages must be imported
```Python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn import tree
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gbc
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
```
This dataset had over 75,000 observation to work with which is very large so a random sample of 1000 was taken to make the runtimes shorter and the data in general easier to work with 
```Python
statcast_initial = pd.read_csv("Statcast_2021.csv")
statcast = statcast_initial.sample(n=1000)
```
Even after roving data certain variables before loading the data into python, there still may be some independant variables that are correlated with eachother in the dataframe. This is refered to as multicolinearity. In order to prevent this a heatmap is created to determine which variables are correlated and must be removed. 
```Python
fig_dims = (12,8)
fig, ax = plt.subplots(figsize = fig_dims)
sns.heatmap(statcast.corr(), ax=ax)
```
<img src="heatmap1.png" width="600" height="400" alt="hi" class="inline"/>

It is evident that there is multicolinearity within the dataset so independant variables are removed: vy0, vx0, vz0, spin_axis, pitch_number, ax, ay , az, at_bat_number, description. 
```Python
statcast = statcast.drop(columns=['vy0','vx0','vz0', 'spin_axis','pitch_number','ax','ay','az','at_bat_number', 'description'])
```
The resulting heatmap is shown below<br />

<img src="heatmap2.png" width="600" height="400" alt="hi" class="inline"/>

Next, all the catagorical variables need to be replaced with dummy variables. In order to check which variables need to be turned into dummy variables the following code is run. 
```Python
statcast.info()
```
Which returns this 
```Markdown
<class 'pandas.core.frame.DataFrame'>
Int64Index: 997 entries, 514157 to 220322
Data columns (total 21 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   release_speed       997 non-null    float64
 1   release_pos_x       997 non-null    float64
 2   release_pos_z       997 non-null    float64
 3   zone                997 non-null    float64
 4   stand               997 non-null    object 
 5   p_throws            997 non-null    object 
 6   balls               997 non-null    int64  
 7   strikes             997 non-null    int64  
 8   pfx_x               997 non-null    float64
 9   pfx_z               997 non-null    float64
 10  plate_x             997 non-null    float64
 11  plate_z             997 non-null    float64
 12  outs_when_up        997 non-null    int64  
 13  inning              997 non-null    int64  
 14  sz_top              997 non-null    float64
 15  sz_bot              997 non-null    float64
 16  release_spin_rate   997 non-null    float64
 17  release_extension   997 non-null    float64
 18  pitch_name          997 non-null    object 
 19  delta_home_win_exp  997 non-null    float64
 20  delta_run_exp       997 non-null    float64
dtypes: float64(14), int64(4), object(3)
memory usage: 171.4+ KB
```
The two variables that need to be turned into a dummy are p_throws, which represents the hand that the pitcher throws with (Right or Left) and the stance, which reprsents which way the batter stands when they are batting (Right or Left). To turn these into dummy variables the following code is run so that a value of 1 is given if the batter or pitcher is left handed. The pitch_name variable is also an object but that does not turn into a dummy at this point because it is our target variable.
```Python
ThrowHandDummy = pd.get_dummies(statcast['p_throws'])
StanceDummy = pd.get_dummies(statcast['stand'])

statcast = pd.concat((statcast, ThrowHandDummy), axis=1)
statcast.rename(columns = {'L':'pitch_left'}, inplace = True)
statcast = statcast.drop(columns=['R'])

statcast = pd.concat((statcast, StanceDummy), axis=1)
statcast.rename(columns = {'L':'hit_left'}, inplace = True)
statcast = statcast.drop(columns=['R','p_throws','stand'])
```
Now the x and y values need to be set. Additionally there is slight class imbalance as the distribution of the target variable is as follows:

Fastball: 35.9% <br />
Slider: 19.8%<br />
Sinker: 14.5%<br />
Changeup: 10.5%<br />
Curveball: 9.3%<br />
Cutter: 7.2%<br />
Knuckle-Curve: 1.8%<br />
Splitter: 1.2%<br />

Because of this SMOTE will be used to combat the class imblanance. The data will also be scaled in this section.
```Python
#X and Y are set
x = statcast.drop(columns=['pitch_name']).values
y = statcast["pitch_name"].values

#Fixing the class imblance
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

#Scaling the data
scale = StandardScaler()
x_scaled = scale.fit_transform(x)
```
As mentioned before, the target variable is catagorical so a label encoder will be used to that classification models can run on the data
```Python
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```
Now all the set up is complete

### Building and Running Models
#### Decision Tree




### References 
Data
https://www.kaggle.com/datasets/s903124/mlb-statcast-data

Background info
https://baseballsavant.mlb.com/about
