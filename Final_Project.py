import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import tkinter as tk
from tkinter import filedialog

#Global variables
global_path = ""
output_created = False
path_selected = False


#open the file selection menu
def open_file():

    global global_path
    global path_selected
    # Open the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    # Check if a file was selected
    if file_path:
        # Update the label with the file path
        file_path_label.config(text=file_path)
        global_path = file_path

        path_selected = True

    #check if no file was selected
    else:
        file_path_label.config(text="No file selected")
        global_path=""


# ----------------------------------------------------------------------------------------------------------------------
def create_segments(dataset):
    # Combine the walking and jumping dataframes and then shuffle the data for training

    # Segment size of 5000ms or 5sec
    segment_size = 5000
    num_segments = len(dataset) // segment_size
    # Creates the segments
    segments = [dataset.iloc[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    # Shuffles the segment
    random.shuffle(segments)
    # Concatenates the segments into a dataframe
    dataset = pd.concat(segments)

    dataset.reset_index(drop=True, inplace=True)

    return dataset


#------------------------------------------------------------------------------------------------------------------
#Classifys the inputted CSV file using the regression model

def Classify_Input():

    global output_created
    #Check if a file was selected
    if path_selected:

        #load the data into a variable
        input_dataframe = pd.read_csv(global_path)
        input_dataframe['Time (s)'] = input_dataframe['Time (s)'].astype(float)

        input_dataframe['Time (s)'] = (input_dataframe['Time (s)'] // 5) * 5

        input_dataframe = input_dataframe.groupby('Time (s)').agg({
            'Acceleration x (m/s^2)': 'mean',
            'Acceleration y (m/s^2)': 'mean',
            'Acceleration z (m/s^2)': 'mean',
            'Absolute acceleration (m/s^2)': 'mean'
        }).reset_index()

        #remove outliers
        for i in range(len(input_dataframe)):
            if input_dataframe.iloc[i, 4] > 15:
                input_dataframe.iloc[i, 4] = 15
            if input_dataframe.iloc[i, 4] < 6:
                input_dataframe.iloc[i, 4] = 6

        input_data, input_labels = Normalize_Data(input_dataframe)

        input_dataframe['Classification'] = clf.predict(input_data.values.reshape(-1,1))

        label_activity = {0:'Walking', 1:'Jumping'}
        input_dataframe['Classification'] = input_dataframe['Classification'].map(label_activity)
        input_dataframe.to_csv('output_file.csv', index=False)

        output_created = True


#-----------------------------------------------------------------------------------------------------------------------
#Code to create the desktop app


main = tk.Tk()

main.title("Walking Vs Jumping Classifier")

main.geometry("500x500")

# Create labels
title_label = tk.Label(main, text = "Walking Vs Jumping", font = ('TkFixedFont bold', 16, ))
title_label.pack(pady = 20)

text_label = tk.Label(main, text = "Select a file to model", font = ('TkFixedFont bold', 13, ))
text_label.pack(pady = 10)

#Create a button to open the file 
open_file_btn = tk.Button(main, text="Choose File", command=open_file)
open_file_btn.pack(pady=5)

# Create a label to display the selected file path
file_path_label = tk.Label(main, text="No file selected yet")
file_path_label.pack(pady=10)
#-----------------------------------------------------------------------------------------------------------------------
#Code to read csv files
tejas_walk = pd.read_csv('Tejas_Walk.csv')
tejas_jump = pd.read_csv('Tejas_Jump.csv')
nicole_walk = pd.read_csv('Nicole_Walk.csv')
nicole_jump = pd.read_csv('Nicole_Jump.csv')
akin_walk = pd.read_csv('Akin_Walk.csv')
akin_jump = pd.read_csv('Akin_Jump.csv')
# ----------------------------------------------------------------------------------------------------------------------
# Code to combine the files for walking first
filenames = ['Tejas_Walk.csv', 'Nicole_Walk.csv', 'Akin_Walk.csv']
dataframes = [pd.read_csv(filename) for filename in filenames]
walk_Combdf = pd.concat(dataframes, ignore_index=True)
walk_Combdf.to_csv('walk_comb.csv', index=False)

# Code to combine the files for jumping
filenames = ['Tejas_Jump.csv', 'Nicole_Jump.csv', 'Akin_Jump.csv']
dataframes = [pd.read_csv(filename) for filename in filenames]
jump_Combdf = pd.concat(dataframes, ignore_index=True)
jump_Combdf.to_csv('jump_comb.csv', index=False)

# Code to split the data into segments and shuffle them
shuffled_walk = create_segments(walk_Combdf)

shuffled_jump = create_segments(jump_Combdf)


# ----------------------------------------------------------------------------------------------------------------------

# Make groups for all the data
with h5py.File('output.hdf5', 'w') as f:
    MemberData = f.create_group('MemberData')
    MemberData.create_dataset('Tejas_Walk', data=tejas_walk.to_numpy())
    MemberData.create_dataset('Akin_Walk', data=nicole_walk.to_numpy())
    MemberData.create_dataset('Nicole_Walk', data=akin_walk.to_numpy())
    MemberData.create_dataset('Tejas_Jump', data=tejas_jump.to_numpy())
    MemberData.create_dataset('Akin_Jump', data=nicole_jump.to_numpy())
    MemberData.create_dataset('Nicole_Jump', data=akin_jump.to_numpy())

    dataGroup = f.create_group('Datasets')
    dataGroup.create_dataset('walk_Combdf', data=walk_Combdf.to_numpy())
    dataGroup.create_dataset('jump_Combdf', data=jump_Combdf.to_numpy())
    dataGroup.create_dataset('walk_shuffled', data=shuffled_walk.to_numpy())
    dataGroup.create_dataset('jump_shuffled', data=shuffled_jump.to_numpy())

with h5py.File('output.hdf5', 'r') as f:
    jump_shuffled = f['Datasets/jump_shuffled'][:]
    walk_shuffled = f['Datasets/walk_shuffled'][:]
# ----------------------------------------------------------------------------------------------------------------------
# Apply the moving average filter on the jump and walking data
jump_df = pd.DataFrame(jump_shuffled)
jump_df.iloc[:, 4] = jump_df.iloc[:, 4].rolling(window=5).mean()

# To remove outliers reduce their value to a resonable number, then apply moving average at end
walk_df = pd.DataFrame(walk_shuffled)
for i in range(len(walk_df)):
    if walk_df.iloc[i, 4] > 15:
        walk_df.iloc[i, 4] = 15
    if walk_df.iloc[i, 4] < 6:
        walk_df.iloc[i, 4] = 6
walk_df.iloc[:, 4] = walk_df.iloc[:, 4].rolling(window=5).mean()

combined = pd.concat([walk_df, jump_df])

# ----------------------------------------------------------------------------------------------------------------------

# Make groups for all the data
with h5py.File('output.hdf5', 'w') as f:
    MemberData = f.create_group('MemberData')
    MemberData.create_dataset('Tejas_Walk', data=tejas_walk.to_numpy())
    MemberData.create_dataset('Akin_Walk', data=nicole_walk.to_numpy())
    MemberData.create_dataset('Nicole_Walk', data=akin_walk.to_numpy())
    MemberData.create_dataset('Tejas_Jump', data=tejas_jump.to_numpy())
    MemberData.create_dataset('Akin_Jump', data=nicole_jump.to_numpy())
    MemberData.create_dataset('Nicole_Jump', data=akin_jump.to_numpy())

    dataGroup = f.create_group('Datasets')
    dataGroup.create_dataset('walk_Combdf', data=walk_Combdf.to_numpy())
    dataGroup.create_dataset('jump_Combdf', data=jump_Combdf.to_numpy())
    dataGroup.create_dataset('walk_shuffled', data=shuffled_walk.to_numpy())
    dataGroup.create_dataset('jump_shuffled', data=shuffled_jump.to_numpy())

with h5py.File('output.hdf5', 'r') as f:
    jump_shuffled = f['Datasets/jump_shuffled'][:]
    walk_shuffled = f['Datasets/walk_shuffled'][:]
# ----------------------------------------------------------------------------------------------------------------------
# Creating Graphs for Data Visualization

def Display_Plots():
    jump_df = pd.DataFrame(jump_shuffled)
    walk_df = pd.DataFrame(walk_shuffled)

    colour = {'Walking': 'blue', 'Jumping': 'red'}
    legends = ['Walking', 'Jumping']

    if output_created:
        num_plots = 3
        new_df = pd.read_csv('output_file.csv')
    else:
        num_plots = 2
    #plotting acceleration in the x direction vs time, for both jumping and walking
    fig, ax = plt.subplots(num_plots, figsize=(20, 10))
    ax[0].plot(jump_df.iloc[:, 1], color = colour['Jumping']), ax[0].set_title('Jumping Acceleration in the X-Direction VS Time')
    ax[1].plot(walk_df.iloc[:, 1], color = colour['Walking']), ax[1].set_title('Walking Acceleration in the X-Direction VS Time')
    if output_created:
        # Filter and plot data for each classification on the same subplot
        prev_time = new_df.iloc[0]['Time (s)']
        prev_acc = new_df.iloc[0]['Acceleration x (m/s^2)']
        prev_class = new_df.iloc[0]['Classification']

        #Loop through the dataframe to change colours depending on walking or jumping
        for index, row in new_df.iterrows():

            this_time = row['Time (s)']
            this_acc = row['Acceleration x (m/s^2)']
            this_class = row['Classification']

            ax[2].plot([prev_time,this_time], [prev_acc, this_acc], color = colour[this_class])

            prev_time = this_time
            prev_class = this_class
            prev_acc = this_acc

        ax[2].set_title('Input Acceleration in the X-Direction VS Time')
        ax[2].legend(legends, title = 'Legend')  # Add a legend to differentiate the activities

    for axes in ax:
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('X-Acceleration (m/s^2)')

    #plotting acceleration in the y direction vs time, for both jumping and walking
    fig2, ax1 = plt.subplots(num_plots, figsize=(20, 10))
    ax1[0].plot(jump_df.iloc[:, 2], color = colour['Jumping']), ax1[0].set_title('Jumping Acceleration in the Y-Direction VS Time')
    ax1[1].plot(walk_df.iloc[:, 2], color = colour['Walking']), ax1[1].set_title('Walking Acceleration in the Y-Direction VS Time')
    if output_created:
        # Filter and plot data for each classification on the same subplot
        prev_time = new_df.iloc[0]['Time (s)']
        prev_acc = new_df.iloc[0]['Acceleration y (m/s^2)']
        prev_class = new_df.iloc[0]['Classification']

        # Loop through the dataframe to change colours depending on walking or jumping
        for index, row in new_df.iterrows():
            this_time = row['Time (s)']
            this_acc = row['Acceleration y (m/s^2)']
            this_class = row['Classification']

            ax1[2].plot([prev_time, this_time], [prev_acc, this_acc], color=colour[this_class])

            prev_time = this_time
            prev_class = this_class
            prev_acc = this_acc

        ax1[2].set_title('Input Acceleration in the Y-Direction VS Time')
        ax1[2].legend(legends, title='Legend')  # Add a legend to differentiate the activities

    for axes in ax1:
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Y-Acceleration (m/s^2)')

    #plotting acceleration in the z direction vs time, for both jumping and walking
    fig3, ax2 = plt.subplots(num_plots, figsize=(20, 10))
    ax2[0].plot(jump_df.iloc[:, 3], color = colour['Jumping']), ax2[0].set_title('Jumping Acceleration in the Z-Direction VS Time')
    ax2[1].plot(walk_df.iloc[:, 3], color = colour['Walking']), ax2[1].set_title('Walking Acceleration in the Z-Direction VS Time')
    if output_created:
        # Filter and plot data for each classification on the same subplot
        prev_time = new_df.iloc[0]['Time (s)']
        prev_acc = new_df.iloc[0]['Acceleration z (m/s^2)']
        prev_class = new_df.iloc[0]['Classification']

        # Loop through the dataframe to change colours depending on walking or jumping
        for index, row in new_df.iterrows():
            this_time = row['Time (s)']
            this_acc = row['Acceleration z (m/s^2)']
            this_class = row['Classification']

            ax2[2].plot([prev_time, this_time], [prev_acc, this_acc], color=colour[this_class])

            prev_time = this_time
            prev_class = this_class
            prev_acc = this_acc

        ax2[2].set_title('Input Acceleration in the Z-Direction VS Time')
        ax2[2].legend(legends, title='Legend')  # Add a legend to differentiate the activities

    for axes in ax2:
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Z-Acceleration (m/s^2)')

    #plotting absolute acceleration vs time, for both jumping and walking
    fig4, ax3 = plt.subplots(num_plots, figsize=(20, 10))
    ax3[0].plot(jump_df.iloc[:, 4], color = colour['Jumping']), ax3[0].set_title('Jumping Absolute Acceleration VS Time')
    ax3[1].plot(walk_df.iloc[:, 4], color = colour['Walking']), ax3[1].set_title('Walking Absolute Acceleration VS Time')
    if output_created:
        # Filter and plot data for each classification on the same subplot
        prev_time = new_df.iloc[0]['Time (s)']
        prev_acc = new_df.iloc[0]['Absolute acceleration (m/s^2)']
        prev_class = new_df.iloc[0]['Classification']

        # Loop through the dataframe to change colours depending on walking or jumping
        for index, row in new_df.iterrows():
            this_time = row['Time (s)']
            this_acc = row['Absolute acceleration (m/s^2)']
            this_class = row['Classification']

            ax3[2].plot([prev_time, this_time], [prev_acc, this_acc], color=colour[this_class])

            prev_time = this_time
            prev_class = this_class
            prev_acc = this_acc

        ax3[2].set_title('Input Absolute Acceleration VS Time')
        ax3[2].legend(legends, title='Legend')  # Add a legend to differentiate the activities
    for axes in ax3:
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Absolute acceleration (m/s^2)')

    #prepairing to graph more specific metrics, including meta data information from each individuals results
    njump_df = pd.DataFrame(nicole_jump)
    nwalk_df = pd.DataFrame(nicole_walk)
    tjump_df = pd.DataFrame(tejas_jump)
    twalk_df = pd.DataFrame(tejas_walk)
    ajump_df = pd.DataFrame(akin_jump)
    awalk_df = pd.DataFrame(akin_walk)

    #plotting the invidual walking and jumping data, to find meta data trends
    figure,axs = plt.subplots(6, figsize=(20,10))
    axs[0].plot(njump_df.iloc[:, 4]), axs[0].set_title('Nicole Jumping Absolute Acceleration VS Time')
    axs[1].plot(nwalk_df.iloc[:, 4]), axs[1].set_title('Nicole Walking Absolute Acceleration VS Time')
    axs[2].plot(tjump_df.iloc[:, 4]), axs[2].set_title('Tejas Jumping Absolute Acceleration VS Time')
    axs[3].plot(twalk_df.iloc[:, 4]), axs[3].set_title('Tejas Walking Absolute Acceleration VS Time')
    axs[4].plot(ajump_df.iloc[:, 4]), axs[4].set_title('Akin Jumping Absolute Acceleration VS Time')
    axs[5].plot(awalk_df.iloc[:, 4]), axs[5].set_title('Akin Walking Absolute Acceleration VS Time')

    axs[5].set_xlabel('Time (s)')
    axs[3].set_ylabel('Absolute Acceleration (m/s^2)')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Find the features of the walk data and plot them in subplots
def Display_Features():
    numIter = len(walk_df) // 5000
    features = pd.DataFrame(index=range(numIter), columns=['mean', 'std', 'max', 'min', 'skew',
                                     'median', 'range', 'kurtosis', 'variance', 'sum'])
    window = 0
    for i in range(numIter):
        features.at[i, 'mean'] = walk_df.iloc[window:window + 5000, 4].mean()
        features.at[i, 'std'] = walk_df.iloc[window:window + 5000, 4].std()
        features.at[i, 'max'] = walk_df.iloc[window:window + 5000, 4].max()
        features.at[i, 'min'] = walk_df.iloc[window:window + 5000, 4].min()
        features.at[i, 'skew'] = walk_df.iloc[window:window + 5000, 4].skew()
        features.at[i, 'median'] = walk_df.iloc[window:window + 5000, 4].median()
        features.at[i, 'range'] = walk_df.iloc[window:window + 5000, 4].max() - walk_df.iloc[window:window+5000, 4].min()
        features.at[i, 'kurtosis'] = walk_df.iloc[window:window + 5000, 4].kurt()
        features.at[i, 'variance'] = walk_df.iloc[window:window + 5000, 4].var()
        features.at[i, 'sum'] = walk_df.iloc[window:window + 5000, 4].sum()
        window += 5000

    fig3, axs = plt.subplots(5, figsize=(20, 20), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    axs[0].scatter(range(len(features)), features.iloc[:, 0]), axs[0].set_title('mean')
    axs[1].scatter(range(len(features)), features.iloc[:, 1]), axs[1].set_title('std')
    axs[2].scatter(range(len(features)), features.iloc[:, 2]), axs[2].set_title('max')
    axs[3].scatter(range(len(features)), features.iloc[:, 3]), axs[3].set_title('min')
    axs[4].scatter(range(len(features)), features.iloc[:, 4]), axs[4].set_title('skew')
    fig4, axs = plt.subplots(5, figsize=(20, 20), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    axs[0].scatter(range(len(features)), features.iloc[:, 5]), axs[0].set_title('median')
    axs[1].scatter(range(len(features)), features.iloc[:, 6]), axs[1].set_title('range')
    axs[2].scatter(range(len(features)), features.iloc[:, 7]), axs[2].set_title('kurtosis')
    axs[3].scatter(range(len(features)), features.iloc[:, 8]), axs[3].set_title('variance')
    axs[4].scatter(range(len(features)), features.iloc[:, 9]), axs[4].set_title('sum')

    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Now the combined data must be normalized for the training set
def Normalize_Data(dataset):


    #apply the scaler to the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataset)
    data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)

    # If the normalized value is greater than 0.4 it will be considered a jump, else a walk
    bin_data = data_scaled.copy()

    # Fill in the NAN values with the mean
    bin_data = bin_data.fillna(bin_data.mean())

    # Make the data and labels for testing
    data2 = bin_data.iloc[:, 4]
    labels2 = bin_data.iloc[:, -1]

    #if a file has been inputted use this to normalize the data
    if path_selected:

        return data2, labels2

    #if no file was inputted create the lables and return
    else:
        bin_data.loc[bin_data.iloc[:, 4] <= 0.4] = 0
        bin_data.loc[bin_data.iloc[:, 4] > 0.4] = 1
        # Make the data and labels for testing
        data2 = bin_data.iloc[:, 4]
        labels2 = bin_data.iloc[:, -1]
        return data2, labels2, bin_data, data_scaled


# ----------------------------------------------------------------------------------------------------------------------
# Start the training
data, labels, bin_comb, comb_scaled = Normalize_Data(combined)

X_train, X_test, y_train, y_test = train_test_split(data.values.reshape(-1, 1), labels, test_size=0.1, random_state=17)
# Initialize the Logistic Regression model
l_reg = LogisticRegression()
# Train the model on the training data
clf = make_pipeline(StandardScaler(), l_reg)
# Training
clf.fit(X_train, y_train)
# Obtaining the predictions and probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
print('y_pred is: ', y_pred)
print('y_clf_prob is: ', y_clf_prob)
acc = accuracy_score(y_test, y_pred)
print('\nThe accuracy is: ', acc)
# The confusion matrix
cm = confusion_matrix(y_test, y_pred)

print('\nLOOK AT THIS xtest is: ', X_test)

# F1 score
score = f1_score(y_test, y_pred)
print('F1 score is: ', score)

# ROC and AUC of model
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])

# AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print('The AUC is: ', auc)

#-----------------------------------------------------------------------------------------------------------------------
#Display the training data values
def Display_Training():

    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    ConfusionMatrixDisplay(cm).plot()

    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot the dataset for the walking jumping, and shuffled combined data
def Display_Data():
    fig1, axs1 = plt.subplots(4, figsize=(20, 10), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    axs1[0].plot(jump_df.iloc[:, 4]), axs1[0].set_title('Combined Jump Data Set')
    axs1[1].plot(walk_df.iloc[:, 4]), axs1[1].set_title('Combined Walking Data Set')
    axs1[2].plot(bin_comb.iloc[:, 1]), axs1[2].set_title('Combined Shuffled Data Set')
    axs1[3].plot(comb_scaled.iloc[:, 4]), axs1[3].axhline(y=0.4, color='g',linestyle='-'), axs1[3].set_title('Combined Data Normalized')
    for ax1 in axs1:
        ax1.set_xlabel('Time(ms)')
        ax1.set_ylabel('Amplitude')

    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Add the rest of the files to the HDF5 file
with h5py.File('output.hdf5', 'a') as f:
    group = f.require_group('/Datasets')
    group.create_dataset('binary_set', data=bin_comb.to_numpy())
    group.create_dataset('normalized_set', data=comb_scaled.to_numpy())
    group.create_dataset('filtered_walk', data=walk_df.to_numpy())
    group.create_dataset('filtered_jump', data=jump_df.to_numpy())

# ----------------------------------------------------------------------------------------------------------------------
# Show the graphs



#add the feature data labels in the GUI
default_label = tk.Label(main, text="Display input File Data ", font = ('TkFixedFont bold', 13, ))
default_label.pack(pady=5)

#create a frame to add the buttons next to eachother

button_frame2 = tk.Frame(main)
button_frame2.pack(pady=10)

#add the button to classify the input data
input_data_btn = tk.Button(button_frame2, text='Classify Input', command=Classify_Input)
input_data_btn.pack(pady=5)

plot_btn = tk.Button(button_frame2, text='Show Acceleration Plots', command = Display_Plots)
plot_btn.pack(pady=5)

input_data_btn.pack(side='left', padx=10)
plot_btn.pack(side='left')

#add the feature data labels in the GUI
default_label = tk.Label(main, text="Displays Original CSV Information", font = ('TkFixedFont bold', 13, ))
default_label.pack(pady=5)

button_frame = tk.Frame(main)
button_frame.pack(pady=10)



#add the button to display the graphs
default_data_btn = tk.Button(button_frame, text='Display Formatted Data ', command=Display_Data)
default_data_btn.pack(pady=5)

#add the button to display training data
training_data_btn = tk.Button(button_frame, text='Display Training Data ', command=Display_Training)
training_data_btn.pack(pady=5)

#add the button to display the other stuff
feature_data_btn = tk.Button(button_frame, text='Display Features ', command=Display_Features)
feature_data_btn.pack(pady=5)

#add the feature data labels in the GUI
Acc_label = tk.Label(main, text=f"Accuracy: {acc}")
Acc_label.pack(pady=10)

F1_label =tk.Label(main, text=f"F1 score: {score}")
F1_label.pack(pady=10)

AUC_label=tk.Label(main, text=f"AUC: {auc}")
AUC_label.pack(pady=10)

#set the buttons next to eachother
default_data_btn.pack(side='left', padx=10)
training_data_btn.pack(side='left')
feature_data_btn.pack(side ='left', padx=10)

#start the GUI
main.resizable(False, False)
main.mainloop()

