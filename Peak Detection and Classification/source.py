
# Import the libraries required

import numpy
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks_cwt, find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize

# ----------------------------------------------------- MAIN DEFINITION -------------------------------------------------------------

def main():

    # Import the necessary data to be operated on
    training_data, index_data, class_data, submission_data = importData()

    # Initialise the necessary variables
    low_cutoff_freq, high_cutoff_freq, sample_rate, training_filter_window_size, submission_filter_window_size, filter_order = initialise()

    # Perform butterworth bandpass filter on both the training and submission data to get rid of majority of noise
    training_filtered_signal =  filter_signal(training_data, low_cutoff_freq, high_cutoff_freq, sample_rate, training_filter_window_size, filter_order)
    submission_filtered_signal =  filter_signal(submission_data, low_cutoff_freq, high_cutoff_freq, sample_rate, submission_filter_window_size, filter_order)
    
    # Find the peak locations in both the training and submission data
    training_peak_locations = find_filtered_peaks(training_filtered_signal)
    submission_peak_locations = find_filtered_peaks(submission_filtered_signal)

    # Divide the data into frames surrounding each peak and store them in a list for both the training and submission signals
    new_training_data, new_testing_data, new_training_index, new_testing_index, new_training_class, new_testing_class = createFrames(training_filtered_signal, index_data, class_data, data_type = 0)
    new_submission_data = createFrames(submission_filtered_signal, submission_peak_locations, class_data, data_type = 1)

    # Create a neural network class called neuronClassificationNetwork
    neuronClassificationNetwork = NeuralNetwork(5, 150, 4, 0.05)

    # Run Principal Component Analysis (PCA) to extract the import bits of data surrounding the peaks
    training_data_normalised, testing_data_normalised, submission_data_normalised = pcaPreclassification(new_training_data, new_testing_data, new_submission_data)

    # Train the neural network on 70% of the  training data
    trainNetwork(training_data_normalised, new_training_index, new_training_class, neuronClassificationNetwork)  
    
    # Test the neural network on the remaining 30% of the training data
    testNetwork(testing_data_normalised, new_testing_index, new_testing_class, neuronClassificationNetwork)

    # Calls the function to classify the submission data set and output a MATLAB file containing the index and classes for the submisison data
    classifySubmissionPeaks(submission_data_normalised, submission_peak_locations, neuronClassificationNetwork)

    return

# ------------------------------------------------- INITIALISE DEFINITION -----------------------------------------------------------

# Definition: initialises static variables used later in the code
# Parameters: none
# Returns: low_cutoff_freq, high_cutoff_freq, sample_rate, training_window_size, submission_window_size, filter_order

def initialise():

    # Filter parameters
    low_cutoff_freq = 50
    high_cutoff_freq = 5000
    sample_rate = 25000
    training_window_size = 25
    submission_window_size = 41
    filter_order = 5

    return low_cutoff_freq, high_cutoff_freq, sample_rate, training_window_size, submission_window_size, filter_order

# ---------------------------------------------------- IMPORT DEFINITION -------------------------------------------------------------

# Definition: imports data from .mat files and converts them into lists for use throughout the code.
# Parameters: none
# Returns: training_data, training_index, training_class, submission_data

def importData():

    # Load the training data and submission data
    training_file = spio.loadmat('training.mat', squeeze_me=True)
    submission_file = spio.loadmat('submission.mat', squeeze_me=True)

    # Import training data signal, peak locations and associated neuron type for each peak
    training_data = training_file['d']
    training_index = training_file['Index']
    training_class = training_file['Class']
    submission_data = submission_file['d']  

    return training_data, training_index, training_class, submission_data

# ------------------------------------------------ FILTER SIGNAL DEFINITION ----------------------------------------------------------

# Definition: bandpass butterworth filters the signal to get rid of noise and then smooths signal using Savitzky-Golay filter
# Parameters: signal, low_cutoff_freq, high_cutoff_freq, sample_rate, filter_window_size, filter_order
# Returns: final_filtered_signal

def filter_signal(signal, low_cutoff_freq, high_cutoff_freq, sample_rate, filter_window_size, filter_order):

    # Calculate Nyquist frequency
    nyq = 0.5 * sample_rate 

    # Normalise low cut off frequency
    low = low_cutoff_freq / nyq 

    # Normalise high cut off frequency
    high = high_cutoff_freq / nyq 

    # Call built in butter function to perform butterworth filter function
    b, a = butter(5, [low, high], btype='band') 

    # filter the signal to produce a DC ofsample_rateet = 0
    bandpass_signal = filtfilt(b, a, signal) 

    # Smooth the signal using Savitzky-Golay filter
    final_filtered_signal = savgol_filter(bandpass_signal, filter_window_size, filter_order)

    return final_filtered_signal

# ---------------------------------------------- FIND FILTERED PEAKS DEFINITION -------------------------------------------------------

# Definition: identifies the locations of peaks from the filtered data
# Parameters: submission_filtered_signal
# Returns: peak_locations

def find_filtered_peaks(submission_filtered_signal):

    # Calculate the standard deviation of the signal to help identify the minimum height of a peak
    std_signal = numpy.std(submission_filtered_signal)

    # Set scaling value
    k = 2.5

    # Identify the locations of peaks 
    peak_locations, _ = find_peaks(submission_filtered_signal, height = k * std_signal)

    return peak_locations

# ------------------------------------------------- CREATE FRAMES DEFINITION  --------------------------------------------------------

# Definition: creates a list of frames surrounding each of the peaks entered
# Parameters: signal, index_data, class_data, data_type
# Returns: new_training_data, new_testing_data, new_training_index, new_testing_index, new_training_class, new_testing_class 
#          or
#          new_submission_data

def createFrames(signal, index_data, class_data, data_type):

    # Define the frame width
    frameWidth = 50

    # Identify what type of data is being split up into frames
    if data_type == 0:
        
        # Split the training indexes and classes according to this ratio
        final_training_data_value = int(0.875 * len(index_data))
        new_training_index = index_data[0 : final_training_data_value]
        new_training_class = class_data[0 : final_training_data_value]
        new_testing_index = index_data[final_training_data_value + 1 :]
        new_testing_class = class_data[final_training_data_value + 1 :]

        # Create an empty 2D array of frames containing the peaks
        new_training_data = numpy.zeros([len(new_training_index), frameWidth])
        new_testing_data = numpy.zeros([len(new_training_index), frameWidth])

        # Populate the arrays with the frame of the signal surrounding the peak
        for i in range (len(new_training_index)):
            new_training_data[i,:] = signal[new_training_index[i] : (new_training_index[i] + frameWidth)]
        for i in range (len(new_testing_index)):
            new_testing_data[i,:] = signal[new_testing_index[i] : (new_testing_index[i] + frameWidth)]

        return new_training_data, new_testing_data, new_training_index, new_testing_index, new_training_class, new_testing_class 

    else:
        # Create an empty 2D array of frames containing the peaks
        new_submission_data = numpy.zeros([len(index_data), frameWidth])

        # Populate the array with the frame of the signal surrounding the peak
        for i in range (len(index_data)):
            new_submission_data[i,:] = signal[index_data[i] - 10 : index_data[i] + 40] 

        return new_submission_data


# ---------------------------------------------------- NETWORK DEFINITION ------------------------------------------------------------

# Definition: initialises a neural network class
# Parameters: input_nodes, hidden_nodes, output_nodes, learning_rate
# Returns: none

class NeuralNetwork:

    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
   
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
   
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
   
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
   
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
   
        return final_outputs

# ----------------------------------------------- PCA PRECLASSIFICATION ---------------------------------------------------------------

# Definition: extracts the principal components of the signals to be passed to the input of the Neural Network
# Parameters: new_training_data, new_testing_data, new_submission_data
# Returns: training_data_normalised, testing_data_normalised, submission_data_normalised

def pcaPreclassification(new_training_data, new_testing_data, new_submission_data):

    # Initialise PCA components
    pca_values = PCA(n_components = 5)

    # Fit the PCA vavlues to the training data - extracts key information
    training_extracted_components = pca_values.fit_transform(new_training_data)
    testing_extracted_components = pca_values.transform(new_testing_data)       
    submission_extracted_components = pca_values.transform(new_submission_data)         

    # Identify variance explained by components
    print("Total Variance Explained: ", numpy.sum(pca_values.explained_variance_ratio_))

    # Normalise the training data so that it can be passed to the Neural Network
    min_max_scaler = MinMaxScaler()
    training_data_normalised = min_max_scaler.fit_transform(training_extracted_components)
    testing_data_normalised = min_max_scaler.fit_transform(testing_extracted_components)
    submission_data_normalised = min_max_scaler.fit_transform(submission_extracted_components)
    
    return training_data_normalised, testing_data_normalised, submission_data_normalised

# -------------------------------------------- TRAIN NEURAL NETWORK DEFINITION --------------------------------------------------------

# Definition: trains the neural network class a set number of times using the normalised pca data as inputs and the neuron type 
#             of each peak as a target
# Parameters: training_data_normalised, new_training_index, new_training_class, network
# Returns: none

def trainNetwork(training_data_normalised, new_training_index, new_training_class, network):

    # Define the number of targets
    number_of_neurons = 4

    for _ in range (5):    
        for record in range (len(new_training_index)):

            # Define target outputs, where target = 0.99, others = 0.01
            targets = numpy.zeros(number_of_neurons) + 0.01

            # Set target as 0.99, depending on what type of neuron the target is
            targets[int(new_training_class[record]) - 1] = 0.99

            # Call the Neural Network function to train network
            network.train(training_data_normalised[record], targets)
        pass    
    pass

    return

# --------------------------------------------- TEST NEURAL NETWORK DEFINITION --------------------------------------------------------

# Definition: tests the neural network on the remaining data 
# Parameters: testing_data_normalised, new_testing_index, new_testing_class, network
# Returns: none

def testNetwork(testing_data_normalised, new_testing_index, new_testing_class, network):
   
    # Define validation variables
    number_of_ones = 0
    number_of_twos = 0
    number_of_threes = 0
    number_of_fours = 0

    # Create a scorecard to assess performance
    scorecard = []

    # Loop through peaks in test data and query the Neural Network
    for record in range (len(new_testing_index)):

        # Identify correct type of peak
        correct_type = int(new_testing_class[record])

        # Query the network for each peak
        outputs = network.query(testing_data_normalised[record])

        # Set the current type to be the highest value in the output (adjust for offset)
        network_type = numpy.argmax(outputs) + 1

        # Record the number of each peak type
        if (network_type == 1): 
            number_of_ones = number_of_ones + 1
        elif (network_type == 2):
            number_of_twos = number_of_twos + 1
        elif (network_type == 3):
            number_of_threes = number_of_threes + 1
        elif (network_type == 4):
            number_of_fours = number_of_fours + 1

        # Display the current neuron type
        print(network_type, "Network Neuron Type")

        # Update scorecard
        if (network_type == correct_type):
            scorecard.append(1)
        else:
            scorecard.append(0)
        pass
    pass
    
   # Display the performance and the total number of the different types of neuron
    scorecard_array = numpy.asarray(scorecard)
    print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')
    print("Number of Ones =", number_of_ones, "Number of Twos =", number_of_twos, "Number of Threes =", number_of_threes, "Number of Fours =", number_of_fours)

    return

# ------------------------------------------------- CLASSIFIER DEFINITION -------------------------------------------------------------

# Definition: runs through the submission data and classifies each identified peak and stores the data in a .mat file
# Parameters: submission_data_normalised, submission_peak_locations, network
# Returns: none

def classifySubmissionPeaks(submission_data_normalised, submission_peak_locations, network):

    # Creates variables to keep track of how many of each type of spike has been classified and a scorecard array to display the percentage after the query has been run
    number_of_ones = 0
    number_of_twos = 0
    number_of_threes = 0
    number_of_fours = 0

    # Create an array to store the results of the classification
    submission_class = []

    # Loop through peaks in submission data and query the Neural Network
    for i in range(len(submission_peak_locations)):

        # Query the network for each peak
        output = network.query(submission_data_normalised[i])

        # Set the current type to be the highest value in the output (adjust for offset)
        network_type_submission = numpy.argmax(output) + 1

        # Insert peak type into array
        submission_class = numpy.append(submission_class, network_type_submission)

        # Record the number of each peak type
        if (network_type_submission == 1): 
            number_of_ones = number_of_ones + 1
        elif (network_type_submission == 2):
            number_of_twos = number_of_twos + 1
        elif (network_type_submission == 3):
            number_of_threes = number_of_threes + 1
        elif (network_type_submission == 4):
            number_of_fours = number_of_fours + 1
    pass
   
    print("Number of Ones =", number_of_ones, "Number of Twos =", number_of_twos, "Number of Threes =", number_of_threes, "Number of Fours =", number_of_fours)

    # Set the index of the peaks to the input value to the function
    submission_index = submission_peak_locations

    # Generate a .mat file containing the index of each peak and the type of neuron represented by the peak
    generateOutputFile(submission_index, submission_class)

    return

# --------------------------------------------- GENERATE OUTPUT DEFINITION ------------------------------------------------------------

# Definition: writes the index of each located peak and the tpye of neuron to a .mat file
# Parameters: submission_index, submission_class
# Returns: none

def generateOutputFile(submission_index, submission_class):

    # Save data into .mat file
    spio.savemat("11236", dict([("Index", submission_index), ("Class", submission_class)]))

    return

# -------------------------------------------------------------------------------------------------------------------------------------

# Run the main function to execute code
main()