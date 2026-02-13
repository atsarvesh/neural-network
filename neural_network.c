#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define EPOCHS 10000

// neural network structure

typedef struct{

    // layer 1: input to hidden

    double weights[INPUT_SIZE][HIDDEN_SIZE]; // 4 weights for each of the 3 input SIZE (4x3)
    double biases[HIDDEN_SIZE]; // 4 biases

    // layer 2: hidden to output

    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE]; // 1 weight for each of the 4 hidden SIZE (1x4)
    double output_biases[OUTPUT_SIZE]; // 1 bias

    // activations (for backpropagation)

    double hidden_layer[HIDDEN_SIZE]; // after ReLU
    double output_layer[OUTPUT_SIZE]; // final output (after linear activation)

    // preactivations

    double z_hidden[HIDDEN_SIZE]; // before ReLU
    double z_output[OUTPUT_SIZE]; // before output
} NeuralNetwork;

// ReLU activation function: max(0, x)

double relu(double x)
{
    return x > 0 ? x : 0;
}

// ReLu derivative: 1 if x > 0, else 0

double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

// weights initialization

void init_network(NeuralNetwork *nn)
{
    srand(time(NULL)); // generates random numbers based on current time

    // initialize weights and biases for input -> hidden layer

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // random weights between -1 and 1
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        nn->biases[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // initialize weights and biases for hidden -> output layer

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            nn->output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        nn->output_biases[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

// forward propagation

void forward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE])
{
    // input -> hidden layer

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        nn->z_hidden[j] = nn->biases[j]; // start with bias

        for (int i = 0; i < INPUT_SIZE; i++)
        {
            nn->z_hidden[j] += input[i] * nn->weights[i][j]; // weighted sum: z = w.x + b
        }

        nn->hidden_layer[j] = relu(nn->z_hidden[j]); // activation: a = ReLU(z)
    }

    // hidden -> output layer

    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        nn->z_output[j] = nn->output_biases[j]; // start with bias

        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            nn->z_output[j] += nn->hidden_layer[i] * nn->output_weights[i][j]; // weighted sum
        }

        nn->output_layer[j] = nn->z_output[j]; // linear activation (for regression)
    }
}

// backpropagation

void backpropagation(NeuralNetwork *nn, double input[INPUT_SIZE], double target[OUTPUT_SIZE])
{
    // calculate output layer error (mean squared error)

    double output_error[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        output_error[j] = nn->output_layer[j] - target[j]; // error = output - target
    }

    // calculate hidden layer error

    double hidden_error[HIDDEN_SIZE];

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_error[i] = 0.0;

        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            hidden_error[i] += output_error[j] * nn->output_weights[i][j]; // backpropagate error
        }

        hidden_error[i] *= relu_derivative(nn->z_hidden[i]); // apply ReLU derivative
    }

    // update weights and biases for hidden -> output layer

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            nn->output_weights[i][j] -= LEARNING_RATE * output_error[j] * nn->hidden_layer[i]; // weight update
        }
    }

    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        nn->output_biases[j] -= LEARNING_RATE * output_error[j]; // bias update
    }

    // update weights and biases for input -> hidden layer

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->weights[i][j] -= LEARNING_RATE * hidden_error[j] * input[i]; // weight update
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        nn->biases[j] -= LEARNING_RATE * hidden_error[j]; // bias update
    }
}

// data normalization

void normalize(double data[][INPUT_SIZE], int samples, double min[INPUT_SIZE], double max[INPUT_SIZE])
{
    for (int j = 0; j < INPUT_SIZE; j++)
    {
        min[j] = data[0][j];
        max[j] = data[0][j];

        // find min and max for each feature

        for (int i = 1; i < samples; i++)
        {
            if (data[i][j] < min[j])
                min[j] = data[i][j];
            if (data[i][j] > max[j])
                max[j] = data[i][j];
        }

        // normalize to [0, 1]

        for (int i = 0; i < samples; i++)
        {
            if (max[j] - min[j] != 0)
                data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
            else
                data[i][j] = 0; // if all values are the same, set to 0
        }
    }
}

// training function

void train(NeuralNetwork *nn, double data[][INPUT_SIZE], double targets[][OUTPUT_SIZE], int samples)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;

        // training loop for each sample
        for (int i = 0; i < samples; i++)
        {
            forward_propagation(nn, data[i]); // forward pass: makes predictions

            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                double error = nn->output_layer[j] - targets[i][j];
                total_loss += error * error; // accumulate loss (MSE)
            }
            backpropagation(nn, data[i], targets[i]); // backward pass: updates weights and biases
        }

        total_loss /= samples; // average loss over all samples

        if (epoch % 1000 == 0)
        {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss); // print progress
        }
    }
}

// main function

int main()
{
    // dataset: rooms, area_sqft, distance_km 

    double training_data[][INPUT_SIZE] = {
       {6, 3000, 0.5},
       {8, 4000, 0.7},
       {5, 2000, 0.3},
       {7, 3500, 0.6},
       {9, 4500, 0.8},
       {4, 1500, 0.2},
       {10, 5000, 0.9},
       {3, 1000, 0.1},
       {11, 5500, 1.0},
       {2, 500, 0.05}};

    // target prices in rupees

    double training_targets[][OUTPUT_SIZE] = {
        {500000},
        {700000},
        {300000},
        {600000},
        {800000},
        {200000},
        {900000},
        {100000},
        {1000000},
        {50000}};
    
    int samples = sizeof(training_data) / sizeof(training_data[0]);

    // normalize input data

    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize(training_data, samples, min, max);

    // normalize targets

    double target_min = training_targets[0][0];
    double target_max = training_targets[0][0];

    for (int i = 1; i < samples; i++)
    {
        if (training_targets[i][0] < target_min)
            target_min = training_targets[i][0];
        if (training_targets[i][0] > target_max)
            target_max = training_targets[i][0];
    }

    for (int i = 0; i < samples; i++)
    {
        if (target_max - target_min != 0)
            training_targets[i][0] = (training_targets[i][0] - target_min) / (target_max - target_min);
        else
            training_targets[i][0] = 0; // if all values are the same, set to 0
    }

    // initialize neural network

    NeuralNetwork nn;
    init_network(&nn);

    printf("Training Neural Network...\n");
    printf("Architecture: %d input neurons, %d hidden neurons, %d output neurons\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // train the neural network

    train(&nn, training_data, training_targets, samples);

    // test predictions on training data

    printf("Predictions:\n");
    for (int i = 0; i < samples; i++)
    {
        forward_propagation(&nn, training_data[i]);

        // denormalize: value = output * (max - min) + min
        double predicted_price = nn.output_layer[0] * (target_max - target_min) + target_min;
        double actual_price = training_targets[i][0] * (target_max - target_min) + target_min;

        printf("Sample %d: Predicted Price: %.2f, Actual Price: %.2f\n", i + 1, predicted_price, actual_price);

    }
    
    // predict on new data
    double new_data[3] = {7, 3200, 0.4};

    // normalize new data using training data min and max
    for (int j = 0; j < INPUT_SIZE; j++){
        if (max[j] - min[j] != 0)
            new_data[j] = (new_data[j] - min[j]) / (max[j] - min[j]);
        else
            new_data[j] = 0; // if all values are the same, set to 0
    }

    forward_propagation(&nn, new_data);
    double predicted_price = nn.output_layer[0] * (target_max - target_min) + target_min;
    
    printf("New Data: %.1f rooms, %.0f sqft, %.1f km\n", new_data[0] * (max[0] - min[0]) + min[0], new_data[1] * (max[1] - min[1]) + min[1], new_data[2] * (max[2] - min[2]) + min[2]);
    printf("Predicted Price: %.2f\n", predicted_price);

    return 0;
}