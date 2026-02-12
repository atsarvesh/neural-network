#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 3
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.0001
#define EPOCHS 10000

// structure of the neural network

typedef struct
{

    // layer 1: input to hidden

    double weights[INPUT_NEURONS][HIDDEN_NEURONS];
    double biases[HIDDEN_NEURONS];

    // layer 2: hidden to output

    double output_weights[HIDDEN_NEURONS][OUTPUT_NEURONS];
    double output_biases[OUTPUT_NEURONS];

    // activations (for backpropagation)

    double hidden_layer[HIDDEN_NEURONS];
    double output_layer[OUTPUT_NEURONS];

    // preactivations

    double z_hidden[HIDDEN_NEURONS];
    double z_output[OUTPUT_NEURONS];
} NeuralNetwork;

// activation function and its derivative

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}