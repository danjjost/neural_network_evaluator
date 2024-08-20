# ℹ️ About

This project holds an Azure Function that, given an uploaded Blob containing a neural network (weights, biases), runs a predetermined number of training cases, and uploads a blob containing a trained neural network.

# 🥅 Goal

Training neural networks locally on my laptop takes a fairly long time. To significantly improve performance, I can parallelize the operations by leveraging more CPU power in the cloud.

# 🏗️ Flow

1. A training epoch begins on the host application for a population of neural networks
2. The host application uploads each neural network of the population to blob storage, which triggers an eventgrid
3. The event grid kicks off an Azure Function for each uploaded neural network, and runs a tranining cycle for that network
4. Each Azure Function, when training completes, uploads an output blob to blob storage
5. The host machine, now sitting idle, will query blob storage every few seconds for newly uploaded blobs in the population
6. Once all blobs in the population have been processed, or a timeout period has been reached, the host will delete all records in the '/in' and '/out' folders, then proceed with its pipeline

# 🗒️ Blob Storage Structure

neuralnetworkevaluator

    Containers
        /input - input neural networks
        /output - output neural networks
