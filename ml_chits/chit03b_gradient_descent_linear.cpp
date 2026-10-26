// Chit 3 (Set B): Gradient Descent Algorithm for Cost Function Minimization (Linear Regression, MSE)
// Compile: g++ chit03b_gradient_descent_linear.cpp -o gd
// Run    : ./gd
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Function to calculate Mean Squared Error
double calculateCost(
    const vector<double>& x,
    const vector<double>& y,
    double w,
    double b
) {
    double cost = 0.0;
    int n = x.size();
    for (int i = 0; i < n; i++) {
        // Prediction
        double prediction = w * x[i] + b;
        // Error
        double error = prediction - y[i];
        // Squared error
        cost += error * error;
    }
    return cost / n;
}

int main() {
    // --------------------------------
    // 1. Training Dataset
    // --------------------------------
    vector<double> x = {1, 2, 3, 4, 5};
    vector<double> y = {2, 4, 6, 8, 10};

    // --------------------------------
    // 2. Initialize parameters
    // --------------------------------
    double w = 0.0;
    double b = 0.0;

    // --------------------------------
    // 3. Hyperparameters
    // --------------------------------
    double learningRate = 0.01;
    int epochs = 1000;
    int n = x.size();

    // --------------------------------
    // 4. Gradient Descent
    // --------------------------------
    for (int epoch = 0; epoch < epochs; epoch++) {
        double dw = 0.0;
        double db = 0.0;

        // Calculate gradients
        for (int i = 0; i < n; i++) {
            double prediction = w * x[i] + b;
            double error = prediction - y[i];
            dw += (2.0 / n) * error * x[i];
            db += (2.0 / n) * error;
        }

        // Update parameters
        w = w - learningRate * dw;
        b = b - learningRate * db;

        // Print progress
        if (epoch % 100 == 0) {
            double cost = calculateCost(x, y, w, b);
            cout << "Epoch: " << epoch
                 << " | Cost: " << cost
                 << " | w: " << w
                 << " | b: " << b
                 << endl;
        }
    }

    // --------------------------------
    // 5. Final Model
    // --------------------------------
    cout << "\n============================\n";
    cout << "Training Completed\n";
    cout << "============================\n";
    cout << "Weight (w): " << w << endl;
    cout << "Bias (b): " << b << endl;
    cout << "\nFinal equation:\n";
    cout << "y = " << w << "x + " << b << endl;

    // --------------------------------
    // 6. Make a prediction
    // --------------------------------
    double input;
    cout << "\nEnter a value of x: ";
    cin >> input;
    double prediction = w * input + b;
    cout << "Predicted y = " << prediction << endl;

    return 0;
}
