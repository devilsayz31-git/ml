// Chit 3 (Set A): Gradient Descent to find the local minimum of y = (x + 3)^2, starting from x0 = 2
// Compile: g++ chit03a_gradient_descent.cpp -o gd
// Run    : ./gd
#include <iostream>
#include <cmath>
using namespace std;

// Function to calculate y
double calculateCost(double x) {
    return (x + 3) * (x + 3);
}

// Derivative dy/dx
double calculateGradient(double x) {
    return 2 * (x + 3);
}

int main() {
    // --------------------------------
    // 1. Initialize parameter
    // --------------------------------
    double x = 2.0;

    // --------------------------------
    // 2. Hyperparameters
    // --------------------------------
    double learningRate = 0.1;
    int epochs = 100;

    // --------------------------------
    // 3. Gradient Descent
    // --------------------------------
    for (int epoch = 0; epoch < epochs; epoch++) {
        double dx = calculateGradient(x);

        // Update parameter
        x = x - learningRate * dx;

        // Print progress
        if (epoch % 10 == 0) {
            cout << "Epoch: " << epoch
                 << " | x: " << x
                 << " | y: " << calculateCost(x)
                 << endl;
        }
    }

    // --------------------------------
    // 4. Final Result
    // --------------------------------
    cout << "\n============================\n";
    cout << "Training Completed\n";
    cout << "============================\n";
    cout << "Local minimum at x = " << x << endl;
    cout << "Minimum value y = " << calculateCost(x) << endl;

    return 0;
}
