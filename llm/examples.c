#include <stdio.h>

int main() {
    // Define the given values
    int B = 2; // Batch size
    int T = 3; // Sequence length
    int C = 4; // Embedding dimensionality

    // Initialize the out array
    float out[2 * 3 * 4] = {
        // Batch 1, time step 1
        1.0, 2.0, 3.0, 4.0,
        // Batch 1, time step 2
        5.0, 6.0, 7.0, 8.0,
        // Batch 1, time step 3
        9.0, 10.0, 11.0, 12.0,
        // Batch 2, time step 1
        13.0, 14.0, 15.0, 16.0,
        // Batch 2, time step 2
        17.0, 18.0, 19.0, 20.0,
        // Batch 2, time step 3
        21.0, 22.0, 23.0, 24.0
    };

    // Calculate b and t for the position we want to access
    int b = 1; // Second batch
    int t = 2; // Third time step

    // Calculate the pointer to the desired output position
    float* out_bt = out + b * T * C + t * C;

    // Print the value at the output position
    printf("Output at batch %d, time step %d: %f\n", b, t, *out_bt);

    return 0;
}
