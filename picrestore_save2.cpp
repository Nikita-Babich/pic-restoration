//Frequently used
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

//#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

//Task specific
#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <cfloat>

//project specific
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(){
	// Load the image
    const char* filename = "panda.png";
    int width, height, channels;
    unsigned char *image = stbi_load(filename, &width, &height, &channels, 0);
    if (!image) { printf("Error loading image\n"); return 1; }
    printf("Image dimensions: %d x %d, channels: %d\n", width, height, channels);
    
    // Create a width by height matrix for grayscale values
    Eigen::MatrixXd grayscaleMatrix(width, height);

    // Calculate grayscale values for each pixel
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            unsigned char red = image[(j * width + i) * channels];       // Red channel
            unsigned char green = image[(j * width + i) * channels + 1]; // Green channel
            unsigned char blue = image[(j * width + i) * channels + 2];  // Blue channel
            // Calculate grayscale value using luminosity method
            double grayscale = 0.21 * red + 0.72 * green + 0.07 * blue;
            if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
                if (rand() % 5 != 0) { grayscale = 255; } // White color
            }
            grayscaleMatrix(i, j) = grayscale;
        }
    }
    // Print some information about the grayscale matrix
    std::cout << "Grayscale matrix dimensions: " << grayscaleMatrix.rows() << " x " << grayscaleMatrix.cols() << std::endl;
	
	// Create a new image buffer
	unsigned char* outputImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	if (!outputImage) {
	    std::cerr << "Error allocating memory for output image" << std::endl;
	    return 1;
	}
	
	// Convert grayscale matrix to image buffer
	for (int i = 0; i < width; ++i) {
	    for (int j = 0; j < height; ++j) {
	        double grayscale = grayscaleMatrix(i, j);
	        unsigned char value = static_cast<unsigned char>(std::round(grayscale)); // Convert double to unsigned char
	        outputImage[j * width + i] = value; // Store pixel value in output image buffer
	    }
	}
	
	// Save outputImage as a new PNG file
	if (!stbi_write_png("grayscale_image.png", width, height, 1, outputImage, 0)) {
	    std::cerr << "Error saving grayscale image" << std::endl;
	    free(outputImage); // Free allocated memory
	    return 1;
	}
	
	// Free allocated memory
	free(outputImage);
	
    // Create a sparse matrix
    Eigen::SparseMatrix<float> sparseMatrix(width * height, width * height);
    
    // Reserve memory for non-zero elements
    sparseMatrix.reserve(Eigen::VectorXi::Constant(width * height, 5));

	// Cycle through each pixel of the grayscale matrix
	for (int i = 0; i < width; ++i) {
	    for (int j = 0; j < height; ++j) {
	        // Check if the pixel is not white (known)
	        if (grayscaleMatrix(i, j) <= 254) {
	            // Set the corresponding element in the sparse matrix to 1.0
	            sparseMatrix.coeffRef(j * width + i, j * width + i) = 1.0;
	        } else {
	            // Pixel is white (unknown), so approximate its value using Laplace method
	            // Place 4 on its diagonal
	            sparseMatrix.coeffRef(j * width + i, j * width + i) = 4.0;
	
	            // Check and set neighbors
	            // Left neighbor
	            if (i > 0) sparseMatrix.coeffRef(j * width + i, j * width + i - 1) = -1.0;
	            // Right neighbor
	            if (i < width - 1) sparseMatrix.coeffRef(j * width + i, j * width + i + 1) = -1.0;
	            // Top neighbor
	            if (j > 0) sparseMatrix.coeffRef(j * width + i, (j - 1) * width + i) = -1.0;
	            // Bottom neighbor
	            if (j < height - 1) sparseMatrix.coeffRef(j * width + i, (j + 1) * width + i) = -1.0;
	        }
	    }
	}
	
	//------------------------
	
	
	// Create vector b
	Eigen::VectorXd b(width * height);
	
	// Cycle through each pixel of the grayscale matrix
	for (int i = 0; i < width; ++i) {
	    for (int j = 0; j < height; ++j) {
	        // Calculate the index in the vector b
	        int index = j * width + i;
	        // Check if the pixel value is known (not white)
	        if (grayscaleMatrix(i, j) <= 254) {
	            // Add the grayscale value to the vector b
	            b(index) = grayscaleMatrix(i, j);
	        } else {
	            // Pixel value is unknown (white), set b to 0
	            b(index) = 0.0;
	        }
	    }
	}
	
	//-----------------
	
	// Solve the linear system Ax = b
	// Create a SparseLU solver object
    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;

    // Convert the dense vector 'b' to a sparse vector
	Eigen::SparseVector<float> b_sparse(width * height);
	for (int i = 0; i < width * height; ++i) {
	    if (b(i) != 0.0) {
	        b_sparse.coeffRef(i) = b(i);
	    }
	}

	// Compute the factorization of the sparse matrix
	solver.compute(sparseMatrix);

	// Solve the linear system Ax = b
	Eigen::VectorXf x = solver.solve(b_sparse);

    // Check if the solution was successful
    if (solver.info() != Eigen::Success) {
        std::cerr << "Failed to solve the linear system" << std::endl;
        return 1;
    }
	
	// Create a matrix to hold the restored image
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> restoredMatrix(height, width);
	
	// Iterate over the elements of the vector x and populate the restoredMatrix
	for (int i = 0; i < width; ++i) {
	    for (int j = 0; j < height; ++j) {
	        // Calculate the index in the vector x
	        int index = j * width + i;
	        // Set the corresponding pixel value in the restoredMatrix
	        restoredMatrix(j, i) = static_cast<unsigned char>(std::round(x(index)));
	    }
	}
	
	// Save the restored image matrix as a new PNG file
	if (!stbi_write_png("result_panda.png", width, height, 1, restoredMatrix.data(), 0)) {
	    std::cerr << "Error saving restored image" << std::endl;
	    return 1;
	}


	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> restoredMatrixTransposed = restoredMatrix.transpose();

// Save the transposed restored image matrix as a new PNG file
if (!stbi_write_png("result_panda_trnsposed.png", width, height, 1, restoredMatrixTransposed.data(), 0)) {
    std::cerr << "Error saving restored image" << std::endl;
    return 1;
}

    // Free image memory
    stbi_image_free(image);
    
}

