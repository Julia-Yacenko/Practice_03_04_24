#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace cv;
using namespace std;

int mandelbrot(double cr, double ci, int max_iterations) 
{
    double zr = 0.0, zi = 0.0;
    int iterations = 0;

    while (zr * zr + zi * zi <= 4.0 && iterations < max_iterations) {
        double temp = zr * zr - zi * zi + cr;
        zi = 2 * zr * zi + ci;
        zr = temp;
        iterations++;
    }

    return iterations;
}

Vec3b color(int iterations) 
{
    if (iterations >= 255) {
        return Vec3b(0, 0, 0);
    }
    else {
        return Vec3b(127 + 63 * sin(0.1 * iterations), 127 + 63 * sin(0.4 * iterations), 127 + 63 * sin(0.3 * iterations));
    }
}

int main(int* argc, char** argv) 
{
    MPI_Init(argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 1000;
    int height = 1000;
    int max_iterations = 800;
    double xmin = -1.3;
    double xmax = -1.05;
    double ymin = -0.5;
    double ymax = -0.25;

    int start = rank * (height / size);
    int end = min((rank + 1) * (height / size), height);

    Mat fractal(end - start, width, CV_8UC3, Scalar(0, 0, 0));

    for (int j = start; j < end; j++) {
        for (int i = 0; i < width; i++) {
            double x = xmin + (xmax - xmin) * i / width;
            double y = ymin + (ymax - ymin) * j / height;

            int iterations = mandelbrot(x, y, max_iterations);

            fractal.at<Vec3b>(j - start, i) = color(iterations);
        }
    }

    Mat final_fractal(height, width, CV_8UC3);

    MPI_Gather(fractal.data, (end - start) * width * 3, MPI_UNSIGNED_CHAR, 
        final_fractal.data, (end - start) * width * 3, MPI_UNSIGNED_CHAR, 
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        imshow("Fractal of Mandelbrot", final_fractal);
        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}