#include <armadillo>
#include <complex>
#include <fftw3.h>
#include <iomanip> // For std::setprecision
#include <iostream>
#include <cstring> // For std::memcpy

void fft_rows(std::complex<double>* input, std::complex<double>* output, int M1, int M2) {
    int rank = 1;                   // 1D FFT
    int n[] = {M2};                 // Length of each 1D FFT
    int howmany = M1;               // Number of 1D FFTs (one per row)
    int istride = 1, ostride = 1;   // Contiguous elements in each row
    int idist = M2, odist = M2;     // Distance between consecutive rows

    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL,
        istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL,
        ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void fft_columns(std::complex<double>* input, std::complex<double>* output, int M1, int M2) {
    int rank = 1; // 1D FFT
    int n[] = {M1}; // Length of each FFT is M1 (number of rows)
    int howmany = M2; // Number of FFTs is M2 (number of columns)
    int istride = M2, ostride = M2; // Stride for column-wise FFT
    int idist = 1, odist = 1; // Distance between start of each column FFT

    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL,
        istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL,
        ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
void print_complex_matrix(const std::complex<double>* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(2)
                      << "(" << data[i * cols + j].real() << ", " << data[i * cols + j].imag() << ") ";
        }
        std::cout << std::endl;
    }
}


void parallel_fft_rows(std::complex<double>* input, std::complex<double>* output, int M1, int M2, int parallel_num)
{
    int rank = 1;                   // 1D FFT
    int n[] = {M2};                 // Length of each 1D FFT
    int howmany = M1;               // Number of 1D FFTs (one per row)
    int istride = 1, ostride = 1;   // Contiguous elements in each row
    int idist = M2, odist = M2;     // Distance between consecutive rows
    // Initialize FFTW threads (only needs to be done once in your program)
    static bool threads_initialized = false;
    if (!threads_initialized) {
        fftw_init_threads();
        threads_initialized = true;
    }

    // Set the number of threads to use
    fftw_plan_with_nthreads(parallel_num);

    // Create the plan
    // Note: std::complex<double> is binary compatible with fftw_complex
    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftw_complex*>(input), NULL, istride, idist,
        reinterpret_cast<fftw_complex*>(output), NULL, ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    // Execute the plan
    fftw_execute(plan);

    // Clean up the plan
    fftw_destroy_plan(plan);

    // Note: We don't call fftw_cleanup_threads() here because that would
    // terminate threading support. Only call it at the end of your program.

}
int main(int argc, char* argv[]) {

    // Set problem dimensions
    const int M1 = 270;  // Number of rows
    const int M2 = 600;  // Number of columns (FFT size)

    int num_threads = 20;
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
    }
    std::cout << "Running with " << num_threads << " threads" << std::endl;
    std::cout << "Matrix dimensions: " << M1 << " x " << M2 << std::endl;

    // Allocate aligned memory for input and output
    fftw_complex* fftw_input = fftw_alloc_complex(M1 * M2);
    fftw_complex* fftw_output = fftw_alloc_complex(M1 * M2);
    fftw_complex* fftw_output_serial = fftw_alloc_complex(M1 * M2);

    // Cast to std::complex for easier manipulation
    std::complex<double>* input = reinterpret_cast<std::complex<double>*>(fftw_input);
    std::complex<double>* output = reinterpret_cast<std::complex<double>*>(fftw_output);
    std::complex<double>* output_serial = reinterpret_cast<std::complex<double>*>(fftw_output_serial);

    // Initialize input data with a simple pattern
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < M2; j++) {
            // Simple sinusoidal pattern with different frequencies per row
            double value = std::sin(2.0 * M_PI * j * (i % 10 + 1) / M2);
            input[i * M2 + j] = std::complex<double>(value, 0.0);
        }
    }
    // Measure time for parallel FFT
    auto start_time = std::chrono::high_resolution_clock::now();

    parallel_fft_rows(input, output, M1, M2, num_threads);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Parallel FFT completed in " << parallel_duration << " ms" << std::endl;
    //end paralell

    //serial fft
    start_time = std::chrono::high_resolution_clock::now();
    fft_rows(input,output_serial,M1,M2);
    end_time = std::chrono::high_resolution_clock::now();
    auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Serial FFT completed in " << serial_duration << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(serial_duration) / parallel_duration << "x" << std::endl;


    //end serial
    fftw_free(fftw_input);
    fftw_free(fftw_output);
    // Cleanup FFTW threads at the end of the program
    fftw_cleanup_threads();
    fftw_cleanup();
    return 0;
}