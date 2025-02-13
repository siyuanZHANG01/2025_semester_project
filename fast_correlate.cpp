#include <iostream>
#include <vector>
#include <complex>
#include <fftw3.h> 

extern "C" {
    void fft_correlate(double* x, double* y, double* result, int N) {
        int fft_size = 1;
        while (fft_size < 2 * N - 1) fft_size *= 2;

        int fft_complex_size = fft_size / 2 + 1; 

        double *x_padded = (double*) fftw_malloc(sizeof(double) * fft_size);
        double *y_padded = (double*) fftw_malloc(sizeof(double) * fft_size);
        fftw_complex *X = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_complex_size);
        fftw_complex *Y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_complex_size);
        fftw_complex *Z = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_complex_size);
        double *ifft_out = (double*) fftw_malloc(sizeof(double) * fft_size);

        if (!x_padded || !y_padded || !X || !Y || !Z || !ifft_out) {
            std::cerr << "Error: FFTW memory allocation failed!" << std::endl;
            return;
        }

        for (int i = 0; i < N; i++) {
            x_padded[i] = x[i];
            y_padded[i] = y[N - 1 - i];
        }
        for (int i = N; i < fft_size; i++) {  
            x_padded[i] = 0.0;
            y_padded[i] = 0.0;
        }

        fftw_plan plan_x = fftw_plan_dft_r2c_1d(fft_size, x_padded, X, FFTW_ESTIMATE);
        fftw_plan plan_y = fftw_plan_dft_r2c_1d(fft_size, y_padded, Y, FFTW_ESTIMATE);
        fftw_plan plan_z = fftw_plan_dft_c2r_1d(fft_size, Z, ifft_out, FFTW_ESTIMATE);

        fftw_execute(plan_x);
        fftw_execute(plan_y);

        for (int i = 0; i < fft_complex_size; i++) {
            Z[i][0] = X[i][0] * Y[i][0] - X[i][1] * Y[i][1];  
            Z[i][1] = X[i][1] * Y[i][0] + X[i][0] * Y[i][1];  
        }

        fftw_execute(plan_z);

        for (int i = 0; i < 2 * N - 1; i++) {
            result[i] = ifft_out[i]/ fft_size ;
        }
        
        fftw_destroy_plan(plan_x);
        fftw_destroy_plan(plan_y);
        fftw_destroy_plan(plan_z);
        fftw_free(x_padded);
        fftw_free(y_padded);
        fftw_free(X);
        fftw_free(Y);
        fftw_free(Z);
        fftw_free(ifft_out);
    }
}
