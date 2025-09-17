//
// Created by adada on 27/5/2025.
//

#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP
#include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cmath>
#include <complex>

#include <cstdio>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
const auto PI = M_PI;

namespace fs = boost::filesystem;
using namespace std::complex_literals; // Brings in the i literal
namespace bp = boost::python;
namespace np = boost::python::numpy;
//This subroutine computes evolution using operator splitting
//one step is exact solution of quasi-linear pde
// uses parallelism to accelerate
class evolution
{
public:
    evolution(const std::string& cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        boost::filesystem::path filePath(cppInParamsFileName);
        this->inCppFileDir= filePath.parent_path().string();
        out_wvfunction_dir=inCppFileDir+"/wavefunction/";
        if (!fs::is_directory(out_wvfunction_dir) || !fs::exists(out_wvfunction_dir))
        {
            fs::create_directories(out_wvfunction_dir);
        }
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty())
            {
                continue; // Skip empty lines
            }

            std::istringstream iss(line);
            //read j1H
            if (paramCounter == 0)
            {
                iss >> j1H;
                if (j1H < 0)
                {
                    std::cerr << "j1H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            } //end reading j1H

            //read j2H
            if (paramCounter == 1)
            {
                iss >> j2H;
                if (j2H < 0)
                {
                    std::cerr << "j2H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }
            //end reading j2H
            //read g0
            if (paramCounter == 2)
            {
                iss >> g0;
                paramCounter++;
                continue;
            }
            //end reading g0
            //read omegam
            if (paramCounter == 3)
            {
                iss >> omegam;
                paramCounter++;
                continue;
            } //end reading omegam

            //read omegap
            if (paramCounter == 4)
            {
                iss >> omegap;
                paramCounter++;
                continue;
            }
            //end reading omegap
            //read omegac
            if (paramCounter == 5)
            {
                iss >> omegac;
                paramCounter++;
                continue;
            } //end reading omegac
            //read er
            if (paramCounter == 6)
            {
                iss >> er;
                if (er <= 0)
                {
                    std::cerr << "er must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }
            //end reading er
            //read thetaCoef
            if (paramCounter == 7)
            {
                iss >> thetaCoef;

                paramCounter++;
                continue;
            }
            //end reading thetaCoef
            //read groupNum
            if (paramCounter == 8)
            {
                iss >> groupNum;
                paramCounter++;
                continue;
            } //end groupNum

            //read rowNum
            if (paramCounter == 9)
            {
                iss >> rowNum;
                paramCounter++;
                continue;
            } //end rowNum
            // read parallel_num
            if (paramCounter == 10)
            {
                iss >> parallel_num;
                paramCounter++;
                continue;
            }//end parallel_num

            //read tTot
            if (paramCounter == 11)
            {
                iss>>tTot;
                paramCounter++;
                continue;
            }//end tTot

            //read Q
            if (paramCounter == 12)
            {
                iss>>Q;
                paramCounter++;
                continue;
            }//end Q

            //read toWrite
            if (paramCounter == 13)
            {
                iss>>toWrite;
                paramCounter++;
                continue;

            }//end toWrite

        } //end while

        //print parameters
        std::cout<<"inCppFileDir="<<inCppFileDir<<std::endl;
        std::cout << std::setprecision(15);
        std::cout << "j1H=" << j1H << ", j2H=" << j2H << ", g0=" << g0
            << ", omegam=" << omegam << ", omegap=" << omegap << ", omegac=" << omegac
            << ", er=" << er << ", thetaCoef=" << thetaCoef << ", groupNum="
            << groupNum << ", rowNum=" << rowNum << ", parallel_num=" << parallel_num
        <<", toWrite="<<toWrite<< std::endl;
        this->L1 = 0.5;
        this->L2 = 1;
        this->r = std::log(er);
        this->theta = thetaCoef * PI;
        this->Deltam = omegam - omegap;
        std::cout << "Deltam=" << Deltam << std::endl;
        this->e2r = std::pow(er, 2.0);

        this->lmd = (e2r - 1 / e2r) / (e2r + 1 / e2r) * Deltam;
        std::cout << "lambda=" << lmd << std::endl;

        this->D = std::pow(lmd * std::sin(theta), 2.0) + std::pow(omegap, 2.0);
        this->mu = lmd * std::cos(theta) + Deltam;
        std::cout << "D=" << D << std::endl;
        std::cout << "mu=" << mu << std::endl;
        double height1 = 0.5;
        double width1 = std::pow(-2.0 * std::log(height1) / omegac, 0.5);
        double minGrid1 = width1 / 10.0;
        this->N2 = 300;


        this->N1 = static_cast<int>(std::ceil(L1 * 2.0 / minGrid1));
        if (N1 % 2 == 1)
        {
            N1 += 1;
        }
        N1=270;
        N2=300;
        std::cout << "L1=" << L1 << ", L2=" << L2 << std::endl;
        std::cout << "N1=" << N1 << std::endl;
        std::cout << "N2=" << N2 << std::endl;

        dx1 = 2.0 * L1 / static_cast<double>(N1);
        dx2 = 2.0 * L2 / static_cast<double>(N2);
        std::cout << "dx1=" << dx1 << std::endl;
        std::cout << "dx2=" << dx2 << std::endl;

        for (int n1 = 0; n1 < N1; n1++)
        {
            this->x1ValsAll.push_back(-L1 + dx1 * n1);
        }
        for (int n2 = 0; n2 < N2; n2++)
        {
            this->x2ValsAll.push_back(-L2 + dx2 * n2);
        }
        for (const auto& val : x1ValsAll)
        {
            x1ValsAllSquared.push_back(std::pow(val, 2));
        }
        for (const auto& val : x2ValsAll)
        {
            x2ValsAllSquared.push_back(std::pow(val, 2));
        }
        for (int n1 = 0; n1 < static_cast<int>(N1 / 2); n1++)
        {
            k1ValsAll_fft.push_back(2.0 * PI * static_cast<double>(n1) / (2.0 * L1));
        }
        for (int n1 = static_cast<int>(N1 / 2); n1 < N1; n1++)
        {
            k1ValsAll_fft.push_back(2.0 * PI * static_cast<double>(n1 - N1) / (2.0 * L1));
        }


        for (const auto& val : k1ValsAll_fft)
        {
            k1ValsAllSquared_fft.push_back(std::pow(val, 2));
        }
        for (int n2 = 0; n2 < static_cast<int>(N2 / 2); n2++)
        {
            k2ValsAll_fft.push_back(2.0 * PI * static_cast<double>(n2) / (2.0 * L2));
        }
        for (int n2 = static_cast<int>(N2 / 2); n2 < N2; n2++)
        {
            k2ValsAll_fft.push_back(2.0 * PI * static_cast<double>(n2 - N2) / (2.0 * L2));
        }
        for (int n2 = 0; n2 < N2; n2++)
        {
            k2ValsAll_interpolation.push_back(2 * PI * static_cast<double>(n2) / (2.0 * L2));
        }

        for (const auto& val : k2ValsAll_fft)
        {
            k2ValsAllSquared_fft.push_back(std::pow(val, 2));
        }

        // this->tTot = 1.0;
        // this->Q = static_cast<int>(1e8);
        this->dt = tTot / static_cast<double>(Q);
        this->totalSize = N1 * N2;
        std::cout << "totalSize=" << totalSize << std::endl;
        std::cout << "tTot=" << tTot << std::endl;
        std::cout << "Q=" << Q << std::endl;
        std::cout << "dt=" << dt << std::endl;

//        double x1_tmp = 1;
//        double x2_tmp = 2;
//        double tau_tmp = 0.1;
//        // std::cout<<"F2="<<F2(x1_tmp)<<std::endl;
//        std::cout << "beta=" << this->beta(x1_tmp, tau_tmp) << std::endl;


        //allocate spaces
        this->psiCurr = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->psiTmpCache = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->psiNext = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->Gamma_matrix = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->A_matrix = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->expA_matrix = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);
        this->B_vec = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[N1]);
        this->d_ptr = std::shared_ptr<std::complex<double>[]>(
            new std::complex<double>[totalSize]);


        std::cout << "after allocating pointer spaces" << std::endl;

        // Enable multi-threading
        fftw_init_threads();
        fftw_plan_with_nthreads(parallel_num);

        //initialize fftw plans
        int rank_psi2d = 1; // 1D FFT
        int M1_psi2d = N1;
        int M2_psi2d = N2;
        int n_psi2d[] = {M2_psi2d}; // Length of each 1D FFT
        int howmany_psi2d = M1_psi2d; // Number of 1D FFTs (one per row)
        int istride_psi2d = 1, ostride_psi2d = 1; // Contiguous elements in each row
        int idist_psi2d = M2_psi2d, odist_psi2d = M2_psi2d; // Distance between consecutive rows

        this->plan_2d_row_fft_forward_psi_to_d_ptr = fftw_plan_many_dft(
            rank_psi2d, n_psi2d, howmany_psi2d,
            reinterpret_cast<fftw_complex*>(psiCurr.get()),NULL,
            istride_psi2d, idist_psi2d,
            reinterpret_cast<fftw_complex*>(d_ptr.get()),NULL,
            ostride_psi2d, odist_psi2d,
            FFTW_FORWARD,FFTW_MEASURE
        );

        this->c_mat = arma::cx_dmat(N1, N2, arma::fill::zeros);
        this->_1_over_N2 = std::complex<double>(1.0 / static_cast<double>(N2), 0);
        this->sign_for_c = arma::cx_drowvec(N2, arma::fill::zeros);
        for (int m2 = 0; m2 < N2; m2++)
        {
            sign_for_c(m2) = std::exp(1i * PI * static_cast<double>(m2));
        } //end for m2
        // sign_for_c.print("sign_for_c:");
    } //end constructor

    ~evolution()
    {
        // delete []psiCurr;
        //
        //     delete []psiTmpCache;
        //
        //     delete []psiNext;
        fftw_destroy_plan(plan_2d_row_fft_forward_psi_to_d_ptr);
        fftw_cleanup_threads(); // Clean up thread resources
    }

public:
    void init_and_run();

    /////////////////////////////////////
    //steps for H1R
    ///
    ///evolve H1R only, for benchmark
    void H1R_only();
    ///
    ///evolution H1R, 1 step
    void H1R_1_step();
    ///
    /// evolution for quasilinear solution
    void psi_multiply_with_expA();
    ///
    /// perform interpolations for all rows of psiTmpCache
    void interpolation_all_rows_parallel();
    ///
    /// @param j row number
    /// this function computes product of jth row of c and expS_{j}
    void interpolation_one_row(const int &j);
    ///
    ///convert to c, in arma
    void d_ptr_2_c();
    ///
    /// forward row fft, from psi to d_ptr
    void row_fft_psi_2_d_ptr();
    //end steps for H1R
    /////////////////////////////////////

    /// @param tau: time step
    ///evolution matrix for quasilinear equation
    void construct_expA_matrix(const double& tau);
    ///
    /// @param x1
    /// @param t
    /// @return auxiliary function beta, see notes
    std::complex<double> beta(const double& x1, const double& t);


    ///
    /// @param x1
    /// @param x2
    /// @param t
    /// @return auxiliary function G, see notes
    std::complex<double> G(const double& x1, const double& x2, const double& t);

    ///
    /// @param x1
    /// @return auxiliary function F12, see notes
    std::complex<double> F12(const double& x1);
    ///
    /// @param x1
    /// @return auxiliary function F11, see notes
    std::complex<double> F11(const double& x1);
    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F10, see notes
    std::complex<double> F10(const double& x1, const double& x2);
    ///
    /// @param x1
    /// @return auxiliary function F9, see notes
    std::complex<double> F9(const double& x1);
    ///
    /// @param x1
    /// @return auxiliary function F8, see notes
    std::complex<double> F8(const double& x1);
    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F7, see notes
    std::complex<double> F7(const double& x1, const double& x2);
    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F6, see notes
    std::complex<double> F6(const double& x1, const double& x2);

    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F5, see notes
    std::complex<double> F5(const double& x1, const double& x2);

    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F4, see notes
    std::complex<double> F4(const double& x1, const double& x2);

    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F3, see notes
    std::complex<double> F3(const double& x1, const double& x2);

    ///
    /// @param x1
    /// @return auxiliary function F2, see notes
    std::complex<double> F2(const double& x1);
    ///
    /// @param x1
    /// @return auxiliary function F1, see notes
    std::complex<double> F1(const double& x1);
    ///
    /// @param x1
    /// @param x2
    /// @return auxiliary function F0, see notes
    std::complex<double> F0(const double& x1, const double& x2);

    ///
    /// @param tau time step
    /// this function computes all expSj matrices
    void compute_all_expSj(const double& tau);

    ///
    /// @param I_k2_mat matrix, each column is k2 vectors for interpolation, multiplied by i
    /// @param n1 index of x1
    /// @param tau time step
    /// @return one expSj matrix
    arma::cx_dmat compute_one_expS_j(const arma::cx_dmat& I_k2_mat, const int& n1, const double& tau);

    ///
    /// @param x1
    /// @param x2
    /// @param tau time step
    /// @return
    double s2(const double& x1, const double& x2, const double& tau);
    ///
    /// @param x1
    /// @return rho(x1)
    double rho(const double& x1);
    void init_psi0();
    ///
    /// @param n2 index of x2
    /// @return wavefunction of phonon at n2
    double f2(int n2);

    ///
    /// @param n1 index of x1
    /// @return wavefunction of photon at n1
    double f1(int n1);
    ///
    /// @param n1 row index
    /// @param n2 col index
    /// @return flattened index
    int flattened_ind(int n1, int n2);


    void save_complex_array_to_pickle(std::complex<double> ptr[],
                                      int size,
                                      const std::string& filename);

public:
    int j1H;
    int j2H;
    double g0;
    double omegam;
    double omegap;
    double omegac;
    double er;
    double thetaCoef;
    int groupNum;
    int rowNum;
    double theta;
    double lmd;
    double Deltam;
    double r;
    double e2r;
    double D;
    double mu;


    int N1; //must be even
    int N2; //must be even
    int totalSize;
    double L1;
    double L2;
    double dx1;

    double dx2;

    double dtEst;
    double tTot;
    double dt;
    int Q;
    int parallel_num;
    int toWrite;

    std::vector<double> x1ValsAll;
    std::vector<double> x2ValsAll;

    std::vector<double> k1ValsAll_fft;
    std::vector<double> k2ValsAll_fft;
    std::vector<double> x1ValsAllSquared;
    std::vector<double> x2ValsAllSquared;
    std::vector<double> k1ValsAllSquared_fft;
    std::vector<double> k2ValsAllSquared_fft;

    std::vector<double> k2ValsAll_interpolation;


    std::shared_ptr<std::complex<double>[]> psiCurr; //current value of psi
    std::shared_ptr<std::complex<double>[]> psiTmpCache; //intermediate value of psi
    std::shared_ptr<std::complex<double>[]> psiNext; //next value of psi

    std::shared_ptr<std::complex<double>[]> Gamma_matrix;
    std::shared_ptr<std::complex<double>[]> B_vec;
    std::shared_ptr<std::complex<double>[]> A_matrix;
    std::shared_ptr<std::complex<double>[]> expA_matrix;
    arma::field<arma::cx_mat> expS; // all expSj

    std::shared_ptr<std::complex<double>[]> d_ptr;
    //2d fft, interpolation
    fftw_plan plan_2d_row_fft_forward_psi_to_d_ptr;

    arma::cx_dmat c_mat;
    std::complex<double> _1_over_N2;

    arma::cx_drowvec sign_for_c;

    std::string inCppFileDir;
    std::string out_wvfunction_dir;

    // arma::cx_dmat expA;
};
#endif //EVOLUTION_HPP
