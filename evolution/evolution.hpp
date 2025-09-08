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
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
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
            }
        } //end while

        //print parameters
        std::cout << std::setprecision(15);
        std::cout << "j1H=" << j1H << ", j2H=" << j2H << ", g0=" << g0
            << ", omegam=" << omegam << ", omegap=" << omegap << ", omegac=" << omegac
            << ", er=" << er << ", thetaCoef=" << thetaCoef << ", groupNum="
            << groupNum << ", rowNum=" << rowNum << ", parallel_num=" << parallel_num << std::endl;
        this->L1 = 5;
        this->L2 = 8;
        this->r = std::log(er);
        this->theta = thetaCoef * PI;
        this->Deltam = omegam - omegap;
        std::cout << "Deltam=" << Deltam << std::endl;
        this->e2r = std::pow(er, 2.0);

        this->lmd = (e2r - 1 / e2r) / (e2r + 1 / e2r) * Deltam;
        std::cout << "lambda=" << lmd << std::endl;

        this->D=std::pow(lmd*std::sin(theta),2.0)+std::pow(omegap,2.0);
        this->mu=lmd*std::cos(theta)+Deltam;
        std::cout<<"D="<<D<<std::endl;
        std::cout<<"mu="<<mu<<std::endl;
        double height1 = 0.5;
        double width1 = std::pow(-2.0 * std::log(height1) / omegac, 0.5);
        double minGrid1 = width1 / 10.0;
        this->N2 = 100;


        this->N1 = static_cast<int>(std::ceil(L1 * 2.0 / minGrid1));
        if (N1 % 2 == 1)
        {
            N1 += 1;
        }

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
            k1ValsAll_fft.push_back(2 * PI * static_cast<double>(n1) / (2.0 * L1));
        }
        for (int n1 = static_cast<int>(N1 / 2); n1 < N1; n1++)
        {
            k1ValsAll_fft.push_back(2 * PI * static_cast<double>(n1 - N1) / (2.0 * L1));
        }


        for (const auto& val : k1ValsAll_fft)
        {
            k1ValsAllSquared_fft.push_back(std::pow(val, 2));
        }
        for (int n2 = 0; n2 < static_cast<int>(N2 / 2); n2++)
        {
            k2ValsAll_fft.push_back(2 * PI * static_cast<double>(n2) / (2.0 * L2));
        }
        for (int n2 = static_cast<int>(N2 / 2); n2 < N2; n2++)
        {
            k2ValsAll_fft.push_back(2 * PI * static_cast<double>(n2 - N2) / (2.0 * L2));
        }

        for (const auto& val : k2ValsAll_fft)
        {
            k2ValsAllSquared_fft.push_back(std::pow(val, 2));
        }

        this->tTot = 5.0;
        this->Q = static_cast<int>(1e3);
        this->dt = tTot / static_cast<double>(Q);
        this->totalSize = N1 * N2;
        std::cout<<"totalSize="<<totalSize<<std::endl;
        std::cout << "tTot=" << tTot << std::endl;
        std::cout << "Q=" << Q << std::endl;
        std::cout << "dt=" << dt << std::endl;

        //allocate spaces
        this->psiCurr = std::shared_ptr<std::complex<double>[]>(
    new std::complex<double>[totalSize]);
        this->psiTmpCache = std::shared_ptr<std::complex<double>[]>(
    new std::complex<double>[totalSize]);
        this->psiNext = std::shared_ptr<std::complex<double>[]>(
    new std::complex<double>[totalSize]);

        std::cout << "after allocating pointer spaces" << std::endl;
    } //end constructor

    ~evolution()
    {
        // delete []psiCurr;
        //
        //     delete []psiTmpCache;
        //
        //     delete []psiNext;
    }

public:
    ///
    /// @param x1
    /// @param x2
    /// @param tau time step
    /// @return
    double s2(const double&x1, const double & x2, const double &tau);
    ///
    /// @param x1
    /// @return rho(x1)
    double rho(const double &x1);
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
    int totalSize ;
    double L1;
    double L2;
    double dx1;

    double dx2;

    double dtEst;
    double tTot;
    double dt;
    int Q;
    int parallel_num;

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

    std::shared_ptr<arma::cx_dmat[]> expS;

    arma::cx_dmat expA;
};
#endif //EVOLUTION_HPP
