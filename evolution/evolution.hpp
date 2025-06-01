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
const auto PI=M_PI;

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
    evolution(const std::string &cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty()) {
                continue; // Skip empty lines
            }

            std::istringstream iss(line);
            //read j1H
            if (paramCounter == 0)
            {
                iss>>j1H;
                if (j1H<0)
                {
                    std::cerr << "j1H must be >=0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }//end reading j1H

            //read j2H
            if (paramCounter == 1)
            {
                iss>>j2H;
                if (j2H<0)
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
                iss>>g0;
                paramCounter++;
                continue;
            }
            //end reading g0
            //read omegam
            if(paramCounter == 3)
            {
                iss>>omegam;
                paramCounter++;
                continue;
            }//end reading omegam

            //read omegap
            if(paramCounter == 4)
            {
                iss>>omegap;
                paramCounter++;
                continue;
            }
            //end reading omegap
            //read omegac
            if(paramCounter == 5)
            {
                iss>>omegac;
                paramCounter++;
                continue;
            }//end reading omegac
            //read er
            if(paramCounter == 6)
            {
                iss>>er;
                if(er<=0)
                {
                    std::cerr << "er must be >0" << std::endl;
                    std::exit(1);
                }
                paramCounter++;
                continue;
            }
            //end reading er
            //read thetaCoef
            if(paramCounter == 7)
            {
                iss>>thetaCoef;
                paramCounter++;
                continue;
            }
            //end reading thetaCoef
            //read groupNum
            if (paramCounter==8)
            {
                iss>>groupNum;
                paramCounter++;
                continue;
            }//end groupNum

            //read rowNum
            if (paramCounter==9)
            {
                iss>>rowNum;
                paramCounter++;
                continue;
            }//end rowNum
            // read parallel_num
            if (paramCounter==10)
            {
                iss>>parallel_num;
                paramCounter++;
                continue;
            }
        }//end while

        //print parameters
        std::cout << std::setprecision(15);
        std::cout<<"j1H="<<j1H<<", j2H="<<j2H<<", g0="<<g0
        <<", omegam="<<omegam<<", omegap="<<omegap<<", omegac="<<omegac
        <<", er="<<er<<", thetaCoef="<<thetaCoef<<", groupNum="
        <<groupNum<<", rowNum="<<rowNum<<", parallel_num="<<parallel_num<<std::endl;
        this->L1=5;
        this->L2=8;
        this->r=std::log(er);
        this->theta=thetaCoef*PI;
        this->Deltam=omegam-omegap;
        std::cout<<"Deltam="<<Deltam<<std::endl;
        this->e2r=std::pow(er,2.0);

        this->lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam;
        std::cout<<"lambda="<<lmd<<std::endl;
        double height1=0.5;
        double width1=std::pow(-2.0*std::log(height1)/omegac,0.5);
        double minGrid1=width1/10.0;
        this->N2=100;


        this->N1=static_cast<int>(std::ceil(L1*2.0/minGrid1));
        if(N1%2==1)
        {
            N1+=1;
        }

        std::cout<<"L1="<<L1<<", L2="<<L2<<std::endl;
        std::cout<<"N1="<<N1<<std::endl;
        std::cout<<"N2="<<N2<<std::endl;

        dx1=2.0*L1/static_cast<double>(N1);
        dx2=2.0*L2/static_cast<double>(N2);
        std::cout<<"dx1="<<dx1<<std::endl;
        std::cout<<"dx2="<<dx2<<std::endl;

        for (int n1 =0;n1<N1;n1++){
            this->x1ValsAll.push_back(-L1+dx1*n1);
        }
        for (int n2=0;n2<N2;n2++){
            this->x2ValsAll.push_back(-L2+dx2*n2);
        }
        for(const auto& val: x1ValsAll){
            x1ValsAllSquared.push_back(std::pow(val,2));
        }
        for(const auto &val:x2ValsAll){
            x2ValsAllSquared.push_back(std::pow(val,2));
        }
        for(int n1=0;n1<static_cast<int>(N1/2);n1++){
            k1ValsAll_fft.push_back(2*PI*static_cast<double >(n1)/(2.0*L1));
        }
        for(int n1=static_cast<int>(N1/2);n1<N1;n1++){
            k1ValsAll_fft.push_back(2*PI*static_cast<double >(n1-N1)/(2.0*L1));
        }

        for(int n1=0;n1<N1;n1++)
        {
            k1ValsAll_interpolation.push_back(2*PI*static_cast<double>(n1)/(2.0*L1));

        }
        for(const auto&val: k1ValsAll_fft){
            k1ValsAllSquared_fft.push_back(std::pow(val,2));
        }
        for(int n2=0;n2<static_cast<int>(N2/2);n2++){
            k2ValsAll_fft.push_back(2*PI*static_cast<double >(n2)/(2.0*L2));
        }
        for(int n2=static_cast<int >(N2/2);n2<N2;n2++){
            k2ValsAll_fft.push_back(2*PI*static_cast<double >(n2-N2)/(2.0*L2));
        }
        this->k2Vals_arma_col_vec.set_size(N2);
        for(int n2=0;n2<N2;n2++)
        {
            double tmp=2*PI*static_cast<double>(n2)/(2.0*L2);
            k2ValsAll_interpolation.push_back(tmp);
            k2Vals_arma_col_vec(n2)=arma::cx_double(tmp,0);
        }

        for(const auto &val:k2ValsAll_fft){
            k2ValsAllSquared_fft.push_back(std::pow(val,2));
        }
        //initialize parameters for A

        D=std::pow(lmd*std::sin(theta),2.0)+std::pow(omegap,2.0);
        mu=lmd*std::cos(theta)+Deltam;
        std::cout<<"D="<<D<<std::endl;
        std::cout<<"mu="<<mu<<std::endl;
        this->F2=g0*std::sqrt(2.0*omegam)*(2.0*std::pow(lmd*std::sin(theta),2.0)+omegap*mu)/(2*D*lmd*sin(theta));

        this->F3=g0*std::sqrt(2.0*omegam)/D*(0.5*mu-omegap);

        this->F4=std::pow(g0,2.0)*(2.0*D*std::pow(lmd*std::sin(theta),2.0)+mu*omegap*std::pow(lmd*std::sin(theta),2.0)+mu*std::pow(omegap,3.0))/(4.0*std::pow(D,2.0)*lmd*omegap*std::sin(theta));

        this->F5=std::pow(g0,2.0)*
            (2.0*omegap*D+mu*std::pow(lmd*std::sin(theta),2.0)-4.0*omegap*std::pow(lmd*std::sin(theta),2.0)+mu*std::pow(omegap,2.0)-4.0*std::pow(omegap,3.0))
        /(4.0*omegap*std::pow(D,2.0));

        this->F6=std::pow(g0,2.0)*
            (8.0*omegap*std::pow(lmd*std::sin(theta),2.0)-4.0*mu*std::pow(lmd*std::sin(theta),2.0)+D*mu)
            /(4*lmd*std::sin(theta)*std::pow(D,2.0));

        this->F7=omegam*mu/(4*lmd*std::sin(theta));

        std::cout<<"F2="<<F2<<std::endl;
        std::cout<<"F3="<<F3<<std::endl;
        std::cout<<"F4="<<F4<<std::endl;
        std::cout<<"F5="<<F5<<std::endl;
        std::cout<<"F6="<<F6<<std::endl;
        std::cout<<"F7="<<F7<<std::endl;
        //end initializing parameters for A
        //initialize parameters for B
        this->R1=std::pow(g0,2.0)*lmd*std::sin(theta)/(2.0*omegap)
                *(D-omegap*mu)/std::pow(D,2.0);

        this->R2=g0*lmd*std::sin(theta)/D*std::sqrt(2.0*omegam);

        this->R3=-2.0*std::pow(g0*lmd*std::sin(theta),2.0)/std::pow(D,2.0);

        this->R4=2.0*std::pow(g0,2.0)*omegap*lmd*std::sin(theta)/std::pow(D,2.0);

        this->R5=omegam/(4.0*lmd*std::sin(theta))*mu;

        this->R6=mu*std::pow(g0,2.0)/(4.0*lmd*std::sin(theta)*D);

        this->R7=-mu*g0/(2.0*D)*std::sqrt(2.0*omegam);

        this->R8=mu*g0*omegap/(2.0*lmd*std::sin(theta)*D)*std::sqrt(2.0*omegam);

        this->R9=mu*std::pow(g0,2.0)/(4.0*lmd*std::sin(theta)*std::pow(D,2.0))*(std::pow(omegap,2.0)-std::pow(lmd*std::sin(theta),2.0));

        this->R10=-mu*omegap*std::pow(g0,2.0)/(2*std::pow(D,2.0));

        std::cout<<"R1="<<R1<<std::endl;
        std::cout<<"R2="<<R2<<std::endl;
        std::cout<<"R3="<<R3<<std::endl;

        std::cout<<"R4="<<R4<<std::endl;
        std::cout<<"R5="<<R5<<std::endl;
        std::cout<<"R6="<<R6<<std::endl;

        std::cout<<"R7="<<R7<<std::endl;
        std::cout<<"R8="<<R8<<std::endl;
        std::cout<<"R9="<<R9<<std::endl;

        std::cout<<"R10="<<R10<<std::endl;
        //end initializing parameters for B

        this->tTot=5.0;
        this->Q=static_cast<int>(1e6);
        this->dt=tTot/static_cast<double>(Q);
        std::cout<<"tTot="<<tTot<<std::endl;
        std::cout<<"Q="<<Q<<std::endl;
        std::cout<<"dt="<<dt<<std::endl;
        this->normalizing_factor2d=std::complex<double>(1.0/(static_cast<double>(N1*N2)),0);
        std::cout<<"normalizing_factor2d="<<normalizing_factor2d<<std::endl;

        //matrices
        this->construct_S_mat_spatial();
        std::cout<<"before allocating pointer spaces"<<std::endl;


        //pointers
        this->d=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        this->Psi_tilde=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        this->Psi=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        this->Psi0=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        this->c=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());

        this->d_2_c_coefs=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());

        this->A=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        this->B=std::shared_ptr<std::complex<double>[]>(new std::complex<double>[N1*N2],std::default_delete<std::complex<double>>());
        std::cout<<"after allocating pointer spaces"<<std::endl;

        //fftw plans
        //row fft of each row of Psi 2 d
        int M1_Psi_2_d=N1;
        int M2_Psi_2_d=N2;
        int rank_Psi_2_d=1;// 1D FFT
        int n_Psi_2_d[]={M2_Psi_2_d}; // Length of each 1D FFT
        int howmany_Psi_2_d = M1_Psi_2_d;               // Number of 1D FFTs (one per row)
        int istride_Psi_2_d = 1, ostride_Psi_2_d = 1;   // Contiguous elements in each row
        int idist_Psi_2_d = M2_Psi_2_d, odist_Psi_2_d = M2_Psi_2_d;     // Distance between consecutive rows

        this->plan_row_fft_Psi_2_d_serial=fftw_plan_many_dft(
        rank_Psi_2_d,n_Psi_2_d,howmany_Psi_2_d,
        reinterpret_cast<fftw_complex*>(this->Psi.get()),NULL,
        istride_Psi_2_d, idist_Psi_2_d,
        reinterpret_cast<fftw_complex*>(this->d.get()), NULL,
        odist_Psi_2_d, idist_Psi_2_d,
        FFTW_FORWARD, FFTW_MEASURE
            );





    }//end constructor

    ~ evolution()
    {

    }
public:
    void init();

    void step_U1(int j );
    //initialize coefficients from d to c
    void init_d_2_c_coefs();
    void init_Psi0();
    ///
    /// @param n1 index of x1
    /// @return wavefunction of photon at n1
    double f1(int n1);
    ///
    /// @param n2 index of x2
    /// @return wavefunction of phonon at n2
    double f2(int n2);
    void construct_eS2_all(const double & tau);

    arma::cx_dmat construct_S_mat( const double &tau);
    void construct_A(const double & tau);
    void construct_B(const double & tau);
    std::complex<double> gen_B_1_elem(const double& x1, const double& x2, const double & tau);
    std::complex<double> gen_A_1_elem(const double& x1, const double& x2, const double & tau);
    double rho(const double& x1);
    double P1(double rhoVal);
    void construct_S_mat_spatial();

    ///
    /// @param n1 row index
    /// @param n2 col index
    /// @return flattened index
    int flattened_ind(int n1,int n2);
public:
    int j1H;
    int j2H;
    double g0;
    double omegam;
    double omegap;
    double omegac ;
    double er ;
    double thetaCoef ;
    int groupNum ;
    int rowNum ;
    double theta;
    double lmd;
    double Deltam;
    double r;
    double e2r;


    int N1;//must be even
    int N2;//must be even
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
    std::vector<double> k1ValsAll_interpolation;
    std::vector<double>k2ValsAll_interpolation;

    double alpha;
    double beta;
    double gamma13;
    double gamma23;
    double gamma15;
    double gamma25;

    //parameters for A
    double D;
    double mu;
    double F2,F3,F4,F5,F6,F7;

    //parameters for B

    double R1,R2,R3,R4,R5,R6,R7,R8,R9,R10;

   std::shared_ptr<std::complex<double>[]>  Psi;
    std::shared_ptr<std::complex<double>[]>  Psi_tilde;
    std::shared_ptr<std::complex<double>[]> Psi0;
   std::shared_ptr<std::complex<double>[]>  d;
    std::shared_ptr<std::complex<double>[]>  c;
    std::shared_ptr<std::complex<double>[]> d_2_c_coefs;
    arma::dmat S2_mat_part1,S2_mat_part2,S2_mat_part3,S2_mat_part4;

    std::shared_ptr<std::complex<double>[]> A;
   std::shared_ptr<std::complex<double>[]> B;
    std::shared_ptr<arma::cx_dmat[]> eS2_all;//exp(ik2 S2) for all k2
    arma::cx_dvec k2Vals_arma_col_vec;


    // fft plans
    fftw_plan plan_row_fft_Psi_2_d_serial;

    //data for U2
    std::complex<double> normalizing_factor2d;
};
#endif //EVOLUTION_HPP
