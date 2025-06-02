//
// Created by adada on 27/5/2025.
//

#include "evolution.hpp"


void evolution::construct_S_mat_spatial()
{
    this->S2_mat_part1 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part2 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part3 = arma::dmat(N1, N2, arma::fill::zeros);
    this->S2_mat_part4 = arma::dmat(N1, N2, arma::fill::zeros);

    //begin initializing S2_mat_part1
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part1(n1, n2) = this->x2ValsAll[n2];
        } //end n2
    } //end n1, end initializing S2_mat_part1

    //begin initializing S2_mat_part2
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part2(n1, n2) = -g0 * lmd * std::sin(theta) / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part2

    //begin initializing S2_mat_part3
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part3(n1, n2) = g0 * omegap / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part3


    //begin initializing S2_mat_part4
    for (int n1 = 0; n1 < N1; n1++)
    {
        double x1n1 = this->x1ValsAll[n1];
        double rhoTmp = this->rho(x1n1);
        for (int n2 = 0; n2 < N2; n2++)
        {
            S2_mat_part4(n1, n2) = -g0 * omegap / D * std::sqrt(2.0 / omegam) * rhoTmp;
        } //end n2
    } //end n1, end initializing S2_mat_part4
}

double evolution::rho(const double& x1)
{
    double val = omegac * std::pow(x1, 2.0) - 0.5;

    return val;
}

double evolution::P1(double rhoVal)
{
    double val = 0.25 * omegac + 0.5 * Deltam - 0.5 * omegac * rhoVal + (2.0 * omegap - mu) / (2.0 * D) * std::pow(
        g0 * rhoVal, 2.0);

    return val;
}

std::complex<double> evolution::gen_A_1_elem(const double& x1, const double& x2, const double& tau)
{
    double rhoVal = this->rho(x1);
    double rhoVal_squared = std::pow(rhoVal, 2.0);

    std::complex<double> part0 = 1i * this->P1(rhoVal) * tau;

    std::complex<double> part1 = 1i * F2 * rhoVal * x2 * std::cos(omegap * tau);

    std::complex<double> part2 = 1i * F3 * rhoVal * x2 * std::sin(omegap * tau);

    std::complex<double> part3 = 1i * F4 * rhoVal_squared * std::cos(2 * omegap * tau);

    std::complex<double> part4 = 1i * F5 * rhoVal_squared * std::sin(2.0 * omegap * tau);

    std::complex<double> part5 = 1i * F6 * rhoVal_squared;

    std::complex<double> part6 = 1i * F7 * std::pow(x2, 2.0);

    std::complex<double> part7(0.5 * lmd * std::sin(theta) * tau, 0);

    std::complex<double> A_Val = part0 + part1 + part2 + part3
        + part4 + part5 + part6 + part7;

    return A_Val;
}

std::complex<double> evolution::gen_B_1_elem(const double& x1, const double& x2, const double& tau)
{
    double rhoVal = this->rho(x1);
    double rhoVal_squared = std::pow(rhoVal, 2.0);

    double expVal = std::exp(lmd * std::sin(theta) * tau);

    double expVal_squared = std::exp(2.0 * lmd * std::sin(theta) * tau);

    std::complex<double> part0 = 1i * R1 * rhoVal_squared;

    std::complex<double> part1 = 1i * R2 * rhoVal * x2 * expVal;

    std::complex<double> part2 = 1i * R3 * rhoVal_squared * std::sin(omegap * tau) * expVal;

    std::complex<double> part3 = 1i * R4 * rhoVal_squared * std::cos(omegap * tau) * expVal;

    std::complex<double> part4 = 1i * (R5 * std::pow(x2, 2.0) + R6 * rhoVal_squared) * expVal_squared;


    std::complex<double> part5 = 1i * R7 * rhoVal * x2 * std::sin(omegap * tau) * expVal_squared;

    std::complex<double> part6 = 1i * R8 * rhoVal * x2 * std::cos(omegap * tau) * expVal_squared;

    std::complex<double> part7 = 1i * R9 * rhoVal_squared * std::cos(2.0 * omegap * tau) * expVal_squared;

    std::complex<double> part8 = 1i * R10 * rhoVal_squared * std::sin(2.0 * omegap * tau) * expVal_squared;

    std::complex<double> B_Val = part0 + part1 + part2 + part3
        + part4 + part5 + part6 + part7 + part8;

    return B_Val;
}

///
/// @param n1 row index
/// @param n2 col index
/// @return flattened index
int evolution::flattened_ind(int n1, int n2)
{
    return n1 * N2 + n2;
}

void evolution::construct_A(const double& tau)
{
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            double x1n1 = this->x1ValsAll[n1];
            double x2n2 = this->x2ValsAll[n2];
            int flat_ind = flattened_ind(n1, n2);
            this->A[flat_ind] = gen_A_1_elem(x1n1, x2n2, tau);
            this->exp_A_all[flat_ind]=std::exp(A[flat_ind]);
        } //end n2
    } //end n1
}


void evolution::construct_B(const double & tau)
{
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            double x1n1 = this->x1ValsAll[n1];
            double x2n2 = this->x2ValsAll[n2];
            int flat_ind = flattened_ind(n1, n2);
            this->B[flat_ind]=gen_B_1_elem(x1n1, x2n2, tau);
            this->exp_B_all[flat_ind]=std::exp(-B[flat_ind]);
        }//end n2
    }//end n1
}

arma::cx_dmat evolution::construct_S_mat(const double& tau)
{
    // this->construct_S_mat_spatial();
    double exp_part = std::exp(lmd * std::sin(theta) * tau);

    double sin_val = std::sin(omegap * tau);

    double cos_val = std::cos(omegap * tau);

    arma::dmat S2_mat = S2_mat_part1 * exp_part + S2_mat_part2 * sin_val * exp_part
        + S2_mat_part3 * cos_val * exp_part + S2_mat_part4;


    return arma::conv_to<arma::cx_dmat>::from(S2_mat);
}
void evolution::construct_eS2_all(const double & tau)
{
  this->eS2_all=  std::shared_ptr<arma::cx_dmat[]>(new arma::cx_dmat[N1]);
    arma::cx_dmat S2_mat=this->construct_S_mat(tau);
    for (int j1=0;j1<N1;j1++)
    {
        arma::cx_drowvec one_row=S2_mat.row(j1);

        arma::cx_dmat k2S2_j1=arma::kron(k2Vals_arma_col_vec,one_row);
        //std::cout<<k2S2_j1.n_rows<<", "<<k2S2_j1.n_cols<<std::endl;
        eS2_all[j1]=arma::exp(1i*k2S2_j1);

    }//end for j1


}

void evolution::init_c_rows_all()
{
    this->c_rows_all =std::shared_ptr<arma::cx_drowvec[]>(new arma::cx_drowvec[N1]);
    for (int j1=0;j1<N1;j1++)
    {
        c_rows_all[j1]=arma::cx_drowvec(N2,arma::fill::zeros);
    }//end j1
    // std::cout<<"finished init c_rows_all"<<std::endl;
}

void evolution::init_Psi_tilde_rows_all()
{
this->Psi_tilde_rows_all=std::shared_ptr<arma::cx_drowvec[]>(new arma::cx_drowvec[N1]);
    for (int j1=0;j1<N1;j1++)
    {
        Psi_tilde_rows_all[j1]=arma::cx_drowvec(N2,arma::fill::zeros);
    }//end j1
}
double evolution::f1(int n1)
{
    double x1TmpSquared = x1ValsAllSquared[n1];
    double x1Tmp = x1ValsAll[n1];

    double valTmp = std::exp(-0.5 * omegac * x1TmpSquared)
        * std::hermite(this->j1H, std::sqrt(omegac) * x1Tmp);


    return valTmp;
}

double evolution::f2(int n2)
{
    double x2TmpSquared = x2ValsAllSquared[n2];
    double x2Tmp = x2ValsAll[n2];


    //    double valTmp=std::exp(-0.5 * omegam*std::exp(-2.0*r) * x2TmpSquared)
    //                  *std::hermite(this->jH2,std::sqrt(omegam*std::exp(-2.0*r))*x2Tmp);
    double valTmp = std::exp(-0.5 * omegam * x2TmpSquared)
        * std::hermite(this->j2H, std::sqrt(omegam) * x2Tmp);

    return valTmp;
}

void evolution::init_Psi0()
{

    arma::cx_dcolvec vec1(N1);
    arma::cx_drowvec vec2(N2);
    for (int n1 = 0; n1 < N1; n1++)
    {
        vec1(n1) = f1(n1);
    }
    for (int n2 = 0; n2 < N2; n2++)
    {
        vec2(n2) = f2(n2);
    }
    arma::cx_dmat Psi0_arma=arma::kron(vec1, vec2);
    std::complex<double> nm(arma::norm(Psi0_arma, "fro"), 0);
    Psi0_arma/=nm;
    int n_row=Psi0_arma.n_rows;
    int n_col=Psi0_arma.n_cols;
    std::cout<<"n_row="<<n_row<<", n_col="<<n_col<<std::endl;
    std::cout<<"norm="<<arma::norm(Psi0_arma,"fro")<<std::endl;
    arma::cx_dmat Psi0_transposed = Psi0_arma.t();  // Creates a new matrix

    std::memcpy(this->Psi0.get(),Psi0_transposed.memptr(),N1*N2*sizeof(std::complex<double>));
    // int n1=20;
    // int n2=60;
    // int flat_ind=flattened_ind(n1,n2);
    // std::cout<<"flat_ind="<<flat_ind<<std::endl;
    // std::cout<<Psi0[flat_ind]<<std::endl;


}

//initialize coefficients from d to c
void evolution::init_d_2_c_coefs()
{
    for (int j=0;j<N1*N2;j++)
    {
        this->d_2_c_coefs[j]=std::complex<double>(1.0/static_cast<double>(N2),0);
        if (j%2==1)
        {
            d_2_c_coefs[j]*=-1.0;
        }//end if
    }//end for j
}
void evolution::init()
{
    //U1, A and B
    this->construct_A(this->dt);
    this->construct_B(this->dt);
    this->construct_eS2_all(this->dt);
    this->init_Psi0();

    this->init_d_2_c_coefs();
    this->init_c_rows_all();
    this->init_Psi_tilde_rows_all();


}

///
/// @param j1 row index
void evolution::interpolation_Psi_tilde(int j1)
{
    //copy c in ptr to c_rows_all[j1]
    int flat_ind=flattened_ind(j1,0);
    std::memcpy(c_rows_all[j1].memptr(),c.get()+flat_ind,N2*sizeof(std::complex<double>));
    Psi_tilde_rows_all[j1]=c_rows_all[j1]*eS2_all[j1];
    std::memcpy(Psi_tilde.get()+flat_ind,Psi_tilde_rows_all[j1].memptr(),N2*sizeof(std::complex<double>));
}

void evolution::step_U1(
       int j )
{

    //Psi to d
    fftw_execute(plan_row_fft_Psi_2_d_serial);

    //d to c
    for (int k=0;k<N1*N2;k++)
    {
        this->c[j]=this->d[j]*this->d_2_c_coefs[j];
    }//end for j

    //serial interpolation
    for (int j1=0;j1<N1;j1++)
    {
        interpolation_Psi_tilde(j1);
    }//end for j1
//next Psi
    for (int k=0;k<N1*N2;k++)
    {
        Psi[k]=Psi_tilde[k]*exp_B_all[k]*exp_A_all[k];
    }//end k
}



void evolution::run_and_save_H1R_only()
{
    //copy Psi0 to Psi
    std::memcpy(Psi.get(),Psi0.get(),N1*N2*sizeof(std::complex<double>));
    std::string outPath="./outData/group"+std::to_string(groupNum)+"/row"
   +std::to_string(rowNum)+"/wavefunction/";
    std::string outFileName;
    if (!fs::is_directory(outPath) || !fs::exists(outPath))
    {
        fs::create_directories(outPath);
    }//end creating outPath
    std::cout<<"created out dir"<<std::endl;
    for (int j=0;j<10;j++)
    {
        this->step_U1(j);
        if (j%1==0)
        {
            std::cout<<"saving at step "<<j<<std::endl;
            outFileName=outPath+"/at_time_step_"+std::to_string(j+1)+".pkl";
            this->save_complex_array_to_pickle(Psi.get(),N1*N2,outFileName);
        }//end if
    }//end for j

}


void evolution::save_complex_array_to_pickle(std::complex<double> ptr[],
                                    int size,
                                    const std::string& filename)
{
    // Initialize Python interpreter if it is not already initialized.
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize();  // Initialize NumPy
    }

    try
    {
        // Import the pickle module and retrieve the dumps function.
        bp::object pickle = bp::import("pickle");
        bp::object pickle_dumps = pickle.attr("dumps");

        // Convert the C++ complex array to a NumPy array.
        np::ndarray numpy_array = np::from_data(
            ptr,                                           // Raw pointer to data
            np::dtype::get_builtin<std::complex<double>>(),// NumPy dtype for std::complex<double>
            bp::make_tuple(size),                          // Shape: 1D array with "size" elements
            bp::make_tuple(sizeof(std::complex<double>)),  // Stride: size of one element
            bp::object()                                   // No base object provided
        );

        // Serialize the NumPy array using pickle.dumps.
        bp::object serialized_obj = pickle_dumps(numpy_array);
        std::string serialized_str = bp::extract<std::string>(serialized_obj);

        // Write the serialized data to a file.
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Optional debug output.
        // std::cout << "Complex array successfully serialized and saved to " << filename << std::endl;
    }
    catch (const bp::error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred while saving complex array." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}