#include "evolution.hpp"


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
///
/// @param n1 row index
/// @param n2 col index
/// @return flattened index
int evolution::flattened_ind(int n1, int n2)
{

    return  n1*N2+n2;
}

///
/// @param x1
/// @return rho(x1)
double evolution::rho(const double &x1)
{
    return omegac*std::pow(x1,2.0)-0.5;
}

///
/// @param x1
/// @param x2
/// @param tau time step
/// @return
double evolution::s2(const double&x1, const double & x2, const double &tau)
{
    double rho_val=rho(x1);
    double exp_val=std::exp(lmd*std::sin(theta)*tau);

    double val1=-g0/D*omegap*std::sqrt(2.0/omegam)*rho_val;
    double val2=x2*exp_val;

    double val3=-g0/D*std::sqrt(2.0/omegam)*lmd*std::sin(theta)*rho_val*std::sin(omegap*tau)*exp_val;

    double val4=g0/D*std::sqrt(2.0/omegam)*omegap*rho_val*std::cos(omegap*tau)*exp_val;
    return val1+val2+val3+val4;
}


///
/// @param tau time step
/// this function computes all expSj matrices
void evolution::compute_all_expSj(const double &tau)
{
    expS.set_size(N1);// Create field to hold N1 matrices
    for (int n1=0;n1<N1;n1++)
    {
        expS(n1).set_size(N2,N2);
        expS(n1).zeros();
    }//end for n1

    arma::cx_dmat I_k2_mat(N2,N2,arma::fill::zeros);
    arma::cx_dvec arma_vector = arma::conv_to<arma::cx_dvec>::from(k2ValsAll_interpolation);
    I_k2_mat.each_col()=arma_vector*1i;

    for (int n1=0;n1<N1;n1++)
    {
        expS(n1)=compute_one_expS_j(I_k2_mat,n1,tau);
    }


}


///
/// @param I_k2_mat matrix, each column is k2 vectors for interpolation, multiplied by i
/// @param n1 index of x1
/// @param tau time step
/// @return one expSj matrix
arma::cx_dmat  evolution::compute_one_expS_j(const arma::cx_dmat & I_k2_mat, const int& n1,const double &tau)
{
    arma::cx_drowvec S2n1(N2);

    double x1n1=this->x1ValsAll[n1];

    for (int n2=0;n2<N2;n2++)
    {
        double x2n2=this->x2ValsAll[n2];
        double S2n1n2=this->s2(x1n1,x2n2,tau);
        S2n1(n2)=std::complex<double>( S2n1n2,0);
    }//end for n2

    arma::cx_dmat one_expSj(N2,N2,arma::fill::zeros);

    one_expSj=I_k2_mat;
    one_expSj.each_row()%=S2n1;
    one_expSj=arma::exp(one_expSj);

    return one_expSj;

}

///
/// @param x1
/// @param x2
/// @return auxiliary function F0, see notes
 std::complex<double> evolution::F0(const double &x1, const double& x2)
{
 std::complex<double> part1=1i*omegam*mu/(4*lmd*std::sin(theta))*std::pow(x2,2.0);

    double rho_val=rho(x1);

    std::complex<double> part2=1i*std::pow(g0/D*rho_val,2.0)*
        ((2.0*omegap-D/(2.0*omegap)-mu/2.0)*lmd*std::sin(theta)+mu*D/(4.0*lmd*std::sin(theta)));
    return part1+part2;

}

///
/// @param x1
/// @return auxiliary function F1, see notes
std::complex<double> evolution::F1(const double &x1)
{
    double rho_val=rho(x1);
std::complex<double> part1=1i*std::pow(g0,2.0)/D*(omegap-mu/2.0)*std::pow(rho_val,2.0);
    return part1;
}

///
/// @param x1
/// @return auxiliary function F2, see notes
std::complex<double> evolution::F2(const double &x1)
{
    double rho_val=rho(x1);
    double lmd_sq=std::pow(lmd,2.0);
    double sin_sq=std::pow(std::sin(theta),2.0);

    std::complex<double> part1=1i*std::pow(g0/D*rho_val,2.0)*
        (2.0*lmd_sq*D*sin_sq+4.0*mu*lmd_sq*omegap*sin_sq
            +mu*std::pow(omegap,3.0)
            -3.0*mu*lmd_sq*omegap*sin_sq)/(4.0*lmd*omegap*std::sin(theta));

    return  part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F3, see notes
std::complex<double> evolution::F3(const double &x1,const double& x2)
{
    double rho_val=rho(x1);
    std::complex<double> part1=1i*g0/D*(0.5*mu-omegap)*std::sqrt(2.0*omegam)*rho_val*x2;
    return part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F4, see notes
std::complex<double> evolution::F4(const double &x1,const double& x2)
{
    double rho_val=rho(x1);
    double lmd_sin_theta=lmd*std::sin(theta);

    std::complex<double> part1=1i*g0/D*std::sqrt(2.0*omegam)
                               *(mu*omegap+2.0*std::pow(lmd_sin_theta,2.0))/(2.0*lmd_sin_theta)
                                *rho_val*x2;
    return  part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F5, see notes
std::complex<double> evolution::F5(const double &x1,const double& x2)
{
    double rho_val=rho(x1);
    std::complex<double>  part1=-1i*mu/(4.0*lmd*std::sin(theta)*D)
                                *(D*omegam*std::pow(x2,2.0)+std::pow(g0*rho_val,2.0));

    return part1;
}


///
/// @param x1
/// @param x2
/// @return auxiliary function F6, see notes
std::complex<double> evolution::F6(const double &x1,const double& x2)
{
    double rho_val=rho(x1);
    std::complex<double>  part1= 1i*mu*g0/D*std::sqrt(omegam/2.0)*rho_val*x2;
    return part1;
}


///
/// @param x1
/// @param x2
/// @return auxiliary function F7, see notes
std::complex<double> evolution::F7(const double &x1,const double& x2)
{
    double rho_val=rho(x1);
    std::complex<double>  part1=-1i*mu*g0/(lmd*std::sin(theta)*D)
                                *std::sqrt(omegam/2.0)*omegap*rho_val*x2;
    return part1;
}


///
/// @param x1
/// @return auxiliary function F8, see notes
std::complex<double> evolution::F8(const double &x1)
{
    double rho_val=rho(x1);
    double lmd_sin_theta=lmd*std::sin(theta);
    std::complex<double> part1=1i*mu*std::pow(g0,2.0)/(4*std::pow(D,2.0)*lmd_sin_theta)
                                *(std::pow(lmd_sin_theta,2.0)-std::pow(omegap,2.0))
                                *std::pow(rho_val,2.0);

    return  part1;


}