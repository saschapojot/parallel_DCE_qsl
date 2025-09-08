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