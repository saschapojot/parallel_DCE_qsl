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
        np::initialize(); // Initialize NumPy
    }

    try
    {
        // Import the pickle module and retrieve the dumps function.
        bp::object pickle = bp::import("pickle");
        bp::object pickle_dumps = pickle.attr("dumps");

        // Convert the C++ complex array to a NumPy array.
        np::ndarray numpy_array = np::from_data(
            ptr, // Raw pointer to data
            np::dtype::get_builtin<std::complex<double>>(), // NumPy dtype for std::complex<double>
            bp::make_tuple(size), // Shape: 1D array with "size" elements
            bp::make_tuple(sizeof(std::complex<double>)), // Stride: size of one element
            bp::object() // No base object provided
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
    return n1 * N2 + n2;
}

///
/// @param x1
/// @return rho(x1)
double evolution::rho(const double& x1)
{
    return omegac * std::pow(x1, 2.0) - 0.5;
}

///
/// @param x1
/// @param x2
/// @param tau time step
/// @return
double evolution::s2(const double& x1, const double& x2, const double& tau)
{
    double rho_val = rho(x1);
    double exp_val = std::exp(lmd * std::sin(theta) * tau);

    double val1 = -g0 / D * omegap * std::sqrt(2.0 / omegam) * rho_val;
    double val2 = x2 * exp_val;

    double val3 = -g0 / D * std::sqrt(2.0 / omegam) * lmd * std::sin(theta) * rho_val * std::sin(omegap * tau) *
        exp_val;

    double val4 = g0 / D * std::sqrt(2.0 / omegam) * omegap * rho_val * std::cos(omegap * tau) * exp_val;
    return val1 + val2 + val3 + val4;
}


///
/// @param tau time step
/// this function computes all expSj matrices
void evolution::compute_all_expSj(const double& tau)
{
    expS.set_size(N1); // Create field to hold N1 matrices
    for (int n1 = 0; n1 < N1; n1++)
    {
        expS(n1).set_size(N2, N2);
        expS(n1).zeros();
    } //end for n1

    arma::cx_dmat I_k2_mat(N2, N2, arma::fill::zeros);
    arma::cx_dvec arma_vector = arma::conv_to<arma::cx_dvec>::from(k2ValsAll_interpolation);
    I_k2_mat.each_col() = arma_vector * 1i;

    for (int n1 = 0; n1 < N1; n1++)
    {
        expS(n1) = compute_one_expS_j(I_k2_mat, n1, tau);
    }
}


///
/// @param I_k2_mat matrix, each column is k2 vectors for interpolation, multiplied by i
/// @param n1 index of x1
/// @param tau time step
/// @return one expSj matrix
arma::cx_dmat evolution::compute_one_expS_j(const arma::cx_dmat& I_k2_mat, const int& n1, const double& tau)
{
    arma::cx_drowvec S2n1(N2);

    double x1n1 = this->x1ValsAll[n1];

    for (int n2 = 0; n2 < N2; n2++)
    {
        double x2n2 = this->x2ValsAll[n2];
        double S2n1n2 = this->s2(x1n1, x2n2, tau);
        S2n1(n2) = std::complex<double>(S2n1n2, 0);
    } //end for n2

    arma::cx_dmat one_expSj(N2, N2, arma::fill::zeros);

    one_expSj = I_k2_mat;
    one_expSj.each_row() %= S2n1;
    one_expSj = arma::exp(one_expSj);

    return one_expSj;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F0, see notes
std::complex<double> evolution::F0(const double& x1, const double& x2)
{
    std::complex<double> part1 = 1i * omegam * mu / (4 * lmd * std::sin(theta)) * std::pow(x2, 2.0);

    double rho_val = rho(x1);

    std::complex<double> part2 = 1i * std::pow(g0 / D * rho_val, 2.0) *
    ((2.0 * omegap - D / (2.0 * omegap) - mu / 2.0) * lmd * std::sin(theta) + mu * D / (4.0 * lmd *
        std::sin(theta)));
    return part1 + part2;
}

///
/// @param x1
/// @return auxiliary function F1, see notes
std::complex<double> evolution::F1(const double& x1)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = 1i * std::pow(g0, 2.0) / D * (omegap - mu / 2.0) * std::pow(rho_val, 2.0);
    return part1;
}

///
/// @param x1
/// @return auxiliary function F2, see notes
std::complex<double> evolution::F2(const double& x1)
{
    double rho_val = rho(x1);
    double lmd_sq = std::pow(lmd, 2.0);
    double sin_sq = std::pow(std::sin(theta), 2.0);

    std::complex<double> part1 = 1i * std::pow(g0 / D * rho_val, 2.0) *
    (2.0 * lmd_sq * D * sin_sq + 4.0 * mu * lmd_sq * omegap * sin_sq
        + mu * std::pow(omegap, 3.0)
        - 3.0 * mu * lmd_sq * omegap * sin_sq) / (4.0 * lmd * omegap * std::sin(theta));

    return part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F3, see notes
std::complex<double> evolution::F3(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = 1i * g0 / D * (0.5 * mu - omegap) * std::sqrt(2.0 * omegam) * rho_val * x2;
    return part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F4, see notes
std::complex<double> evolution::F4(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    double lmd_sin_theta = lmd * std::sin(theta);

    std::complex<double> part1 = 1i * g0 / D * std::sqrt(2.0 * omegam)
        * (mu * omegap + 2.0 * std::pow(lmd_sin_theta, 2.0)) / (2.0 * lmd_sin_theta)
        * rho_val * x2;
    return part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F5, see notes
std::complex<double> evolution::F5(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = -1i * mu / (4.0 * lmd * std::sin(theta) * D)
        * (D * omegam * std::pow(x2, 2.0) + std::pow(g0 * rho_val, 2.0));

    return part1;
}


///
/// @param x1
/// @param x2
/// @return auxiliary function F6, see notes
std::complex<double> evolution::F6(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = 1i * mu * g0 / D * std::sqrt(omegam / 2.0) * rho_val * x2;
    return part1;
}


///
/// @param x1
/// @param x2
/// @return auxiliary function F7, see notes
std::complex<double> evolution::F7(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = -1i * mu * g0 / (lmd * std::sin(theta) * D)
        * std::sqrt(omegam / 2.0) * omegap * rho_val * x2;
    return part1;
}


///
/// @param x1
/// @return auxiliary function F8, see notes
std::complex<double> evolution::F8(const double& x1)
{
    double rho_val = rho(x1);
    double lmd_sin_theta = lmd * std::sin(theta);
    std::complex<double> part1 = 1i * mu * std::pow(g0, 2.0) / (4 * std::pow(D, 2.0) * lmd_sin_theta)
        * (std::pow(lmd_sin_theta, 2.0) - std::pow(omegap, 2.0))
        * std::pow(rho_val, 2.0);

    return part1;
}

///
/// @param x1
/// @return auxiliary function F9, see notes
std::complex<double> evolution::F9(const double& x1)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = 1i * mu / 2.0 * std::pow(g0 * rho_val / D, 2.0) * omegap;
    return part1;
}

///
/// @param x1
/// @param x2
/// @return auxiliary function F10, see notes
std::complex<double> evolution::F10(const double& x1, const double& x2)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = -1i * g0 / D * std::sqrt(2.0 * omegam)
        * lmd * std::sin(theta) * rho_val * x2;
    return part1;
}


///
/// @param x1
/// @return auxiliary function F11, see notes
std::complex<double> evolution::F11(const double& x1)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = 1i * 2.0 * std::pow(g0 * lmd * std::sin(theta) / D * rho_val, 2.0);
    return part1;
}

///
/// @param x1
/// @return auxiliary function F12, see notes
std::complex<double> evolution::F12(const double& x1)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = -1i * 2.0 * lmd * omegap
        * std::sin(theta) * std::pow(g0 / D * rho_val, 2.0);

    return part1;
}

///
/// @param x1
/// @param x2
/// @param t
/// @return auxiliary function G, see notes
std::complex<double> evolution::G(const double& x1, const double& x2, const double& t)
{
    std::complex<double> sin_2omegap_t = std::complex<double>(std::sin(2.0 * omegap * t), 0);
    std::complex<double> cos_2omegap_t = std::complex<double>(std::cos(2.0 * omegap * t), 0);

    std::complex<double> sin_omegap_t = std::complex<double>(std::sin(omegap * t), 0);
    std::complex<double> cos_omegap_t = std::complex<double>(std::cos(omegap * t), 0);

    std::complex<double> exp_lmd_sintheta_t = std::complex<double>(std::exp(lmd * std::sin(theta) * t), 0);

    std::complex<double> exp_2lmd_sintheta_t = std::pow(exp_lmd_sintheta_t, 2.0);

    std::complex<double> val0 = F0(x1, x2);

    std::complex<double> val1 = F1(x1) * t;

    std::complex<double> val2 = -1.0 / (2.0 * omegap) * F1(x1) * sin_2omegap_t;

    std::complex<double> val3 = F2(x1) * cos_2omegap_t;
    // std::cout<<"F2(x2)="<<F2(x2)<<std::endl;
    // std::cout<<"cos_2omegap_t="<<cos_2omegap_t<<std::endl;
    std::complex<double> val4 = F3(x1, x2) * sin_omegap_t;

    std::complex<double> val5 = F4(x1, x2) * cos_omegap_t;

    std::complex<double> val6 = F5(x1, x2) * exp_2lmd_sintheta_t;

    std::complex<double> val7 = F6(x1, x2) * sin_omegap_t * exp_2lmd_sintheta_t;

    std::complex<double> val8 = F7(x1, x2) * cos_omegap_t * exp_2lmd_sintheta_t;


    std::complex<double> val9 = F8(x1) * cos_2omegap_t * exp_2lmd_sintheta_t;

    std::complex<double> val10 = F9(x1) * sin_2omegap_t * exp_2lmd_sintheta_t;

    std::complex<double> val11 = F10(x1, x2) * exp_lmd_sintheta_t;

    std::complex<double> val12 = F11(x1) * sin_omegap_t * exp_lmd_sintheta_t;
    std::complex<double> val13 = F12(x1) * cos_omegap_t * exp_lmd_sintheta_t;
    // Print all values
    // std::cout << "val0:  " << val0 << std::endl;
    // std::cout << "val1:  " << val1 << std::endl;
    // std::cout << "val2:  " << val2 << std::endl;
    // std::cout << "val3:  " << val3 << std::endl;
    // std::cout << "val4:  " << val4 << std::endl;
    // std::cout << "val5:  " << val5 << std::endl;
    // std::cout << "val6:  " << val6 << std::endl;
    // std::cout << "val7:  " << val7 << std::endl;
    // std::cout << "val8:  " << val8 << std::endl;
    // std::cout << "val9:  " << val9 << std::endl;
    // std::cout << "val10: " << val10 << std::endl;
    // std::cout << "val11: " << val11 << std::endl;
    // std::cout << "val12: " << val12 << std::endl;
    // std::cout << "val13: " << val13 << std::endl;


    return val0 + val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9
        + val10 + val11 + val12 + val13;
}


void evolution::init_and_run()
{
    this->init_psi0();
    //use Strang splitting
    this->compute_all_expSj(dt);
    this->construct_expA_matrix(dt);

   // double  x1Tmp=0.1;
   // double x2Tmp=0.2;
   // double t=0.01;
   //  double s2Val=s2(x1Tmp,x2Tmp,t);
   // std::complex<double> GVal=G(x1Tmp,x2Tmp,t);
   //  std::cout<<"s2Val="<<s2Val<<", GVal="<<GVal<<std::endl;
    this->H1R_only();
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

void evolution::init_psi0()
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
    arma::cx_dmat psi0_arma = arma::kron(vec1, vec2);
    std::complex<double> nm(arma::norm(psi0_arma, "fro"), 0);
    psi0_arma /= nm;
    int n_row = psi0_arma.n_rows;
    int n_col = psi0_arma.n_cols;
    std::cout << "n_row=" << n_row << ", n_col=" << n_col << std::endl;
    std::cout << "norm=" << arma::norm(psi0_arma, "fro") << std::endl;
    arma::cx_dmat psi0_arma_T = psi0_arma.t();
    std::memcpy(this->psiCurr.get(), psi0_arma_T.memptr(), totalSize * sizeof(std::complex<double>));

    // int flat_ind=2060;
    // std::cout<<psi0_arma(20,60)<<std::endl;
    // std::cout<<psi0_arma(N1/2,N2/2)<<std::endl;
    // save_complex_array_to_pickle(psi0_arma.memptr(),N1*N2,"psi0_arma.pkl");
}


/// @param x1
/// @param t
/// @return auxiliary function beta, see notes
std::complex<double> evolution::beta(const double& x1, const double& t)
{
    double rho_val = rho(x1);
    std::complex<double> part1 = -1i * 0.5 * omegac * rho_val;

    std::complex<double> part2 = 1i * 1.0 / 4.0 * omegac;

    std::complex<double> part3 = 1i * 1.0 / 2.0 * Deltam;

    std::complex<double> part4 = std::complex<double>(1.0 / 2.0 * lmd * sin(theta), 0);

    std::complex<double> retVal = (part1 + part2 + part3 + part4) * t;

    return retVal;
}


/// @param tau: time step
///evolution matrix for quasilinear equation
void evolution::construct_expA_matrix(const double& tau)
{
    // construct Gamma matrix
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int flat_ind = this->flattened_ind(n1, n2);
            double x1Tmp = x1ValsAll[n1];
            double x2Tmp = x2ValsAll[n2];
            this->Gamma_matrix[flat_ind] = G(x1Tmp, x2Tmp, tau);
        } //end for n2
    } // end for n1

    //construct B vec
    for (int j = 0; j < N1; j++)
    {
        double x1Tmp = x1ValsAll[j];
        this->B_vec[j] = beta(x1Tmp, tau);
    } //end for j

    //construct A matrix
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int flat_ind = this->flattened_ind(n1, n2);
            this->A_matrix[flat_ind] = this->Gamma_matrix[flat_ind] + this->B_vec[n1];
        } //end for n2
    } //end for n1
    // construct expA matrix

    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int flat_ind = this->flattened_ind(n1, n2);
            this->expA_matrix[flat_ind] = std::exp(A_matrix[flat_ind]);
        } //end for n2
    } //end for n1
}


///
/// forward row fft, from psi to d_ptr
void evolution::row_fft_psi_2_d_ptr()
{
    fftw_execute(this->plan_2d_row_fft_forward_psi_to_d_ptr);
}


///
///convert to c, in arma
void evolution::d_ptr_2_c()
{
    // Create intermediate matrix that interprets d_ptr as column-major N2×N1
    arma::cx_mat intermediate_matrix(
        reinterpret_cast<arma::cx_double*>(d_ptr.get()),
        N2, N1, // Swapped dimensions: N2 rows, N1 cols
        false, // don't copy data, just create a view
        false // allow external memory
    );

    // Now intermediate_matrix is N2×N1 but contains transposed data
    // Transpose it to get the correct N1×N2 orientation
    this->c_mat = intermediate_matrix.t();
    c_mat *= this->_1_over_N2;

    c_mat.each_row() %= sign_for_c;
}


///
/// @param j row number
/// this function computes product of jth row of c and expS_{j}
void evolution::interpolation_one_row(const int& j)
{
    const arma::cx_dmat& expSj = this->expS(j);

    auto c_row_j_view = c_mat.row(j); // Type: arma::subview_row<cx_double>
    arma::cx_drowvec psi_tilde_row_j = c_row_j_view * expSj;

    int flat_ind_start = this->flattened_ind(j, 0);

    // Direct memory copy of N2 complex values
    std::memcpy(psiTmpCache.get() + flat_ind_start,
                psi_tilde_row_j.memptr(),
                N2 * sizeof(std::complex<double>));
}

///
/// perform interpolations for all rows of psiTmpCache
void evolution::interpolation_all_rows_parallel()
{
    std::vector<std::thread> threads;
    threads.reserve(parallel_num);
    // Calculate chunk size for each thread
    const int chunk_size = (N1 + parallel_num - 1) / parallel_num;
    for (int t = 0; t < parallel_num; ++t)
    {
        int start_row = t * chunk_size;
        int end_row = std::min(start_row + chunk_size, N1);

        if (start_row >= N1) break; // No more rows to process

        threads.emplace_back([this, start_row, end_row]()
        {
            for (int j = start_row; j < end_row; ++j)
            {
                this->interpolation_one_row(j);
            }
        });
    } //end for t

    // Join all threads
    for (auto& thread : threads)
    {
        thread.join();
    }
}


///
/// evolution for quasilinear solution
void evolution::psi_multiply_with_expA()
{
    std::vector<std::thread> threads;
    threads.reserve(parallel_num);
    const int chunk_size = (totalSize + parallel_num - 1) / parallel_num;

    for (int t = 0; t < parallel_num; ++t)
    {
        int start_idx = t * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, totalSize);

        if (start_idx >= totalSize) break;

        threads.emplace_back([this, start_idx, end_idx]()
        {
            for (int i = start_idx; i < end_idx; ++i)
            {
                psiCurr[i] = psiTmpCache[i] * expA_matrix[i];
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

///
///evolution H1R, 1 step
void evolution::H1R_1_step()
{
    //1. perform row fft, forward
    this->row_fft_psi_2_d_ptr();
    //2. get matrix c
    this->d_ptr_2_c();
    //3. interpolation
    this->interpolation_all_rows_parallel();
    //4. quasilinear solution
    this->psi_multiply_with_expA();
}


///
///evolve H1R only, for benchmark
void evolution::H1R_only()
{
    //save initial value
    std::string out_file_name0=out_wvfunction_dir + "/psi" + std::to_string(0) + ".pkl";
    this->save_complex_array_to_pickle(psiCurr.get(), totalSize, out_file_name0);
    for (int q = 0; q < Q; q++)
    {
        const auto t_evo_Start{std::chrono::steady_clock::now()};
        this->H1R_1_step();
        if (q%toWrite==0)
        {
            std::string out_file_name = out_wvfunction_dir + "/psi" + std::to_string(q + 1) + ".pkl";
            this->save_complex_array_to_pickle(psiCurr.get(), totalSize, out_file_name);
        }//end write

        const auto t_evo_End{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{t_evo_End - t_evo_Start};
        std::cout << "step " + std::to_string(q) + ": " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    }
}
