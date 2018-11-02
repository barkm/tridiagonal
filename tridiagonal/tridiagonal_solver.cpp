#include <vector>
#include <type_traits>

#include <eigen3/Eigen/Core>

#include "utils.hpp"

namespace tridiagonal {
    using std::vector;
    using Eigen::MatrixBase;

    namespace internal {

        // #####################################################################################
        //
        //                                   SOLVER
        //
        // #####################################################################################

        template <typename A, typename B, typename C, typename D,
                typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
        >
        vector<D> solve_tridiagonal(vector<A> const & lower_diagonal,
                                    vector<B> const & diagonal,
                                    vector<C> const & upper_diagonal,
                                    vector<D> const & rhs) {

            unsigned long n = diagonal.size();

            vector<C> upper_diagonal_prime;
            vector<D> rhs_prime;
            vector<D> solution;

            upper_diagonal_prime.resize(n - 1);
            rhs_prime.resize(n);
            solution.resize(n);

            for(int i = 0; i < n - 1; i++){
                if(i == 0){
                    // C_0' = B_0.inv() * C_0
                    upper_diagonal_prime[0] = diagonal[0].inverse() * upper_diagonal[0];
                }
                else{
                    // C_i' = (B_i - A_{i-1} * C_{i-1}').inv() * C[i]
                    upper_diagonal_prime[i] = (diagonal[i] - lower_diagonal[i - 1] * upper_diagonal_prime[i - 1]).inverse()
                                              * upper_diagonal[i];
                }
            }


            // D_0' = B_0.inv() * D_0
            rhs_prime[0] = diagonal[0].inverse() * rhs[0];
            for(int i = 1; i < n; i++){
                // D_i' = (B_i.inv() - A_{i-1} * C_{i-1}').inv() * (D_i - A_{i-1} * D_{i-1}')
                rhs_prime[i] = (diagonal[i] - lower_diagonal[i - 1] * upper_diagonal_prime[i - 1]).inverse()
                               * (rhs[i] - lower_diagonal[i - 1] * rhs_prime[i - 1]);
            }

            // X_{n-1} = D_{n-1}'
            solution[solution.size() - 1] = rhs_prime.at(diagonal.size()-1);
            for(int i = n - 2; i >= 0; i--){
                // X_i = D_{n-1}' - C_i' * X_{i+1}
                solution[i] = rhs_prime[i] - upper_diagonal_prime[i] * solution[i+1];
            }

            return solution;
        }


        // #####################################################################################
        //
        //                                  INPUT CHECKING
        //
        // #####################################################################################

        template <typename A, typename B, typename C, typename D,
                typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
        >
        void check_valid_tridiagonal_number_of_matrices(vector<A> const &lower_diagonal,
                                                        vector<B> const &diagonal,
                                                        vector<C> const &upper_diagonal,
                                                        vector<D> const &rhs) {
            if(diagonal.empty()){
                throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                               diagonal,
                                                                               upper_diagonal,
                                                                               rhs));
            }

            if(diagonal.size() != lower_diagonal.size() + 1 or
               diagonal.size() != upper_diagonal.size() + 1 or
               diagonal.size() != rhs.size()){
                throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                               diagonal,
                                                                               upper_diagonal,
                                                                               rhs));
            }
        }


        template <typename A, typename B, typename C, typename D,
                typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
        >
        void check_valid_tridiagonal_matrix_dimensions(vector<A> const &lower_diagonal,
                                                       vector<B> const &diagonal,
                                                       vector<C> const &upper_diagonal,
                                                       vector<D> const &rhs) {

            // upper left corner
            if(diagonal[0].cols() != lower_diagonal[0].cols() or
               diagonal[0].rows() != upper_diagonal[0].rows() or
               diagonal[0].rows() != rhs[0].rows()){
                throw std::invalid_argument("Invalid matrix dimensions");
            }

            // middle
            for(int i = 1; i < diagonal.size()-1; i++){
                if(diagonal[i].cols() != upper_diagonal[i-1].cols() or
                   diagonal[i].cols() != lower_diagonal[i].cols() or
                   diagonal[i].rows() != upper_diagonal[i].rows() or
                   diagonal[i].rows() != lower_diagonal[i-1].rows() or
                   diagonal[i].rows() != rhs[i].rows()){
                    throw std::invalid_argument("Invalid matrix dimensions");
                }
            }

            // bottom right corner
            if(diagonal[diagonal.size()-1].cols() != upper_diagonal[upper_diagonal.size()-1].cols() or
               diagonal[diagonal.size()-1].rows() != lower_diagonal[lower_diagonal.size()-1].rows() or
               diagonal[diagonal.size()-1].rows() != rhs[rhs.size()-1].rows()){
                throw std::invalid_argument("Invalid matrix dimensions");
            }

            // check that all diagonal matrices are square
            for (auto const & d : diagonal){
                if (d.cols() != d.rows()){
                    throw std::invalid_argument("All diagonal matrices must be square");
                }
            }

            // check that all rhs matrices have same number of columns
            long col_size = rhs[0].cols();
            for (auto const & r : rhs) {
                if (r.cols() != col_size){
                    throw std::invalid_argument("All right-hand-side matrices must have the same number of columns.");
                }
            }

        }



    }

    // #####################################################################################
    //
    //                                INTERFACE FUNCTIONS
    //
    // #####################################################################################

    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<!is_dynamic<A>::value, int>::type,
            typename std::enable_if<!is_dynamic<B>::value, int>::type,
            typename std::enable_if<!is_dynamic<C>::value, int>::type,
            typename std::enable_if<!is_dynamic<D>::value, int>::type,
            typename std::enable_if<A::RowsAtCompileTime == A::ColsAtCompileTime, int>::type,
            typename std::enable_if<A::ColsAtCompileTime == B::ColsAtCompileTime, int>::type,
            typename std::enable_if<A::RowsAtCompileTime == C::RowsAtCompileTime, int>::type,
            typename std::enable_if<A::RowsAtCompileTime == D::RowsAtCompileTime, int>::type
    >
    vector<D> solve_tridiagonal(vector<A> const & lower_diagonal,
                                vector<B> const & diagonal,
                                vector<C> const & upper_diagonal,
                                vector<D> const & rhs) {

        internal::check_valid_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);

        return internal::solve_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
    }


    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<is_dynamic<A>::value or
                                    is_dynamic<B>::value or
                                    is_dynamic<C>::value or
                                    is_dynamic<D>::value, int>::type
    >
    vector<D> solve_tridiagonal(vector<A> const & lower_diagonal,
                                vector<B> const & diagonal,
                                vector<C> const & upper_diagonal,
                                vector<D> const & rhs) {

        internal::check_valid_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);
        internal::check_valid_tridiagonal_matrix_dimensions(lower_diagonal, diagonal, upper_diagonal, rhs);

        return internal::solve_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
    }

}

