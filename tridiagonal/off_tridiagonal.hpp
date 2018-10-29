#ifndef TRIDIAGONAL_OFF_TRIDIAGONAL_H
#define TRIDIAGONAL_OFF_TRIDIAGONAL_H

#include <vector>
#include <type_traits>

#include <eigen3/Eigen/Core>

#include "utils.hpp"

namespace tridiagonal {
    using std::vector;
    using Eigen::MatrixBase;

    /**
    * Solve tri-diagonal block matrix system on the form
    *
    * |  B_0  C_0  0    0    0     ...     A_0   | | X_1   |     | D_1   |
    * |  A_1  B_1  C_1  0    0     ...     0     | | X_2   |     | D_2   |
    * |  0    A_2  B_2  C_2  0     ...     0     | |       |     |       |
    * |                                          | |  .    |     |  .    |
    * |  .           .    .    .           .     | |  .    |  =  |  .    |
    * |  .             .    .    .         .     | |  .    |     |  .    |
    * |  .               .    .    .       .     | |       |     |       |
    * |                      A_n-2  B_n-2  C_n-2 | | X_n-2 |     | D_n-2 |
    * |  C_n-1               0      A_n-1  B_n-1 | | X_n-1 |     | D_n-1 |
    *
    * using a recursive algorithm
    *
    * @param lower_diagonal : list of matrices on the lower diagonal {A_0, ..., A_n-1}
    * @param diagonal : list of matrices on the diagonal {B_1, ..., B_n-1}
    * @param upper_diagonal : list of matrices on the upper diagonal {C_0, ..., C_n-1}
    * @param rhs : list of matrices on the right and side {D_0, ..., D_n-1}
    * @return list of solutions {X_0, ..., X_n-1}
    */
    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
    >
    vector<D> _solve_off_tridiagonal(const vector<A> & lower_diagonal,
                                    const vector<B> & diagonal,
                                    const vector<C> & upper_diagonal,
                                    const vector<D> & rhs){

        unsigned long n = diagonal.size();

        if(n == 1){
            //
            //  | B_0 | | x_0 | = | D_0 |
            //
            D x = diagonal[0].fullPivHouseholderQr().solve(rhs[0]);
            return {x};
        }

        vector<A> new_lower_diagonal;
        vector<B> new_diagonal;
        vector<C> new_upper_diagonal;
        vector<D> new_rhs;

        A a0 = lower_diagonal[0];
        B b0 = diagonal[0];
        C c0 = upper_diagonal[0];
        D d0 = rhs[0];

        auto b0inv_c0 = b0.inverse() * c0;
        auto b0inv_d0 = b0.inverse() * d0;

        if(n == 2){
            //
            //  | B_0 C_0 | | x_0 | = | D_0 |
            //  | A_0 B_1 | | x_1 | = | D_1 |
            //

            // B_1' = B_1 - A_0 * B_0.inv() * C0
            new_diagonal.push_back(diagonal[1] - a0 * b0inv_c0);
            new_rhs.push_back(rhs[1] - a0 * b0inv_d0);
            vector<D> x_list = _solve_off_tridiagonal(new_lower_diagonal, new_diagonal, new_upper_diagonal, new_rhs);

            D x = b0inv_d0 - b0inv_c0 * x_list[0];

            x_list.insert(x_list.begin(), x);

            return x_list;
        }

        auto b0inv_a0 = b0.inverse() * a0;

        if(n == 3){
            //
            //  | B_0 C_0 A_0 | | x_0 | = | D_0 |
            //  | A_1 B_1 C_1 | | x_1 | = | D_1 |
            //  | C_2 A_2 B_2 | | x_2 | = | D_2 |
            //

            // A_2' = A_2 - C_2 * B_0.inv() * C_0
            new_lower_diagonal.push_back(lower_diagonal[2] - upper_diagonal[2] * b0inv_c0);

            // C_1' = C_1 - A_1 * B_0.inv() * A_0
            new_upper_diagonal.push_back(upper_diagonal[1] - lower_diagonal[1] * b0inv_a0);

        }else{

            // A_1' = - A_1 * B_0.inv() * A_0
            new_lower_diagonal.push_back(-1*lower_diagonal[1] * b0inv_a0);
            for(int i = 2; i < n; i++){
                new_lower_diagonal.push_back(lower_diagonal[i]);
            }


            for(int i = 1; i < n - 1; i++){
                new_upper_diagonal.push_back(upper_diagonal[i]);
            }
            // C_{n-1}' = - C_{n-1} * B_0*inv() * C_0
            new_upper_diagonal.push_back(-1 * upper_diagonal[n - 1] * b0inv_c0);
        }


        // B_1' = B_1 - A_1 * B_0.inv() * C_0
        new_diagonal.push_back(diagonal[1] - lower_diagonal[1] * b0inv_c0);
        for(int i = 2; i < n - 1; i++){
            new_diagonal.push_back(diagonal[i]);
        }
        // B_{n-1}' = B_{n-1} - C_{n-1} * B_0.inv() * A_0
        new_diagonal.push_back(diagonal[n - 1] - upper_diagonal[n - 1] * b0inv_a0);


        // D_1' = D_1 - A_1 * B_0.inv() * D_0
        new_rhs.push_back(rhs[1] - lower_diagonal[1] * b0inv_d0);
        for(int i = 2; i < n - 1; i++){
            new_rhs.push_back(rhs[i]);
        }
        // D_{n-1}' = D_{n-1} - C_{n-1} * B_0.inv() * D_0
        new_rhs.push_back(rhs[n - 1] - upper_diagonal[n - 1] * b0inv_d0);


        vector<D> x_list = _solve_off_tridiagonal(new_lower_diagonal, new_diagonal, new_upper_diagonal, new_rhs);

        // X_0 = B_0.inv() * (D_0 - C_0 * X_1 - A_0 * X_{n-1})
        D x = b0inv_d0 - b0inv_c0 * x_list[0] - b0inv_a0 * x_list[x_list.size() - 1];

        x_list.insert(x_list.begin(), x);

        return x_list;
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
    void check_valid_off_tridiagonal_number_of_matrices(vector<A> const &lower_diagonal,
                                                        vector<B> const &diagonal,
                                                        vector<C> const &upper_diagonal,
                                                        vector<D> const &rhs) {

        if(diagonal.size() != rhs.size()){
            throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                           diagonal,
                                                                           upper_diagonal,
                                                                           rhs));
        }

        if(diagonal.empty()){
            throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                           diagonal,
                                                                           upper_diagonal,
                                                                           rhs));
        }
        else if(diagonal.size() == 1){
            if(lower_diagonal.empty() and !upper_diagonal.empty()) {
                throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                               diagonal,
                                                                               upper_diagonal,
                                                                               rhs));
            }
        }
        else if(diagonal.size() == 2){
            if(lower_diagonal.size() != 1 and upper_diagonal.size() != 1) {
                throw std::invalid_argument(invalid_number_of_elements_message(lower_diagonal,
                                                                               diagonal,
                                                                               upper_diagonal,
                                                                               rhs));
            }
        }
        else if(diagonal.size() != lower_diagonal.size() or diagonal.size() != upper_diagonal.size()){
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
    void check_valid_off_tridiagonal_matrix_dimensions(vector<A> const &lower_diagonal,
                                                       vector<B> const &diagonal,
                                                       vector<C> const &upper_diagonal,
                                                       vector<D> const &rhs) {
        unsigned long n = diagonal.size();

        if (n == 1){
            if (diagonal[0].rows() != rhs[0].rows()){
                throw std::invalid_argument("Invalid matrix dimensions");
            }
        }
        else if(n == 2){
            // upper left corner
            if(diagonal[0].cols() != lower_diagonal[0].cols() or
               diagonal[0].rows() != upper_diagonal[0].rows() or
               diagonal[0].rows() != rhs[0].rows()) {
                throw std::invalid_argument("Invalid matrix dimensions");
            }

            // bottom right corner
            if(diagonal[1].cols() != upper_diagonal[0].cols() or
               diagonal[1].rows() != lower_diagonal[0].rows() or
               diagonal[1].rows() != rhs[1].rows()){
                throw std::invalid_argument("Invalid matrix dimensions");
            }
        }
        else{
            // upper left corner
            if(diagonal[0].cols() != lower_diagonal[1].cols() or
               diagonal[0].cols() != upper_diagonal[n-1].cols() or
               diagonal[0].rows() != upper_diagonal[0].rows() or
               diagonal[0].rows() != lower_diagonal[0].rows() or
               diagonal[0].rows() != rhs[0].rows()) {
                throw std::invalid_argument("Invalid matrix dimensions");
            }

            // middle
            for(int i = 1; i < diagonal.size()-1; i++){
                if(diagonal[i].cols() != upper_diagonal[i-1].cols() or
                   diagonal[i].cols() != lower_diagonal[i+1].cols() or
                   diagonal[i].rows() != upper_diagonal[i].rows() or
                   diagonal[i].rows() != lower_diagonal[i].rows() or
                   diagonal[i].rows() != rhs[i].rows()){
                    throw std::invalid_argument("Invalid matrix dimensions");
                }
            }

            // bottom right corner
            if(diagonal[n-1].cols() != upper_diagonal[n-2].cols() or
               diagonal[n-1].cols() != lower_diagonal[0].cols() or
               diagonal[n-1].rows() != lower_diagonal[n-1].rows() or
               diagonal[n-1].rows() != upper_diagonal[n-1].rows() or
               diagonal[n-1].rows() != rhs[n-1].rows()){
                throw std::invalid_argument("Invalid matrix dimensions");
            }
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


    // #####################################################################################
    //
    //                                INTERFACE FUNCTIONS
    //
    // #####################################################################################

    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<!is_dynamic<A>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<B>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<C>::value, int>::type = 0,
            typename std::enable_if<!is_dynamic<D>::value, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == A::ColsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::ColsAtCompileTime == B::ColsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == C::RowsAtCompileTime, int>::type = 0,
            typename std::enable_if<A::RowsAtCompileTime == D::RowsAtCompileTime, int>::type = 0
    >
    vector<D> solve_off_tridiagonal(vector<A> const & lower_diagonal,
                                    vector<B> const & diagonal,
                                    vector<C> const & upper_diagonal,
                                    vector<D> const & rhs) {

        check_valid_off_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);

        return _solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
    }


    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<is_dynamic<A>::value or
                                    is_dynamic<B>::value or
                                    is_dynamic<C>::value or
                                    is_dynamic<D>::value, int>::type = 0
    >
    vector<D> solve_off_tridiagonal(vector<A> const & lower_diagonal,
                                    vector<B> const & diagonal,
                                    vector<C> const & upper_diagonal,
                                    vector<D> const & rhs) {

        check_valid_off_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);
        check_valid_off_tridiagonal_matrix_dimensions(lower_diagonal, diagonal, upper_diagonal, rhs);

        return _solve_off_tridiagonal(lower_diagonal, diagonal, upper_diagonal, rhs);
    }


}


#endif // TRIDIAGONAL_OFF_TRIDIAGONAL_H
