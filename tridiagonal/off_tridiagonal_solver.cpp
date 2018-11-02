#include <vector>
#include <type_traits>
#include <iterator>

#include <eigen3/Eigen/Core>
#include <iostream>

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

        template <typename IterA, typename IterB, typename IterC, typename IterD,
                typename A = typename std::iterator_traits<IterA>::value_type,
                typename B = typename std::iterator_traits<IterB>::value_type,
                typename C = typename std::iterator_traits<IterC>::value_type,
                typename D = typename std::iterator_traits<IterD>::value_type,
                typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
                typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
        >
        vector<D> solve_off_tridiagonal(IterA lower_diagonal_begin, IterA lower_diagonal_end,
                                        IterB diagonal_begin, IterB diagonal_end,
                                        IterC upper_diagonal_begin, IterC upper_diagonal_end,
                                        IterD rhs_begin, IterD rhs_end){

            long n = std::distance(diagonal_begin, diagonal_end);

            if(n == 1){
                //
                //  | B_0 | | x_0 | = | D_0 |
                //
                D x = (*diagonal_begin).fullPivHouseholderQr().solve(*rhs_begin);
                return {x};
            }


            A const & a0 = *lower_diagonal_begin;
            B const & b0 = *diagonal_begin;
            C const & c0 = *upper_diagonal_begin;
            D const & d0 = *rhs_begin;


            auto b0inv_c0 = b0.inverse() * c0;
            auto b0inv_d0 = b0.inverse() * d0;

            ++lower_diagonal_begin; ++diagonal_begin; ++upper_diagonal_begin; ++rhs_begin;

            if(n == 2){
                //
                //  | B_0 C_0 | | x_0 | = | D_0 |
                //  | A_0 B_1 | | x_1 | = | D_1 |
                //

                // B_1' = B_1 - A_0 * B_0.inv() * C_0
                *diagonal_begin = *diagonal_begin - a0 * b0inv_c0;

                // D_1' = D_1 - A_0 * B_0.inv() * D_0
                *rhs_begin = *rhs_begin - a0 * b0inv_d0;

                vector<D> x_list = solve_off_tridiagonal(lower_diagonal_begin, lower_diagonal_end,
                                                         diagonal_begin, diagonal_end,
                                                         upper_diagonal_begin, upper_diagonal_end,
                                                         rhs_begin, rhs_end);

                D x = b0inv_d0 - b0inv_c0 * x_list[0];

                x_list.insert(x_list.begin(), x);

                return x_list;
            }

            --lower_diagonal_end; --diagonal_end; --upper_diagonal_end; --rhs_end;
            auto b0inv_a0 = b0.inverse() * a0;

            // B_1' = B_1 - A_1 * B_0.inv() * C_0
            *diagonal_begin = *diagonal_begin - *lower_diagonal_begin * b0inv_c0;

            // B_{n-1}' = B_{n-1} - C_{n-1} * B_0.inv() * A_0
            *diagonal_end = *diagonal_end - *upper_diagonal_end * b0inv_a0;

            // D_1' = D_1 - A_1 * B_0.inv() * D_0
            *rhs_begin = *rhs_begin - *lower_diagonal_begin * b0inv_d0;

            // D_{n-1}' = D_{n-1} - C_{n-1} * B_0.inv() * D_0
            *rhs_end = *rhs_end - *upper_diagonal_end * b0inv_d0;

            if(n == 3){
                //
                //  | B_0 C_0 A_0 | | x_0 | = | D_0 |
                //  | A_1 B_1 C_1 | | x_1 | = | D_1 |
                //  | C_2 A_2 B_2 | | x_2 | = | D_2 |
                //

                // C_1' = C_1 - A_1 * B_0.inv() * A_0
                *upper_diagonal_begin = *upper_diagonal_begin - *lower_diagonal_begin * b0inv_a0;

                // A_2' = A_2 - C_2 * B_0.inv() * C_0
                *lower_diagonal_end = *lower_diagonal_end - *upper_diagonal_end * b0inv_c0;

                ++lower_diagonal_begin; --upper_diagonal_end;
            }
            else{
                // A_1' = - A_1 * B_0.inv() * A_0
                *lower_diagonal_begin = -1 * *lower_diagonal_begin * b0inv_a0;

                // C_{n-1}' = - C_{n-1} * B_0*inv() * C_0
                *upper_diagonal_end = -1 * *upper_diagonal_end * b0inv_c0;
            }

            ++lower_diagonal_end; ++diagonal_end; ++upper_diagonal_end; ++rhs_end;
            vector<D> x_list = solve_off_tridiagonal(lower_diagonal_begin, lower_diagonal_end,
                                                     diagonal_begin, diagonal_end,
                                                     upper_diagonal_begin, upper_diagonal_end,
                                                     rhs_begin, rhs_end);

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
            typename std::enable_if<A::RowsAtCompileTime == B::RowsAtCompileTime, int>::type,
            typename std::enable_if<A::ColsAtCompileTime == B::ColsAtCompileTime, int>::type,
            typename std::enable_if<A::RowsAtCompileTime == C::RowsAtCompileTime, int>::type,
            typename std::enable_if<A::ColsAtCompileTime == C::ColsAtCompileTime, int>::type,
            typename std::enable_if<A::RowsAtCompileTime == D::RowsAtCompileTime, int>::type
    >
    vector<D> solve_off_tridiagonal(vector<A> const & lower_diagonal,
                                    vector<B> const & diagonal,
                                    vector<C> const & upper_diagonal,
                                    vector<D> const & rhs) {

        internal::check_valid_off_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);

        vector<A> _lower_diagonal = lower_diagonal;
        vector<B> _diagonal = diagonal;
        vector<C> _upper_diagonal = upper_diagonal;
        vector<D> _rhs = rhs;

        return internal::solve_off_tridiagonal(_lower_diagonal.begin(), _lower_diagonal.end(),
                                               _diagonal.begin(), _diagonal.end(),
                                               _upper_diagonal.begin(), _upper_diagonal.end(),
                                               _rhs.begin(), _rhs.end());
    }


    template <typename A, typename B, typename C, typename D,
            typename std::enable_if<is_dynamic<A>::value or
                                    is_dynamic<B>::value or
                                    is_dynamic<C>::value or
                                    is_dynamic<D>::value, int>::type
    >
    vector<D> solve_off_tridiagonal(vector<A> const & lower_diagonal,
                                    vector<B> const & diagonal,
                                    vector<C> const & upper_diagonal,
                                    vector<D> const & rhs) {

        internal::check_valid_off_tridiagonal_number_of_matrices(lower_diagonal, diagonal, upper_diagonal, rhs);
        internal::check_valid_off_tridiagonal_matrix_dimensions(lower_diagonal, diagonal, upper_diagonal, rhs);

        vector<A> _lower_diagonal = lower_diagonal;
        vector<B> _diagonal = diagonal;
        vector<C> _upper_diagonal = upper_diagonal;
        vector<D> _rhs = rhs;

        return internal::solve_off_tridiagonal(_lower_diagonal.begin(), _lower_diagonal.end(),
                                               _diagonal.begin(), _diagonal.end(),
                                               _upper_diagonal.begin(), _upper_diagonal.end(),
                                               _rhs.begin(), _rhs.end());
    }


}

