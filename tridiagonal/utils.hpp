#ifndef TRIDIAGONAL_UTILS_HPP
#define TRIDIAGONAL_UTILS_HPP

#include <eigen3/Eigen/Core>
#include <type_traits>

using Eigen::MatrixBase;
using std::vector;

template<typename M,
        typename std::enable_if<std::is_base_of<MatrixBase<M>, M>::value, int>::type = 0
>
struct is_dynamic{
    static const bool value = (M::RowsAtCompileTime == Eigen::Dynamic || M::ColsAtCompileTime == Eigen::Dynamic);
};


template <typename A, typename B, typename C, typename D,
        typename std::enable_if<std::is_base_of<MatrixBase<A>, A>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<MatrixBase<B>, B>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<MatrixBase<C>, C>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<MatrixBase<D>, D>::value, int>::type = 0
>
std::string invalid_number_of_elements_message(const vector<A> &lower_diagonal,
                                               const vector<B> &diagonal,
                                               const vector<C> &upper_diagonal,
                                               const vector<D> &rhs) {
    return std::string("Invalid number of block matrix elements:"
                               " diagonal: " + std::to_string(diagonal.size()) +
                       ", lower: " + std::to_string(lower_diagonal.size()) +
                       ", upper: " + std::to_string(upper_diagonal.size()) +
                       ", rhs: " + std::to_string(rhs.size()));
}

#endif // TRIDIAGONAL_UTILS_HPP
