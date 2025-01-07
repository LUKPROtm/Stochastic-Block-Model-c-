#include <iostream>
#include <sstream>
#include <string>
#include <random>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using namespace std;

namespace sftrabbit {

  template <typename RealType = double>
  class beta_distribution
  {
    public:
      typedef RealType result_type;

      class param_type
      {
        public:
          typedef beta_distribution distribution_type;

          explicit param_type(RealType a = 2.0, RealType b = 2.0)
            : a_param(a), b_param(b) { }

          RealType a() const { return a_param; }
          RealType b() const { return b_param; }

          bool operator==(const param_type& other) const
          {
            return (a_param == other.a_param &&
                    b_param == other.b_param);
          }

          bool operator!=(const param_type& other) const
          {
            return !(*this == other);
          }

        private:
          RealType a_param, b_param;
      };

      explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
        : a_gamma(a), b_gamma(b) { }
      explicit beta_distribution(const param_type& param)
        : a_gamma(param.a()), b_gamma(param.b()) { }

      void reset() { }

      param_type param() const
      {
        return param_type(a(), b());
      }

      void param(const param_type& param)
      {
        a_gamma = gamma_dist_type(param.a());
        b_gamma = gamma_dist_type(param.b());
      }

      template <typename URNG>
      result_type operator()(URNG& engine)
      {
        return generate(engine, a_gamma, b_gamma);
      }

      template <typename URNG>
      result_type operator()(URNG& engine, const param_type& param)
      {
        gamma_dist_type a_param_gamma(param.a()),
                        b_param_gamma(param.b());
        return generate(engine, a_param_gamma, b_param_gamma); 
      }

      result_type min() const { return 0.0; }
      result_type max() const { return 1.0; }

      result_type a() const { return a_gamma.alpha(); }
      result_type b() const { return b_gamma.alpha(); }

      bool operator==(const beta_distribution<result_type>& other) const
      {
        return (param() == other.param() &&
                a_gamma == other.a_gamma &&
                b_gamma == other.b_gamma);
      }

      bool operator!=(const beta_distribution<result_type>& other) const
      {
        return !(*this == other);
      }

    private:
      typedef std::gamma_distribution<result_type> gamma_dist_type;

      gamma_dist_type a_gamma, b_gamma;

      template <typename URNG>
      result_type generate(URNG& engine,
        gamma_dist_type& x_gamma,
        gamma_dist_type& y_gamma)
      {
        result_type x = x_gamma(engine);
        return x / (x + y_gamma(engine));
      }
  };

  template <typename CharT, typename RealType>
  std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os,
    const beta_distribution<RealType>& beta)
  {
    os << "~Beta(" << beta.a() << "," << beta.b() << ")";
    return os;
  }

  template <typename CharT, typename RealType>
  std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& is,
    beta_distribution<RealType>& beta)
  {
    std::string str;
    RealType a, b;
    if (std::getline(is, str, '(') && str == "~Beta" &&
        is >> a && is.get() == ',' && is >> b && is.get() == ')') {
      beta = beta_distribution<RealType>(a, b);
    } else {
      is.setstate(std::ios::failbit);
    }
    return is;
  }

}

// int main(){
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     sftrabbit::beta_distribution<> beta(2, 2);
//     for (int i = 0; i < 10000; i++) {
//     std::cout << beta(gen) << std::endl;
//     }

// }



void printW(const std::map<int, std::map<int, double>>& W) {
    for (const auto& outer_pair : W) {
        std::cout << outer_pair.first << ": ";
        for (const auto& inner_pair : outer_pair.second) {
            std::cout << "{" << inner_pair.first << ": " << inner_pair.second << "} ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name) {
    cout << name << ": " << endl;
    for (const T& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}


double calculateNMI(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    if (labels1.size() != labels2.size()) {
        cout << "size1: " << labels1.size() << " size2: " << labels2.size() << endl;
        throw std::invalid_argument("Vectors must be of the same length");
    }

    int n = labels1.size();

    // Create contingency matrix
    std::map<int, int> label1_map, label2_map;
    int label1_count = 0, label2_count = 0;

    for (int i = 0; i < n; ++i) {
        if (label1_map.find(labels1[i]) == label1_map.end()) {
            label1_map[labels1[i]] = label1_count++;
        }
        if (label2_map.find(labels2[i]) == label2_map.end()) {
            label2_map[labels2[i]] = label2_count++;
        }
    }
    if (label1_count == 1 || label2_count == 1) {
        return 0.0;
    }

    MatrixXd contingency = MatrixXd::Zero(label1_count, label2_count);

    for (int i = 0; i < n; ++i) {
        contingency(label1_map[labels1[i]], label2_map[labels2[i]]) += 1;
    }

    // Calculate the mutual information
    double mi = 0.0;
    VectorXd row_sums = contingency.rowwise().sum();
    VectorXd col_sums = contingency.colwise().sum();
    double total_sum = contingency.sum();

    for (int i = 0; i < label1_count; ++i) {
        for (int j = 0; j < label2_count; ++j) {
            if (contingency(i, j) > 0) {
                double pij = contingency(i, j) / total_sum;
                double pi = row_sums[i] / total_sum;
                double pj = col_sums[j] / total_sum;
                mi += pij * std::log(pij / (pi * pj));
            }
        }
    }

    // Calculate the entropies
    double h1 = 0.0, h2 = 0.0;
    for (int i = 0; i < label1_count; ++i) {
        double pi = row_sums[i] / total_sum;
        if (pi > 0) {
            h1 -= pi * std::log(pi);
        }
    }
    for (int j = 0; j < label2_count; ++j) {
        double pj = col_sums[j] / total_sum;
        if (pj > 0) {
            h2 -= pj * std::log(pj);
        }
    }

    // Calculate the normalized mutual information
    double nmi = mi / std::sqrt(h1 * h2);
    return nmi;
}
