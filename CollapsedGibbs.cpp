

// __/\\\\____________/\\\\__/\\\\\\\\\\\\\\\__/\\\\____________/\\\\_______________/\\\\\\\\\\\____/\\\\\\\\\\\\\____/\\\\____________/\\\\_        
//  _\/\\\\\\________/\\\\\\_\/\\\///////////__\/\\\\\\________/\\\\\\_____________/\\\/////////\\\_\/\\\/////////\\\_\/\\\\\\________/\\\\\\_       
//   _\/\\\//\\\____/\\\//\\\_\/\\\_____________\/\\\//\\\____/\\\//\\\____________\//\\\______\///__\/\\\_______\/\\\_\/\\\//\\\____/\\\//\\\_      
//    _\/\\\\///\\\/\\\/_\/\\\_\/\\\\\\\\\\\_____\/\\\\///\\\/\\\/_\/\\\_____________\////\\\_________\/\\\\\\\\\\\\\\__\/\\\\///\\\/\\\/_\/\\\_     
//     _\/\\\__\///\\\/___\/\\\_\/\\\///////______\/\\\__\///\\\/___\/\\\________________\////\\\______\/\\\/////////\\\_\/\\\__\///\\\/___\/\\\_    
//      _\/\\\____\///_____\/\\\_\/\\\_____________\/\\\____\///_____\/\\\___________________\////\\\___\/\\\_______\/\\\_\/\\\____\///_____\/\\\_   
//       _\/\\\_____________\/\\\_\/\\\_____________\/\\\_____________\/\\\____________/\\\______\//\\\__\/\\\_______\/\\\_\/\\\_____________\/\\\_  
//        _\/\\\_____________\/\\\_\/\\\_____________\/\\\_____________\/\\\___________\///\\\\\\\\\\\/___\/\\\\\\\\\\\\\/__\/\\\_____________\/\\\_ 
//         _\///______________\///__\///______________\///______________\///______________\///////////_____\/////////////____\///______________\///__


#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include "Eigen/Dense"
#include <set>
#include <sstream>
#include <string>
#include <limits>
#include <algorithm>
#include <numeric>
#include "Beta.cpp"


using namespace std;
using namespace Eigen;


class CollapsedGibbsSampler {

private:

    vector<char> Y;
    int n, k, k_init;
    vector<int> X;
    double gamma, a, b;
    MatrixXd W; // It is called eta in the thesis
    Eigen::MatrixXd one_minus_W, log_W, log_one_minus_W;

    MatrixXi Abar0, Abar1;
    bool need_to_update_Abar = true;
    vector<int> mod;
    std::mt19937 gen;

    std::vector<double> precalculated_log_gamma_edge;
    std::vector<double> precalculated_log_gamma_non_edge;
    std::vector<double> precalculated_log_gamma_total;
    std::vector<double> precalculated_log_kx, precalculated_log_int;            // I could remove precalculated_log_int since it is the first row of precalculated_log_kx
    std::vector<double> precalculated_log_poisson;

    std::vector<int> matrix_edges, matrix_non_edges;
    std::vector<double> pre_Vn;

    int Vn_counter;
    int kmax;
    double correction;


public:

    CollapsedGibbsSampler(const vector<char>& Y, double gamma = 1.0, double a = 1.0, double b = 1.0, int k_init = 10)
        : Y(Y), gamma(gamma), a(a), b(b), k_init(k_init) {
        n = static_cast<int>(sqrt(Y.size()));
        X.resize(n);
        mod.resize(k_init);
        random_device rd;
        mt19937 gen(rd());
        k = k_init;
        kmax = n + 2;
    }


    vector<int> sample(int tmax) {
        initialize();
        std::random_device rd;
        std::mt19937 gen(rd());
        sftrabbit::beta_distribution<> beta(a, b);
        update_Abar();
        for (int t = 0; t < tmax; ++t) {
            step1();
            for (int i = 0; i < n; ++i) {
                step2(i);
            }
        }
        step1();
        cout << "predicted k: " << k << endl;
        print_vector(mod, "Number of nodes in each class");
        return X;
    }


private:

    void diminish(int index){
        mod.erase(mod.begin() + index); 
        for (int i = 0; i < n; ++i) {
            if (X[i] > index) {
                X[i]--;
            }
        }
        k--;
    }

    void initialize(){
        uniform_int_distribution<> dist(0, k_init - 1);
        for (int i = 0; i < n; ++i) {
            X[i] = dist(gen);
            mod[X[i]]++;
        }
        for (int index = k_init - 1; index >= 0; index--) {
            if (mod[index] == 0) {
                diminish(index);
            }
        }
        correction = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);

        W = MatrixXd::Zero(k, k);
        precalculated_log_gamma_edge.resize(n);
        precalculated_log_gamma_non_edge.resize(n);
        precalculated_log_gamma_total.resize(n);

        for(int t = 0; t < n; ++t){
            precalculated_log_gamma_edge[t] = std::lgamma(t + a);
            precalculated_log_gamma_non_edge[t] = std::lgamma(t + b);
            precalculated_log_gamma_total[t] = std::lgamma(t + a + b);
        }
        precalculated_log_int.resize(kmax);
        precalculated_log_int[0] = -numeric_limits<double>::infinity();
        for(int t = 1; t < kmax; ++t){
            precalculated_log_int[t] = std::log(t);
        }
        precalculated_log_kx.resize(kmax * n);
        for(int kappa = 0; kappa < kmax; ++kappa){
            for(int x = 0; x < n; ++x){
                precalculated_log_kx[kappa * n + x] = std::log(kappa * gamma + x);
            }
        }

        precalculated_log_poisson.resize(kmax, 0);
        for(int kk = 0; kk < kmax; ++kk){
            precalculated_log_poisson[kk] = -1 - std::lgamma(kk + 1);
        }

        Vn_counter = k + 1;
        pre_Vn.resize(Vn_counter + 1, 0);
        for (int t = 0; t < Vn_counter + 1; ++t) {
            pre_Vn[t] = Vn(t);
        }
        update_full_edges();
    }


    void update_Abar() {
        Abar0 = MatrixXi::Zero(k, k);
        Abar1 = MatrixXi::Zero(k, k);

        for (int i = 0; i < n; ++i) {
            for(int r = 0; r < k; ++r){
                Abar0(X[i], r) += matrix_non_edges[r * n + i];
                Abar1(X[i], r) += matrix_edges[r * n + i];
            }
        }
        for(int r = 0; r < k; ++r){
            Abar0(r,r) /= 2;
            Abar1(r,r) /= 2;
        }
        need_to_update_Abar = false;

    }


    void fast_update_Abar(int i, int old_class, int new_class) {
        for(int r = 0; r < k; ++r){
            Abar0(old_class, r) -= matrix_non_edges[r * n + i];
            Abar1(old_class, r) -= matrix_edges[r * n + i];
            Abar0(new_class, r) += matrix_non_edges[r * n + i];
            Abar1(new_class, r) += matrix_edges[r * n + i];
            if(new_class != r){
                Abar0(r, new_class) += matrix_non_edges[r * n + i];
                Abar1(r, new_class) += matrix_edges[r * n + i];
            }
            if (old_class != r){
                Abar0(r, old_class) -= matrix_non_edges[r * n + i];
                Abar1(r, old_class) -= matrix_edges[r * n + i];
            }
        }

    }
    

    void add_W_temp() {
        sftrabbit::beta_distribution<> beta(a, b);
        Eigen::VectorXd newVec(k);
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            newVec[k_idx] = beta(gen);
        }  
        addRowAndColumn(W, Map<VectorXd>(newVec.data(), newVec.size()));    
        update_log_W(); 

    }


    void addRowAndColumn(Eigen::MatrixXd& mat, const Eigen::VectorXd& newRow) {
        int newSize = mat.rows() + 1;
        
        // Ridimensiona la matrice per aggiungere una nuova riga e colonna
        mat.conservativeResize(newSize, newSize);

        // Aggiungi la nuova riga (escluso l'elemento diagonale)
        mat.row(newSize - 1) = newRow;

        // Aggiungi la nuova colonna (escluso l'elemento diagonale)
        mat.col(newSize - 1) = newRow;
    }


    void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
        {
            unsigned int numRows = matrix.rows()-1;
            unsigned int numCols = matrix.cols();

            if( rowToRemove < numRows )
                matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

            matrix.conservativeResize(numRows,numCols);
        }


    void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
        {
            unsigned int numRows = matrix.rows();
            unsigned int numCols = matrix.cols()-1;

            if( colToRemove < numCols )
                matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

            matrix.conservativeResize(numRows,numCols);
        }


    void step1() {
        W = MatrixXd::Zero(k, k);
        if(need_to_update_Abar){
            update_Abar();
        }
        double value;
        for (int r = 0; r < k; ++r) {
            for (int s = r; s < k; ++s) {
                sftrabbit::beta_distribution<> mod_beta(Abar1(r, s) + a, Abar0(r,s) + b);
                value = mod_beta(gen);
                W(r, s) = value;
                W(s, r) = value;
            }
        }
        update_log_W();
    }


    void step2(int i) {
        int past = X[i];
        mod[past]--;
        vector<double> log_probas(k + 1, 0);
        log_probas = log_proba_existing_tables(i);
        log_probas[k] = log_proba_new_table(i, mod[past]);
        normalize_proba(log_probas);
        discrete_distribution<int> dist(log_probas.begin(), log_probas.end());
        int index = dist(gen);
        if(index == k){
            if (mod[past] == 0){
                // go back to the old class (that otherwise would be removed)
                X[i] = past;
                mod[past]++;
                return;
            } else {
                // new class
                X[i] = k;
                mod.push_back(1);
                k++;
                add_W_temp();
                need_to_update_Abar = true;
                add_edges(i, past, X[i]);
                if(k > Vn_counter - 1){
                    Vn_counter += 1;
                    pre_Vn.push_back(Vn(Vn_counter));
                }
            }
        } else {
            // Existing class
            X[i] = index;
            mod[X[i]]++;
            if (mod[past] == 0) {
                removeRow(W, past);
                removeColumn(W, past);
                diminish(past);
                update_log_W();
                need_to_update_Abar = true;
                // remove_edges(i, past, X[i]);
                update_full_edges();
            } else if(index != past){
                if (!need_to_update_Abar){
                    fast_update_Abar(i, past, index);
                }
                update_edges_i(i, past, X[i]);
            }
        }
    }


    void update_edges_i(int i, int old_class, int new_class){
        //update edges and non-edges
        // non il massimo a livello di memoria
        for(int j = 0; j < n; ++j){
            if (j != i){
                if(Y[i * n + j]){
                    matrix_edges[old_class * n + j] -= 1;
                    matrix_edges[new_class * n + j] += 1;
                } else {
                    matrix_non_edges[old_class * n + j] -= 1;
                    matrix_non_edges[new_class * n + j] += 1;
                }
            }
        }
    }


    void add_edges(int i, int old_class, int new_class){
              // k è già stato modificato? assumiamo di si
        matrix_edges.resize(k * n, 0);
        matrix_non_edges.resize(k * n, 0);
        update_edges_i(i, old_class, new_class);
    }


    void update_full_edges(){
        matrix_edges.clear();
        matrix_non_edges.clear();
        matrix_edges.resize(k * n, 0);
        matrix_non_edges.resize(k * n, 0);

        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                if (j != i){
                    if(Y[i * n + j]){
                        matrix_edges[X[j] * n + i] += 1;
                    } else {
                        matrix_non_edges[X[j] * n + i] += 1;
                    }
                }
            }
        }
    }


    std::vector<double> log_proba_existing_tables(int i){
        std::vector<double> log_probas(k + 1, 0);
        for (int r = 0; r < k; ++r){
            if (mod[r] == 0){
                log_probas[r] = -numeric_limits<double>::infinity();
            } else {
            log_probas[r] = precalculated_log_kx[n + mod[r]];
            }
        } 
        for(int r = 0; r < k; ++r){
            for(int s = 0; s < k; ++s){
                log_probas[r] += matrix_edges[s * n + i] * log_W(s, r);
                log_probas[r] += matrix_non_edges[s * n + i] * log_one_minus_W(s, r);
            }
        }
        return log_probas;
    }

   
    void update_log_W(){
        one_minus_W.conservativeResize(k, k);
        log_W.conservativeResize(k, k);
        log_one_minus_W.conservativeResize(k, k);
        one_minus_W = Eigen::MatrixXd::Ones(k, k) - W;
        log_W = W.array().log();
        log_one_minus_W = one_minus_W.array().log();
    }


    double log_proba_new_table(int i, int mod_past) {
        double temp = 0;
        int t = mod_past == 0 ? k - 1 : k;
        temp += pre_Vn[t + 1];
        temp -= pre_Vn[t];
        temp += precalculated_log_kx[n];
        temp += m(i);
        return temp;
    }


    double Vn(int t) {
        vector<double> temp(kmax - 1, 0);
        for (int kk = 1; kk < kmax; ++kk) {
            if (kk < t) {
                temp[kk - 1] = -numeric_limits<double>::infinity();
            } else {
                if (t != 0) {
                    for (int x = 0; x < t; ++x) {
                        temp[kk - 1] += precalculated_log_int[kk - x];
                    }
                }
                for (int x = 0; x < n; ++x) {
                    temp[kk - 1] -= precalculated_log_kx[kk * n + x];
                }

                temp[kk - 1] += precalculated_log_poisson[kk - 1];

            }
        }
        return logsumexp(temp);
    }


    double m(int i) {
        double temp = 0;
        // protrei farlo esterno, o potrei lasciarlo automatico da precalculated[0]
        temp -= k* correction;
        for (int r = 0; r < k; ++r) {
            if (mod[r] == 0) {
                // in realtà penso sia automatico
                temp += correction;
                continue;
            }
            temp += precalculated_log_gamma_edge[matrix_edges[r * n + i]];
            temp += precalculated_log_gamma_non_edge[matrix_non_edges[r * n + i]]; 
            temp -= precalculated_log_gamma_total[matrix_edges[r * n + i] + matrix_non_edges[r * n + i]];
        }
        return temp;
    }


    double logsumexp(const std::vector<double>& x) {
        double c = *std::max_element(x.begin(), x.end());
        double sum_exp = 0.0;
        for (double val : x) {
            sum_exp += std::exp(val - c);
        }
        return c + std::log(sum_exp);
    }


    void normalize_proba( std::vector<double>& x) {
        // SIMD?
        double lse = logsumexp(x);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::exp(x[i] - lse);
        }
    }

};