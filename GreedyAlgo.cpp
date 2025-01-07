

//  ____________/\\\\\\\\\\\\___________/\\\\\\\\\_______________/\\\\\\\\\\\____/\\\\\\\\\\\\\____/\\\\____________/\\\\___________         
//   ___________\/\\\////////\\\______/\\\////////______________/\\\/////////\\\_\/\\\/////////\\\_\/\\\\\\________/\\\\\\___________        
//    ___________\/\\\______\//\\\___/\\\/______________________\//\\\______\///__\/\\\_______\/\\\_\/\\\//\\\____/\\\//\\\___________       
//     ___________\/\\\_______\/\\\__/\\\_________________________\////\\\_________\/\\\\\\\\\\\\\\__\/\\\\///\\\/\\\/_\/\\\___________      
//      ___________\/\\\_______\/\\\_\/\\\____________________________\////\\\______\/\\\/////////\\\_\/\\\__\///\\\/___\/\\\___________     
//       ___________\/\\\_______\/\\\_\//\\\______________________________\////\\\___\/\\\_______\/\\\_\/\\\____\///_____\/\\\___________    
//        ___________\/\\\_______/\\\___\///\\\_____________________/\\\______\//\\\__\/\\\_______\/\\\_\/\\\_____________\/\\\___________   
//         ___________\/\\\\\\\\\\\\/______\////\\\\\\\\\___________\///\\\\\\\\\\\/___\/\\\\\\\\\\\\\/__\/\\\_____________\/\\\___________  
//          ___________\////////////___________\/////////______________\///////////_____\/////////////____\///______________\///____________ 


#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include "Eigen/Dense"
#include <set>
#include <limits>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::cout;
using std::endl;

struct Result {
    std::vector<int> config;
    double score;
};

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





class GreedyAlgorithm {
// private:
private:
    const std::vector<char>& Y;
    int n, k;
    std::vector<int> X;      // Class assignments
    std::vector<int> degree;  //
    std::vector<int> u;
    std::vector<int> m;
    std::vector<int> kappa;      //stub
    std::vector<int> kit; 
    std::vector<int> m_saved, kappa_saved, kit_saved;

    std::vector<double> precalculated_b;
    
    


public:
    // Costruttore
    GreedyAlgorithm(const std::vector<char>& Y, int k) : Y(Y), k(k) {
        n = sqrt(Y.size());
        degree.resize(n, 0);
        u.resize(n, 0);
        X.resize(n);
        m.resize(k * k, 0);
        kappa.resize(k);
        kit.resize(n * k, 0);

    }

    // Funzione infer
    Result infer(const int iterations = 1) {
        Result final_result;
        final_result.score = -std::numeric_limits<double>::infinity();
        Result temp_result;
        for(int t = 0; t < iterations; ++t) {
            // cout << "t: " << t << endl;
            initialize();
            run(temp_result);
            // cout << temp_result.score << endl;
            if (temp_result.score > final_result.score) {
                final_result = temp_result;
            }
        }
        

        return final_result;
    }

    // Single infer function
    Result single_infer() {
        Result temp_result;
        initialize();
        run(temp_result);
        return temp_result;
    }

private:
    void initialize() {
        // Random initialization of X
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, k - 1);
        for (int i = 0; i < n; ++i) {
            X[i] = dis(gen);
        }
        
        // Compute degrees and self-edges
		for (int i = 0; i < n; ++i) {
            u[i] = Y[i*n + i] / 2;
            for(int j = 0; j < n; ++j){
                degree[i] += Y[i*n + j];
            }
        }
        int edges = 0;
        for (int i = 0; i < n; ++i) {
            edges += degree[i];
        }
        int max_degree = *std::max_element(degree.begin(), degree.end());

        precalculated_b.resize( edges + 2 * max_degree);
        precalculated_b[0] = 0;
        for (int i = 1; i <  edges + 2 * max_degree ; ++i) {
            precalculated_b[i] = i * std::log(i);
        }
        update();
    }



    void update() {
        // Update m
        for(int r = 0; r < k; ++r){
            for(int s = 0; s < k; ++s){
                m[r*k + s] = 0;
            }
        }
        for (int j = 0; j < n; ++j){
            for (int i = 0; i < n; ++i) {
                m[X[i]*k + X[j]] += Y[i*n + j];
            }
        }

        // Update kappa
        for (int r = 0; r < k; ++r) {
            kappa[r] = 0;
            for (int s = 0; s < k; ++s) {
                kappa[r] += m[r*k + s];
            }
        }

        // Update kit
        for (int i = 0; i < n; ++i) {
            for(int r = 0; r < k; ++r){
                kit[i*k + r] = 0;
            }
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    kit[i * k + X[j]] += Y[i*n + j];
                }
            }
        }
    }

    void run(Result& final_res) {
        final_res = Result{X, score()};         // result the run function will return
        double max_change;                      // stores the maximum change in loglikelihood
        double change;                          // temporary change in loglikelihood
        double current_score = final_res.score;  
        bool nochanges;                  // flag to check if there are no changes in the current iteration   
        while (true) {
            nochanges = true;
            // TODO: Inefficient implementation, change data structure
            // TODO: l'ho cambiata, forse è un filo meglio (FORSE), ma è comunque molto inefficiente. 
            std::vector<int> available_nodes;
            for(int i = 0; i < n; ++i)
                available_nodes.push_back(i);

            // current_score = score();

            for (int t = 0; t < n; ++t) {
                max_change = -std::numeric_limits<double>::infinity();              
                int node = -1;
                int new_class = -1;

                // loop over all available nodes
                // for (int i : nodes) {
                for(int i : available_nodes){
                    // loop over all classes (NB, they should be different !!!!)
                    for (int s = 0; s < k; ++s) {
                        // la metto che sia differente
                        if (X[i] == s) //questo potrebbe essere un problema alla pipeline
                            continue;
                        change = change_in_loglikel(i, X[i], s);
                        if (change > max_change) {
                            node = i;
                            new_class = s;
                            max_change = change;
                        }
                    }
                }
                move(node, new_class);
                
                // in teoria avrebbe più senso fare così, ma non si sa perchè va più lento
                // current_score += max_change;
                current_score = score();

                if (current_score > final_res.score) {
                    final_res.config = X;
                    final_res.score = current_score;
                    m_saved = m;
                    kappa_saved = kappa;
                    kit_saved = kit;
                    nochanges = false;
                }

                // nodes.erase(node);
                available_nodes.erase(std::lower_bound(available_nodes.begin(), available_nodes.end(), node));
            }

            if (nochanges) {
                return;
            } else {
                X = final_res.config;
                m = m_saved;
                kappa = kappa_saved;
                kit = kit_saved;
            }
        }
    }

    void move(int node, int new_class) {
        int old_class = X[node];
        X[node] = new_class;
        // Fast update
        int Y_i_node;
        for (int i = 0; i < n; ++i) {
            if (i != node) {
                Y_i_node = Y[node * n + i];
                kit[i * k + old_class]  -= Y_i_node;
                kit[i * k + new_class] += Y_i_node;

                // dato che m è simmetrica, potrei fare solo la metà
                m[old_class * k + X[i]] -= Y_i_node;
                m[X[i] * k + old_class] -= Y_i_node;
                m[new_class * k + X[i]] += Y_i_node;
                m[X[i] * k + new_class] += Y_i_node;
            }
        }

        m[old_class * k + old_class] -= Y[node * n + node];
        m[new_class * k + new_class] += Y[node * n + node];

        kappa[old_class] -= degree[node];
        kappa[new_class] += degree[node];
    }

    double score() {
        double temp = 0.0;
        for (int r = 0; r < k; ++r) {
            for (int s = 0; s < k; ++s) {
                if (m[r * k + s] > 0) {
                    // const double epsilon = 1e-10; // Valore minimo positivo per evitare log di valori non validi
                    double ratio = static_cast<double>(m[r * k + s]) / static_cast<double>(kappa[r]);
                    ratio /= static_cast<double>(kappa[s]);     
                    // ratio = std::max(ratio, epsilon); 
                    temp += m[r * k + s] * log(ratio);

                    // temp += precalculated_b[m[r * k + s]] - m[r * k + s] * log (kappa[r]) - m[r * k + s] * log (kappa[s]);
                    

                }
            }
        }
        return temp;
    }



    double change_in_loglikel(int i, int r, int s) {
        // if (r == s) return -std::numeric_limits<double>::infinity();
        double temp = 0.0;
        for (int t = 0; t < k; ++t) {
            if (t != r && t != s) {
                temp += 2 * precalculated_b[m[r * k + t] - kit[i*k + t]] - 2 * precalculated_b[m[r * k + t]];
                temp += 2 * precalculated_b[m[s * k + t] + kit[i * k + t]] - 2 * precalculated_b[m[s * k + t]];
            }
        }
        temp += 2 * precalculated_b[m[r * k + s] + kit[i * k + r] - kit[i * k + s]] - 2 * precalculated_b[m[r * k + s] ];
        temp += precalculated_b[m[r * k + r] - 2 * (kit[i * k + r] + u[i])] - precalculated_b[m[r * k + r] ];
        temp += precalculated_b[m[s * k + s] + 2 * (kit[i * k + s] + u[i])] - precalculated_b[m[s * k + s] ];
        temp += -2 * precalculated_b[kappa[r] - degree[i]] + 2 * precalculated_b[kappa[r]];
        temp += - 2 * precalculated_b[kappa[s] + degree[i]] + 2 * precalculated_b[kappa[s]];
        return temp;
    }

 };
