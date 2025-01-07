#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include "Eigen/Dense"
#include <omp.h> // Per OpenMP
#include <ctime>
#include "CollapsedGibbs.cpp"



using std::vector;
using std::cout;
using std::endl;



class StochasticBlockModel {
public:
    int n; // number of nodes
    int k; // number of classes
    vector<double> proba_classes; // prior probabilities on the classes
    Eigen::MatrixXd eta; // eta
    vector<int> X; // class assignments for nodes
    vector<char> Y; // adjacency matrix 


    // Constructor
    StochasticBlockModel(const int n, const int k, const vector<double> proba_classes, const Eigen::MatrixXd eta)
        : n(n), k(k), proba_classes(proba_classes), eta(eta) { 
        Y.resize(n * n, 0);
        std::default_random_engine generator(static_cast<long unsigned int>(std::time(0)));
        std::discrete_distribution<int> distribution(proba_classes.begin(), proba_classes.end());
        
        // Assign classes to nodes
        for (int i = 0; i < n; ++i) {
            X.push_back(distribution(generator));
        }

        // Initialize adjacency matrix with zeros        
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (uniform(generator) < eta(X[i], X[j])) {
                    Y[i * n + j] = 1;
                    Y[j * n + i] = 1;
                }
            }
        }
    }

    // Print adjacency matrix and class assignments
    void print() const {
        std::cout << "Adj matrix\n";
        int degree;
        for (int i = 0; i < n; ++i) {
            degree = 0;
            for (int j = 0; j < n; ++j) {
                std::cout << static_cast<int>(Y[i*n + j]) << " ";
                degree += Y[i*n + j];
            }
            std::cout << "   " << degree;
            std::cout << "\n";
        }

        std::cout << "\nClasses:\n";
        for (int val : X) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

};


 
// Esempio di utilizzo
int main(int argc, char* argv[]) {
    int n; // number of nodes
    int expected_degree;
    int iterations;
    if (argc > 1) {
        try {
            // Converte il parametro in un intero
            n = std::stoi(argv[1]); 
            if (n <= 0) {
                std::cerr << "Errore: il numero di nodi deve essere maggiore di 0!" << std::endl;
                return 1;
            }
            std::cout << "Number of nodes: " << n << std::endl;
        } catch (const std::invalid_argument& e) {
            std::cerr << "Errore: il parametro passato non è un intero valido!" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Errore: il numero è troppo grande per essere rappresentato come int!" << std::endl;
        }
        if (argc > 2) {
            try {
            // Converte il parametro in un intero
            expected_degree = std::stoi(argv[2]); 
            if (expected_degree <= 0) {
                std::cerr << "Expected degree must be greater than 0!" << std::endl;
                return 1;
            }
            std::cout << "Expected degree: " << expected_degree << std::endl;
        } catch (const std::invalid_argument& e) {
            std::cerr << "Errore: il parametro passato non è un intero valido!" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Errore: il numero è troppo grande per essere rappresentato come int!" << std::endl;
        }
        if (argc > 3) {
            try {
            // Converte il parametro in un intero
            iterations = std::stoi(argv[3]); 
            if (iterations <= 0) {
                std::cerr << "the number of iterations must be greater than 0!" << std::endl;
                return 1;
            }
            std::cout << "Iterations: " << iterations << std::endl;
        } catch (const std::invalid_argument& e) {
            std::cerr << "Errore: il parametro passato non è un intero valido!" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Errore: il numero è troppo grande per essere rappresentato come int!" << std::endl;
        }
        } else{
            std::cerr << "Manca il numero di iterazioni" << std::endl;
            return 1;
        }}
        else {
            std::cerr << "Manca expected degree" << std::endl;
            return 1;
        }}
         else {
        std::cout << "Nessun parametro passato!" << std::endl;
        return 1;
    }

    // creation of the graph 

    int k = 4;
    vector<double> proba_classes = {0.25, 0.25, 0.25, 0.25};
    Eigen::MatrixXd eta(k,k);

    Eigen::MatrixXd pre_eta(k,k);
    double gamma = 4;
    pre_eta << gamma, 1, 1, 1,
            1, gamma, 1, 1,
            1, 1, gamma, 1,
            1, 1, 1, gamma;
    double divisor = static_cast<double>(n * (gamma + k - 1))/(expected_degree * k);
    eta = pre_eta / divisor;

    cout << eta << endl;
    // Eigen::MatrixXd W(k,k);
    // W = eta;
    // W.diagonal()*= 2;
    
    StochasticBlockModel sbm(n, k, proba_classes, eta);   

    // inference part

    CollapsedGibbsSampler cgs(sbm.Y);
    std::vector<int> prediction = cgs.sample(iterations);



    double nmi_score = calculateNMI(sbm.X, prediction );
    std::cout << "NMI Score: " << nmi_score << std::endl;
    cout << "fine" << endl;
    return 0;
}

