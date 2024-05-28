#include <crow.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

using json = nlohmann::json;
using namespace Eigen;

std::pair<VectorXcd, MatrixXcd> eigenvalues_and_vectors(MatrixXd matrix) {
    EigenSolver<MatrixXd> es(matrix);
    return {es.eigenvalues(), es.eigenvectors()};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd> lu_decomposition(MatrixXd matrix) {
    FullPivLU<MatrixXd> lu(matrix);
    return {lu.permutationP().toDenseMatrix(), lu.matrixL(), lu.matrixU()};
}

std::pair<MatrixXd, MatrixXd> cholesky_decomposition(MatrixXd matrix) {
    LLT<MatrixXd> llt(matrix);
    if (llt.info() == Success) {
        MatrixXd L = llt.matrixL();
        return {L, L.transpose()};
    } else {
        return {MatrixXd(), MatrixXd()};
    }
}

std::pair<MatrixXd, MatrixXd> doolittle_decomposition(MatrixXd matrix) {
    int n = matrix.rows();
    MatrixXd L = MatrixXd::Zero(n, n);
    MatrixXd U = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L(i, j) * U(j, k);
            }
            U(i, k) = matrix(i, k) - sum;
        }

        for (int k = i; k < n; k++) {
            if (i == k) {
                L(i, i) = 1.0;
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += L(k, j) * U(j, i);
                }
                L(k, i) = (matrix(k, i) - sum) / U(i, i);
            }
        }
    }
    return {L, U};
}

std::pair<MatrixXd, MatrixXd> crout_decomposition(MatrixXd matrix) {
    int n = matrix.rows();
    MatrixXd L = MatrixXd::Zero(n, n);
    MatrixXd U = MatrixXd::Zero(n, n);

    for (int j = 0; j < n; j++) {
        U(j, j) = 1.0;
        for (int i = j; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L(i, k) * U(k, j);
            }
            L(i, j) = matrix(i, j) - sum;
        }
        for (int i = j + 1; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L(j, k) * U(k, i);
            }
            U(j, i) = (matrix(j, i) - sum) / L(j, j);
        }
    }
    return {L, U};
}

int main() {
    crow::SimpleApp app;

    CROW_ROUTE(app, "/calculate").methods("POST"_method)
    ([](const crow::request& req) {
        auto data = json::parse(req.body);
        auto matrix_data = data["matrix"].get<std::vector<std::vector<double>>>();
        std::string method = data["method"].get<std::string>();

        int n = matrix_data.size();
        MatrixXd matrix(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix(i, j) = matrix_data[i][j];
            }
        }

        json result;

        if (method == "eigen") {
            auto [eigenvalues, eigenvectors] = eigenvalues_and_vectors(matrix);
            result["eigenvalues"] = std::vector<double>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
            result["eigenvectors"] = std::vector<double>(eigenvectors.data(), eigenvectors.data() + eigenvectors.size());
        } else if (method == "lu") {
            auto [P, L, U] = lu_decomposition(matrix);
            result["P"] = std::vector<double>(P.data(), P.data() + P.size());
            result["L"] = std::vector<double>(L.data(), L.data() + L.size());
            result["U"] = std::vector<double>(U.data(), U.data() + U.size());
        } else if (method == "cholesky") {
            auto [L, U] = cholesky_decomposition(matrix);
            if (L.size() > 0) {
                result["L"] = std::vector<double>(L.data(), L.data() + L.size());
                result["L.T"] = std::vector<double>(U.data(), U.data() + U.size());
            } else {
                result = "Cholesky decomposition not applicable.";
            }
        } else if (method == "doolittle") {
            auto [L, U] = doolittle_decomposition(matrix);
            result["L"] = std::vector<double>(L.data(), L.data() + L.size());
            result["U"] = std::vector<double>(U.data(), U.data() + U.size());
        } else if (method == "crout") {
            auto [L, U] = crout_decomposition(matrix);
            result["L"] = std::vector<double>(L.data(), L.data() + L.size());
            result["U"] = std::vector<double>(U.data(), U.data() + U.size());
        } else {
            result = "Invalid method.";
        }

        return crow::response(result.dump());
    });

    app.port(8080).multithreaded().run();
}
