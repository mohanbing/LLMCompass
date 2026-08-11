#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Helper: Compute product of a shape vector
long long compute_product(const std::vector<long long>& shape) {
    long long product = 1;
    for (auto dim : shape) {
        product *= dim;
    }
    return product;
}

int main() {
    // Load the JSON file
    std::ifstream file("dep_graph_llama_one_layer.json");
    if (!file.is_open()) {
        std::cerr << "Could not open JSON file.\n";
        return 1;
    }

    json j;
    file >> j;

    long long total_params = 0;

    for (auto& [tensor_name, tensor_obj] : j.items()) {
        // Only look at Matmul operations with an op_1 dependency (the weight matrix)
        if (tensor_obj.contains("dep") && tensor_obj.contains("op")) {
            std::string op_type = tensor_obj["op"];
            if (op_type == "Matmul" && tensor_obj["dep"].contains("op_1")) {
                auto weight = tensor_obj["dep"]["op_1"];
                if (weight.contains("shape")) {
                    auto shape = weight["shape"];
                    std::vector<long long> dims = shape.get<std::vector<long long>>();
                    long long param_count = compute_product(dims);
                    total_params += param_count;

                    std::cout << tensor_name << " -> Matmul weight, shape: [";
                    for (size_t i = 0; i < dims.size(); ++i) {
                        std::cout << dims[i];
                        if (i < dims.size() - 1) std::cout << ", ";
                    }
                    std::cout << "], params: " << param_count << "\n";
                }
            }
        }
    }

    std::cout << "------------------------\n";
    std::cout << "Total learnable parameters: " << total_params << "\n";

    return 0;
}
