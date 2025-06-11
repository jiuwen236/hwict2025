#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <tuple>       // 为了 std::tuple, std::make_tuple, std::get (原代码中已使用)
#include <stdexcept>   // 为了 std::runtime_error (用于动态异常说明的示例)

// 5034326 45572021 2025-05-30 22:31:36

using namespace std;

// --- 添加的代码：在C++11下编译（有弃用警告），在C++17下编译失败 ---
// 动态异常说明 (非空的 throw(...) ) 在C++11中被弃用，并在C++17中被移除。
// 下面的函数声明和定义将导致在 -std=c++17 下编译错误。
void function_using_deprecated_feature() throw(std::runtime_error); // 声明

void function_using_deprecated_feature() throw(std::runtime_error) { // 定义
    // 此函数的内容对于C++17中的编译失败并不重要，关键在于其签名。
    // 在C++11中，它会编译通过，但编译器通常会发出关于动态异常说明已弃用的警告。
    // 为避免编译器对空函数体或未使用变量发出不必要的警告（在C++11下），可以添加一些占位代码。
    volatile int touch = 0; (void)touch; // 例如，确保变量被“使用”
}
// --- 添加的代码结束 ---

struct Server {
    int g, k, m;  // NPU数量，推理速度系数，显存大小
};

struct User {
    int s, e, cnt;  // 开始时间，结束时间，样本数量
};

int main() {
    // 可选地调用一下函数，以确保编译器处理它。
    // 不过，仅函数定义本身就足以在C++17下引发编译失败。
    // 如果调用，需确保其不影响主逻辑 (此示例函数为空，无副作用)。
    try {
        function_using_deprecated_feature();
    } catch (const std::runtime_error& e) {
        // 处理异常，虽然此函数实际上不抛出
    }


    ios_base::sync_with_stdio(false); // 优化cin/cout性能
    cin.tie(NULL);                   // 优化cin性能

    int N;
    cin >> N;

    vector<Server> servers(N);
    for (int i = 0; i < N; i++) {
        cin >> servers[i].g >> servers[i].k >> servers[i].m;
    }

    int M;
    cin >> M;

    vector<User> users(M);
    for (int i = 0; i < M; i++) {
        cin >> users[i].s >> users[i].e >> users[i].cnt;
    }

    vector<vector<int>> latency(N, vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cin >> latency[i][j];
        }
    }

    int a, b;
    cin >> a >> b;

    // 简单的调度策略：为每个用户选择最合适的服务器
    for (int i = 0; i < M; i++) {
        int best_server = 0;
        int min_latency = latency[0][i];

        // 选择延迟最小的服务器
        for (int j = 1; j < N; j++) {
            if (latency[j][i] < min_latency) {
                min_latency = latency[j][i];
                best_server = j;
            }
        }

        // 计算最大可用的batch size（不超过显存限制）
        int max_batch = (servers[best_server].m - b) / a;
        if (max_batch <= 0) max_batch = 1;
        if (max_batch > 1000) max_batch = 1000;

        // 计算需要多少次请求
        int remaining = users[i].cnt;
        vector<tuple<int, int, int, int>> requests; // time, server, npu, batch

        int current_time = users[i].s;
        int npu = 0; // 使用第一个NPU

        while (remaining > 0) {
            int batch_size = min(remaining, max_batch);

            // 确保batch size满足显存约束
            while (a * batch_size + b > servers[best_server].m && batch_size > 1) {
                batch_size--;
            }

            if (batch_size <= 0) batch_size = 1;

            requests.push_back(make_tuple(current_time, best_server + 1, npu + 1, batch_size));
            remaining -= batch_size;

            // 计算下次发送时间（考虑延迟）
            if (remaining > 0) {
                current_time += latency[best_server][i] + 1;
            }
        }

        // 输出结果
        cout << requests.size() << "\n";
        for (size_t r_idx = 0; r_idx < requests.size(); ++r_idx) {
            cout << get<0>(requests[r_idx]) << " "
                 << get<1>(requests[r_idx]) << " "
                 << get<2>(requests[r_idx]) << " "
                 << get<3>(requests[r_idx]);
            if (r_idx < requests.size() - 1) {
                 cout << " ";
            }
        }
        cout << "\n";
    }

    return 0;
}