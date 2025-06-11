#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm>
#include <queue>
#include <filesystem>
#include <cctype>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <array>
#include <optional>
#include <cstring>
#include <climits>
#include <unordered_map>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <signal.h>

// 使用方法：
// 1. 编译本文件： g++ -O3 -march=native judge.cpp -o judge
// 2. command: ./judge [-cpp cpp_file_path]
// 3. output file: score.log (in the same directory as the cpp file)


namespace fs = std::filesystem;

// 分数详细信息结构体
struct ScoreDetails {
    double total_score;
    int K;                      // 超时用户数量
    double h_K;                 // h(K)
    double timeout_percentage;  // 超时用户百分比
    double avg_h_xi;           // h(xi) 平均值 (局部超时惩罚)
    double avg_p_mi;           // p(mi) 平均值 (迁移惩罚)
    std::vector<double> h_xi_values;  // 每个用户的h(xi)
    std::vector<double> p_mi_values;  // 每个用户的p(mi)
    std::vector<int> user_end_times;  // 每个用户的结束时间
    std::vector<int> move_counts;     // 每个用户的迁移次数
};

// 编译C++文件
void compile_cpp(const std::string& cpp_file, const std::string& exe_path, const std::string& log_file) {
    std::string cmd = "g++ -O2 -std=c++17 " + cpp_file + " -o " + exe_path;
    int status = std::system(cmd.c_str());
    if (status != 0) {
        std::ofstream f(log_file);
        f << "Compilation Error\n";
        f << "g++ returned non-zero exit status\n";
        std::cerr << "Compilation Error\n";
        exit(1);
    }
}

// 运行程序并捕获输出
std::string run_program(const std::string& exe_path, const std::string& input_file, int timeout_seconds) {
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        throw std::runtime_error("pipe() failed");
    }

    // 设置读端为非阻塞
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        throw std::runtime_error("fork() failed");
    }

    if (pid == 0) { // 子进程
        close(pipefd[0]); // 关闭读端
        
        // 重定向标准输入
        if (!input_file.empty()) {
            FILE* in_file = freopen(input_file.c_str(), "r", stdin);
            if (!in_file) {
                perror("freopen failed");
                exit(EXIT_FAILURE);
            }
        }

        // 重定向标准输出到管道
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        execl(exe_path.c_str(), exe_path.c_str(), (char*)NULL);
        perror("execl failed");
        exit(EXIT_FAILURE);
    } else { // 父进程
        close(pipefd[1]); // 关闭写端

        std::string output;
        char buffer[4096];
        ssize_t count;
        auto start_time = std::chrono::steady_clock::now();

        while (true) {
            // 检查超时
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (elapsed >= timeout_seconds) {
                kill(pid, SIGKILL);
                waitpid(pid, NULL, 0);
                throw std::runtime_error("Timeout");
            }

            // 检查子进程状态
            int status;
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == -1) {
                close(pipefd[0]);
                throw std::runtime_error("waitpid failed");
            } else if (result == pid) {
                // 子进程已退出
                break;
            }

            // 尝试读取输出
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(pipefd[0], &fds);
            struct timeval tv = {0, 100000}; // 100ms

            if (select(pipefd[0] + 1, &fds, NULL, NULL, &tv) > 0) {
                if (FD_ISSET(pipefd[0], &fds)) {
                    while ((count = read(pipefd[0], buffer, sizeof(buffer)))) {
                        if (count > 0) {
                            output.append(buffer, count);
                        } else if (count < 0) {
                            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                                break;
                            }
                            close(pipefd[0]);
                            throw std::runtime_error("read failed");
                        }
                    }
                }
            }
        }

        // 读取剩余输出
        while (true) {
            count = read(pipefd[0], buffer, sizeof(buffer));
            if (count > 0) {
                output.append(buffer, count);
            } else if (count == 0) {
                break; // EOF
            } else if (count < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 没有更多数据
                    break;
                }
                close(pipefd[0]);
                throw std::runtime_error("read failed");
            }
        }

        close(pipefd[0]);

        // 检查子进程退出状态
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Program exited with non-zero status");
        }

        return output;
    }
}

// 计算分数
ScoreDetails compute_score(const std::string& in_file, const std::string& stdout_str) {
    std::ifstream fin(in_file);
    if (!fin) {
        throw std::runtime_error("Cannot open input file: " + in_file);
    }

    // 读取输入数据
    std::vector<double> data;
    double value;
    while (fin >> value) {
        data.push_back(value);
    }
    fin.close();

    size_t idx = 0;
    int N = static_cast<int>(data[idx++]);
    std::vector<int> g, m;
    std::vector<double> k;
    for (int i = 0; i < N; i++) {
        g.push_back(static_cast<int>(data[idx++]));
        k.push_back(data[idx++]);
        m.push_back(static_cast<int>(data[idx++]));
    }

    int M = static_cast<int>(data[idx++]);
    std::vector<int> s, e, cnt;
    for (int i = 0; i < M; i++) {
        s.push_back(static_cast<int>(data[idx++]));
        e.push_back(static_cast<int>(data[idx++]));
        cnt.push_back(static_cast<int>(data[idx++]));
    }

    std::vector<std::vector<int>> latency(N, std::vector<int>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            latency[i][j] = static_cast<int>(data[idx++]);
        }
    }

    int a = static_cast<int>(data[idx++]);
    int b = static_cast<int>(data[idx++]);

    // 解析选手输出
    std::istringstream iss(stdout_str);
    std::string line;
    std::vector<std::string> out_lines;
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            out_lines.push_back(line);
        }
    }

    if (out_lines.size() != 2 * M) {
        throw std::runtime_error("Invalid Output");
    }

    struct Request {
        int user;
        int time;
        int server;
        int npu;
        int B;
        int arrival;
        int proc_time;
        int mem;
        bool started;
        bool completed;
    };

    std::vector<Request> requests;
    std::vector<std::vector<int>> V_list(M);

    for (int u = 0; u < M; u++) {
        // T_i
        std::istringstream t_iss(out_lines[2 * u]);
        int T;
        if (!(t_iss >> T) || T < 1 || T > 300) {
            throw std::runtime_error("Invalid Output");
        }

        std::istringstream items_iss(out_lines[2 * u + 1]);
        std::vector<int> items;
        int item;
        for (int j = 0; j < 4 * T; j++) {
            if (!(items_iss >> item)) {
                throw std::runtime_error("Invalid Output");
            }
            items.push_back(item);
        }

        int prev_time = -1;
        int sumB = 0;
        std::vector<int> V;

        for (int j = 0; j < T; j++) {
            int time_j = items[4 * j];
            int server_j = items[4 * j + 1];
            int npu_j = items[4 * j + 2];
            int B_j = items[4 * j + 3];

            if (time_j <= prev_time || time_j < 0 || time_j > 1000000) {
                throw std::runtime_error("Invalid Output");
            }
            if (server_j < 1 || server_j > N) {
                throw std::runtime_error("Invalid Output");
            }
            if (npu_j < 1 || npu_j > g[server_j - 1]) {
                throw std::runtime_error("Invalid Output");
            }
            if (B_j < 1 || B_j > 1000 || a * B_j + b > m[server_j - 1]) {
                throw std::runtime_error("Invalid Output");
            }
            if (time_j < s[u]) {
                throw std::runtime_error("Invalid Output");
            }

            prev_time = time_j;
            sumB += B_j;
            V.push_back(server_j * 100 + (npu_j - 1));

            requests.push_back({
                u,
                time_j,
                server_j - 1,
                npu_j - 1,
                B_j,
                0, // arrival (to be set later)
                0, // proc_time (to be set later)
                0, // mem (to be set later)
                false,
                false
            });
        }

        if (sumB != cnt[u]) {
            throw std::runtime_error("Invalid Output");
        }
        V_list[u] = V;
    }

    // 检查通信间隔约束
    std::vector<std::vector<Request*>> user_reqs(M);
    for (auto& r : requests) {
        user_reqs[r.user].push_back(&r);
    }

    for (int u = 0; u < M; u++) {
        std::sort(user_reqs[u].begin(), user_reqs[u].end(), 
                 [](const Request* a, const Request* b) { return a->time < b->time; });
        
        for (size_t i = 0; i < user_reqs[u].size() - 1; i++) {
            Request* cur = user_reqs[u][i];
            Request* nxt = user_reqs[u][i + 1];
            int delay = latency[cur->server][u];
            if (nxt->time == cur->time) {
                throw std::runtime_error("Invalid Output");
            }
            if (nxt->time < cur->time + delay + 1) {
                throw std::runtime_error("Invalid Output");
            }
        }
    }

    // 准备仿真数据
    std::map<int, std::vector<Request*>> arrival_dict;
    for (auto& r : requests) {
        r.arrival = r.time + latency[r.server][r.user];
        r.proc_time = static_cast<int>(std::ceil(r.B / (k[r.server] * std::sqrt(r.B))));
        r.mem = a * r.B + b;
        arrival_dict[r.arrival].push_back(&r);
    }

    // 服务器状态
    struct Server {
        int mem_cap;
        std::vector<int> mem_used; // per-NPU memory usage
        std::vector<std::pair<int, Request*>> active; // (completion_time, request)
        std::vector<std::vector<Request*>> queues;    // per NPU queues
    };

    std::vector<Server> servers;
    for (int i = 0; i < N; i++) {
        servers.push_back({
            m[i],
            std::vector<int>(g[i]),
            {},
            std::vector<std::vector<Request*>>(g[i])
        });
    }

    int total_reqs = requests.size();
    if (total_reqs == 0) {
        throw std::runtime_error("Invalid Output");
    }

    int done = 0;
    std::vector<int> user_end(M, 0);

    // 找到最早到达时间
    int t = arrival_dict.empty() ? 0 : arrival_dict.begin()->first;

    // 事件驱动主循环
    while (done < total_reqs) {
        // 1) 完成任务
        for (auto& srv : servers) {
            auto it = srv.active.begin();
            while (it != srv.active.end()) {
                if (it->first == t) {
                    srv.mem_used[it->second->npu] -= it->second->mem;
                    it->second->completed = true;
                    done++;
                    user_end[it->second->user] = std::max(user_end[it->second->user], t);
                    it = srv.active.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (done >= total_reqs) break;

        // 2) 新到达
        auto arr_it = arrival_dict.find(t);
        if (arr_it != arrival_dict.end()) {
            for (auto rq : arr_it->second) {
                servers[rq->server].queues[rq->npu].push_back(rq);
            }
        }

        // 3) 对刚到达的队列做排序
        if (arr_it != arrival_dict.end()) {
            for (auto& srv : servers) {
                for (auto& q : srv.queues) {
                    if (!q.empty()) {
                        std::sort(q.begin(), q.end(), [](Request* a, Request* b) {
                            if (a->arrival != b->arrival) 
                                return a->arrival < b->arrival;
                            return a->user < b->user;
                        });
                    }
                }
            }
        }

        // 4) 调度
        bool scheduled = false;
        for (auto& srv : servers) {
            for (auto& q : srv.queues) {
                std::vector<Request*> new_q;
                for (auto rq : q) {
                    if (rq->started) continue;
                    if (srv.mem_used[rq->npu] + rq->mem <= srv.mem_cap) {
                        rq->started = true;
                        int comp = t + rq->proc_time;
                        srv.active.push_back({comp, rq});
                        srv.mem_used[rq->npu] += rq->mem;
                        scheduled = true;
                    } else {
                        new_q.push_back(rq);
                    }
                }
                q = new_q;
            }
        }

        // 5) 跳到下一个事件时刻
        int next_t = INT_MAX;
        
        // 下一个到达
        auto next_arr_it = arrival_dict.upper_bound(t);
        if (next_arr_it != arrival_dict.end()) {
            next_t = std::min(next_t, next_arr_it->first);
        }
        
        // 下一个完成
        for (const auto& srv : servers) {
            for (const auto& [comp_t, _] : srv.active) {
                if (comp_t > t) {
                    next_t = std::min(next_t, comp_t);
                }
            }
        }
        
        if (next_t == INT_MAX) {
            t++;
        } else {
            t = next_t;
        }
    }

    if (done < total_reqs) {
        throw std::runtime_error("Invalid Output");
    }

    // 计算分数
    auto h = [](double x) { return std::pow(2.0, -x / 100.0); };
    auto p = [](double x) { return std::pow(2.0, -x / 200.0); };

    ScoreDetails details;
    details.K = 0;
    for (int i = 0; i < M; i++) {
        if (user_end[i] > e[i]) details.K++;
    }

    details.h_K = h(details.K);
    details.timeout_percentage = M > 0 ? (static_cast<double>(details.K) / M) * 100.0 : 0.0;
    details.user_end_times = user_end;
    details.move_counts.resize(M);
    details.h_xi_values.resize(M);
    details.p_mi_values.resize(M);

    double total = 0.0;
    double sum_h_xi = 0.0;
    double sum_p_mi = 0.0;
    
    for (int i = 0; i < M; i++) {
        double xi = static_cast<double>(user_end[i] - e[i]) / (e[i] - s[i]);
        int mi = 0;
        for (size_t j = 0; j < V_list[i].size() - 1; j++) {
            if (V_list[i][j] != V_list[i][j + 1]) mi++;
        }
        
        details.move_counts[i] = mi;
        details.h_xi_values[i] = h(xi);
        details.p_mi_values[i] = p(mi);
        
        sum_h_xi += details.h_xi_values[i];
        sum_p_mi += details.p_mi_values[i];
        total += details.h_xi_values[i] * details.p_mi_values[i];
    }

    details.avg_h_xi = M > 0 ? sum_h_xi / M : 0.0;
    details.avg_p_mi = M > 0 ? sum_p_mi / M : 0.0;
    details.total_score = details.h_K * total * 10000.0;

    return details;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    std::string cpp_path = "main.cpp";
    if (argc > 2 && std::string(argv[1]) == "-cpp") {
        cpp_path = argv[2];
    }

    // 确定基础路径和文件路径
    fs::path base_dir = fs::path(cpp_path).parent_path();
    if (base_dir.empty()) base_dir = ".";
    
    fs::path exe_path = base_dir / "main";
    fs::path log_file = base_dir / "score.log";
    
    // 获取脚本所在目录
    fs::path script_dir = fs::path(argv[0]).parent_path();
    if (script_dir.empty()) script_dir = ".";
    fs::path data_dir = script_dir / "data";

    // 编译C++文件
    compile_cpp(cpp_path, exe_path.string(), log_file.string());

    // 收集测试用例
    std::vector<fs::path> in_files;
    if (fs::exists(data_dir)) {
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            if (entry.path().extension() == ".in") {
                in_files.push_back(entry.path());
            }
        }
    }
    std::sort(in_files.begin(), in_files.end());
    
    int num_cases = in_files.size();
    double total_score = 0.0;
    double total_time = 0.0;
    double total_h_K = 0.0;
    double total_timeout_percentage = 0.0;
    double total_avg_h_xi = 0.0;
    double total_avg_p_mi = 0.0;
    
    std::vector<std::string> log_lines;
    log_lines.push_back("测试用例数量: " + std::to_string(num_cases));

    for (const auto& infile : in_files) {
        std::string name = infile.filename().string();
        double dur = 0.0;
        ScoreDetails details;
        details.total_score = 0.0;
        details.h_K = 0.0;
        details.timeout_percentage = 0.0;
        details.avg_h_xi = 0.0;
        details.avg_p_mi = 0.0;
        std::string error_msg;
        
        try {
            auto start = std::chrono::steady_clock::now();
            std::string output = run_program(exe_path.string(), infile.string(), 30);
            auto end = std::chrono::steady_clock::now();
            dur = std::chrono::duration<double>(end - start).count();
            total_time += dur;
            
            details = compute_score(infile.string(), output);
            total_score += details.total_score;
            total_h_K += details.h_K;
            total_timeout_percentage += details.timeout_percentage;
            total_avg_h_xi += details.avg_h_xi;
            total_avg_p_mi += details.avg_p_mi;
        } catch (const std::exception& e) {
            error_msg = e.what();
            if (error_msg == "Timeout") dur = 30.0;
        }

        log_lines.push_back("文件名: " + name);
        log_lines.push_back("分数: " + std::to_string(details.total_score) + 
                           ", 时间: " + std::to_string(dur) + "s" +
                           ", 全局超时惩罚: " + std::to_string(details.h_K) + "/" + std::to_string(details.timeout_percentage) + "%" +
                           ", 局部超时惩罚: " + std::to_string(details.avg_h_xi) +
                           ", 迁移惩罚: " + std::to_string(details.avg_p_mi));
        if (!error_msg.empty()) {
            log_lines.push_back("错误: " + error_msg);
        }
        log_lines.push_back("");
    }

    double avg_score = num_cases > 0 ? total_score / num_cases : 0.0;
    double avg_time = num_cases > 0 ? total_time / num_cases : 0.0;
    double avg_h_K = num_cases > 0 ? total_h_K / num_cases : 0.0;
    double avg_timeout_percentage = num_cases > 0 ? total_timeout_percentage / num_cases : 0.0;
    double avg_h_xi = num_cases > 0 ? total_avg_h_xi / num_cases : 0.0;
    double avg_p_mi = num_cases > 0 ? total_avg_p_mi / num_cases : 0.0;
    
    log_lines.push_back("总分: " + std::to_string(total_score));
    log_lines.push_back("平均分: " + std::to_string(avg_score) + 
                       ", 平均时间: " + std::to_string(avg_time) + "s" +
                       ", 全局超时惩罚: " + std::to_string(avg_h_K) + "/" + std::to_string(avg_timeout_percentage) + "%" +
                       ", 局部超时惩罚: " + std::to_string(avg_h_xi) +
                       ", 迁移惩罚: " + std::to_string(avg_p_mi));

    // 写入日志文件
    std::ofstream log_out(log_file);
    for (const auto& line : log_lines) {
        log_out << line << "\n";
    }
    
    std::cout << "总分: " << std::to_string(total_score) << "\n";
    std::cout << "平均分: " << std::to_string(avg_score) << ", 平均时间: " << std::to_string(avg_time) << "s" 
              << ", 全局超时惩罚: " << std::to_string(avg_h_K) << "/" << std::to_string(avg_timeout_percentage) << "%" 
              << ", 局部超时惩罚: " << std::to_string(avg_h_xi) 
              << ", 迁移惩罚: " << std::to_string(avg_p_mi) << "\n";
    
    return 0;
}