#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip> // Not strictly needed for final code, but useful for debugging

// 禁用迁移；每次请求的大小尽量装满NPU显存
// 用户按(e_i-s_i)*cnt_i排序，最小的优先；在有更高优先级用户使用NPU时，低优先级的用户不能抢占
// 最优选择：用户发送请求时会计算对每个NPU的实际推理结束时间，选择最早结束的。

// Structure to hold server type information
struct ServerType {
    int id_out; // 1-based ID for output
    int g;      // NPU count
    int k;      // Speed coefficient
    int m;      // Memory per NPU
    long long max_batch_size; // Pre-calculated max batch size for this server type
};

// Structure to hold request details for output
struct RequestDetail {
    long long send_time;
    int server_type_id_out;        // 1-based server type ID
    int npu_id_in_server_type_out; // 1-based NPU ID within its type
    int batch_size;
};

// Structure to hold user information
struct User {
    int original_id_0_indexed; // 0-based original index, used for accessing latency etc.
    long long s_time, e_time, total_samples;
    double priority_val; // Value to sort by, (e_i - s_i) * cnt_i
};

// Comparison for sorting users
bool compareUsers(const User& a, const User& b) {
    if (a.priority_val != b.priority_val) {
        return a.priority_val < b.priority_val;
    }
    return a.original_id_0_indexed < b.original_id_0_indexed; // Tie-breaking
}

long long global_mem_param_a, global_mem_param_b;

// Function to calculate inference duration
long long calculate_inference_duration_ms(int batch_size, int k_coefficient) {
    if (batch_size <= 0) return 0; 
    // Inference time: ceil(B / (k * sqrt(B))) = ceil(sqrt(B) / k)
    return (long long)std::ceil(std::sqrt((double)batch_size) / (double)k_coefficient);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int num_server_types;
    std::cin >> num_server_types;

    std::vector<ServerType> server_type_configs(num_server_types);
    for (int i = 0; i < num_server_types; ++i) {
        server_type_configs[i].id_out = i + 1; // 1-based ID
        std::cin >> server_type_configs[i].g >> server_type_configs[i].k >> server_type_configs[i].m;
    }

    int num_users;
    std::cin >> num_users;

    std::vector<User> user_configs(num_users);
    for (int i = 0; i < num_users; ++i) {
        user_configs[i].original_id_0_indexed = i;
        std::cin >> user_configs[i].s_time >> user_configs[i].e_time >> user_configs[i].total_samples;
    }

    // Latencies: latencies_matrix[server_type_idx_0_based][user_original_id_0_based]
    std::vector<std::vector<int>> latencies_matrix(num_server_types, std::vector<int>(num_users));
    for (int i = 0; i < num_server_types; ++i) {
        for (int j = 0; j < num_users; ++j) {
            std::cin >> latencies_matrix[i][j];
        }
    }

    std::cin >> global_mem_param_a >> global_mem_param_b;

    // Pre-calculate max batch size for each server type
    for (int i = 0; i < num_server_types; ++i) {
        if (global_mem_param_a == 0) { // Avoid division by zero
            server_type_configs[i].max_batch_size = 1; // Or some other appropriate fallback
        } else {
            server_type_configs[i].max_batch_size = (server_type_configs[i].m - global_mem_param_b) / global_mem_param_a;
        }
        // As per problem constraints, m_i >= 1000, b <= 200, a >= 10. So m-b >= 800. (m-b)/a >= 800/20 = 40.
        // So max_batch_size should always be positive and reasonably large.
        if (server_type_configs[i].max_batch_size <= 0) { // Defensive check
             server_type_configs[i].max_batch_size = 1; // Fallback, though unlikely to be needed
        }
    }

    // Calculate priority for users and store them for sorting
    std::vector<User> sorted_users = user_configs; // Copy to sort
    for (int i = 0; i < num_users; ++i) {
        // Using long long for priority_val to avoid potential precision issues with double if numbers are huge,
        // though double is generally fine for sorting.
        sorted_users[i].priority_val = (double)(sorted_users[i].e_time - sorted_users[i].s_time) * sorted_users[i].total_samples;
    }
    std::sort(sorted_users.begin(), sorted_users.end(), compareUsers);

    // NPU finish times: npu_finish_times_globally[server_type_idx_0_based][npu_idx_0_based_within_type]
    std::vector<std::vector<long long>> npu_finish_times_globally(num_server_types);
    for(int i = 0; i < num_server_types; ++i) {
        npu_finish_times_globally[i].resize(server_type_configs[i].g, 0);
    }

    // This vector will store the final schedules, indexed by original user ID
    std::vector<std::vector<RequestDetail>> final_user_schedules(num_users);

    // Process users based on their sorted priority
    for (const auto& current_user_ref : sorted_users) { 
        if (current_user_ref.total_samples == 0) { // Skip users with no samples
            final_user_schedules[current_user_ref.original_id_0_indexed].clear(); // Ensure empty schedule
            continue;
        }

        long long best_simulated_overall_finish_time = -1; // Use -1 to indicate not yet found
        int chosen_server_type_0_idx = -1;
        int chosen_npu_0_idx_in_type = -1;

        // Iterate over all possible NPUs to find the best one for this user
        for (int s_type_0_idx = 0; s_type_0_idx < num_server_types; ++s_type_0_idx) {
            const auto& server_conf = server_type_configs[s_type_0_idx];
            if (server_conf.max_batch_size <= 0) continue; // Cannot process on this server type

            for (int npu_local_0_idx = 0; npu_local_0_idx < server_conf.g; ++npu_local_0_idx) {
                // Simulate processing all requests for current_user_ref on this NPU
                long long npu_available_time_sim = npu_finish_times_globally[s_type_0_idx][npu_local_0_idx];
                long long user_next_send_allowed_time_sim = current_user_ref.s_time;
                long long samples_left_to_send_sim = current_user_ref.total_samples;
                long long current_user_last_sample_finish_time_sim = 0; 

                while (samples_left_to_send_sim > 0) {
                    long long current_batch_size = std::min(samples_left_to_send_sim, server_conf.max_batch_size);
                    
                    long long req_arrival_at_server_sim = user_next_send_allowed_time_sim + latencies_matrix[s_type_0_idx][current_user_ref.original_id_0_indexed];
                    long long req_processing_start_time_sim = std::max(req_arrival_at_server_sim, npu_available_time_sim);
                    long long inference_duration_sim = calculate_inference_duration_ms(current_batch_size, server_conf.k);
                    long long req_processing_end_time_sim = req_processing_start_time_sim + inference_duration_sim;

                    current_user_last_sample_finish_time_sim = req_processing_end_time_sim; 
                    npu_available_time_sim = req_processing_end_time_sim; 
                    user_next_send_allowed_time_sim = req_arrival_at_server_sim + 1; 
                    samples_left_to_send_sim -= current_batch_size;
                }
                
                if (chosen_server_type_0_idx == -1 || current_user_last_sample_finish_time_sim < best_simulated_overall_finish_time) {
                    best_simulated_overall_finish_time = current_user_last_sample_finish_time_sim;
                    chosen_server_type_0_idx = s_type_0_idx;
                    chosen_npu_0_idx_in_type = npu_local_0_idx;
                }
            }
        }
        
        if (chosen_server_type_0_idx == -1) { // No suitable NPU found
            // This implies the user cannot be scheduled (e.g. all max_batch_size were effectively 0, or other issue)
            // For competition, this might lead to "Samples Not Fully Processed" error for this user.
            // The final_user_schedules for this user will remain empty.
            continue; 
        }

        // Permanently schedule the user on the chosen NPU and update global NPU finish times
        long long samples_to_schedule_final = current_user_ref.total_samples;
        long long user_next_send_allowed_time_final = current_user_ref.s_time;
        
        long long& actual_chosen_npu_finish_time_ref = npu_finish_times_globally[chosen_server_type_0_idx][chosen_npu_0_idx_in_type];
        const auto& chosen_server_conf = server_type_configs[chosen_server_type_0_idx];

        std::vector<RequestDetail> requests_for_this_user;
        while (samples_to_schedule_final > 0) {
            long long current_batch_size_final = std::min(samples_to_schedule_final, chosen_server_conf.max_batch_size);

            long long current_send_time_final = user_next_send_allowed_time_final; 
            long long req_arrival_at_server_final = current_send_time_final + latencies_matrix[chosen_server_type_0_idx][current_user_ref.original_id_0_indexed];
            long long req_processing_start_time_final = std::max(req_arrival_at_server_final, actual_chosen_npu_finish_time_ref);
            long long inference_duration_final = calculate_inference_duration_ms(current_batch_size_final, chosen_server_conf.k);
            long long req_processing_end_time_final = req_processing_start_time_final + inference_duration_final;

            RequestDetail req_det;
            req_det.send_time = current_send_time_final;
            req_det.server_type_id_out = chosen_server_conf.id_out; 
            req_det.npu_id_in_server_type_out = chosen_npu_0_idx_in_type + 1; 
            req_det.batch_size = current_batch_size_final;
            requests_for_this_user.push_back(req_det);

            actual_chosen_npu_finish_time_ref = req_processing_end_time_final; 
            user_next_send_allowed_time_final = req_arrival_at_server_final + 1;
            samples_to_schedule_final -= current_batch_size_final;
        }
        final_user_schedules[current_user_ref.original_id_0_indexed] = requests_for_this_user;
    }

    // Output results in the original user order
    for (int i = 0; i < num_users; ++i) {
        const auto& scheduled_reqs_for_user_i = final_user_schedules[i];
        std::cout << scheduled_reqs_for_user_i.size() << "\n";
        for (size_t j = 0; j < scheduled_reqs_for_user_i.size(); ++j) {
            const auto& req = scheduled_reqs_for_user_i[j];
            std::cout << req.send_time << " "
                      << req.server_type_id_out << " "
                      << req.npu_id_in_server_type_out << " "
                      << req.batch_size;
            if (j < scheduled_reqs_for_user_i.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }

    return 0;
}