import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import matplotlib

# python data_analysis/data_analysis.py
# 仅支持初赛数据

data_file_path = 'data_analysis/data0530.in'

# Set matplotlib parameters for better display
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def parse_data_file(filename):
    """Parse the data file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    
    # Number of servers
    N = int(lines[idx].strip())
    idx += 1
    
    # Server configurations
    servers = []
    for i in range(N):
        g, k, m = map(int, lines[idx].strip().split())
        servers.append({'gpu_count': g, 'speed_coeff': k, 'memory': m})
        idx += 1
    
    # Number of users
    M = int(lines[idx].strip())
    idx += 1
    
    # User requirements
    users = []
    for i in range(M):
        s, e, cnt = map(int, lines[idx].strip().split())
        users.append({'start_time': s, 'end_time': e, 'sample_count': cnt})
        idx += 1
    
    # Communication latency matrix - N rows, each containing latencies to M users
    latency_matrix = []
    for i in range(N):
        # Split the long line of numbers into M latency values
        line_data = list(map(int, lines[idx].strip().split()))
        latency_matrix.append(line_data[:M])  # Take first M values
        idx += 1
    
    # Memory parameters
    a, b = map(int, lines[idx].strip().split())
    memory_params = {'a': a, 'b': b}
    
    # Transpose matrix to make it user x server format
    latency_matrix = np.array(latency_matrix).T.tolist()
    
    return servers, users, latency_matrix, memory_params

def analyze_servers(servers):
    """Analyze server configurations"""
    print("=== Server Configuration Analysis ===")
    
    # Create DataFrame
    server_df = pd.DataFrame(servers)
    server_df['server_id'] = range(1, len(servers) + 1)
    
    print(f"Total servers: {len(servers)}")
    print(f"Total GPUs: {server_df['gpu_count'].sum()}")
    print(f"Average GPU count: {server_df['gpu_count'].mean():.2f}")
    print(f"Average speed coefficient: {server_df['speed_coeff'].mean():.2f}")
    print(f"Average memory size: {server_df['memory'].mean():.0f} MB")
    print()
    
    # Plot server configuration analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Server Configuration Analysis', fontsize=16, fontweight='bold')
    
    # GPU count distribution
    axes[0, 0].bar(server_df['server_id'], server_df['gpu_count'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('GPU Count per Server')
    axes[0, 0].set_xlabel('Server ID')
    axes[0, 0].set_ylabel('GPU Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Speed coefficient distribution
    axes[0, 1].bar(server_df['server_id'], server_df['speed_coeff'], color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Speed Coefficient per Server')
    axes[0, 1].set_xlabel('Server ID')
    axes[0, 1].set_ylabel('Speed Coefficient')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory size distribution
    axes[1, 0].bar(server_df['server_id'], server_df['memory'], color='orange', alpha=0.7)
    axes[1, 0].set_title('Memory Size per Server')
    axes[1, 0].set_xlabel('Server ID')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Server performance evaluation
    server_df['total_memory'] = server_df['gpu_count'] * server_df['memory']
    server_df['performance_score'] = server_df['gpu_count'] * server_df['speed_coeff']
    
    scatter = axes[1, 1].scatter(server_df['performance_score'], server_df['total_memory'], 
                      s=100, alpha=0.7, c=server_df['server_id'], cmap='viridis')
    axes[1, 1].set_title('Server Performance vs Total Memory')
    axes[1, 1].set_xlabel('Performance Score (GPU Count × Speed Coeff)')
    axes[1, 1].set_ylabel('Total Memory (MB)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Server ID')
    
    plt.tight_layout()
    plt.savefig('server_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return server_df

def analyze_users(users):
    """Analyze user requirements"""
    print("=== User Requirements Analysis ===")
    
    # Create DataFrame
    user_df = pd.DataFrame(users)
    user_df['user_id'] = range(1, len(users) + 1)
    user_df['duration'] = (user_df['end_time'] - user_df['start_time'])
    user_df['samples_per_ms'] = user_df['sample_count'] / user_df['duration']
    
    print(f"Total users: {len(users)}")
    print(f"Total samples: {user_df['sample_count'].sum()}")
    print(f"Average samples: {user_df['sample_count'].mean():.0f}")
    print(f"Max time window: {user_df['duration'].max()} ms")
    print(f"Average time window: {user_df['duration'].mean():.0f} ms")
    print(f"Max sample density: {user_df['samples_per_ms'].max():.4f} samples/ms")
    print()
    
    # Plot user requirement analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('User Requirements Analysis', fontsize=16, fontweight='bold')
    
    # Sample count distribution
    axes[0, 0].hist(user_df['sample_count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Sample Count Distribution')
    axes[0, 0].set_xlabel('Sample Count')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time window distribution
    axes[0, 1].hist(user_df['duration'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Time Window Distribution')
    axes[0, 1].set_xlabel('Duration (ms)')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample density distribution
    axes[0, 2].hist(user_df['samples_per_ms'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_title('Sample Density Distribution')
    axes[0, 2].set_xlabel('Sample Density (samples/ms)')
    axes[0, 2].set_ylabel('Number of Users')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Start time distribution
    axes[1, 0].hist(user_df['start_time'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_title('Task Start Time Distribution')
    axes[1, 0].set_xlabel('Start Time (ms)')
    axes[1, 0].set_ylabel('Number of Users')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Timeline view (sample subset for visibility)
    subset_size = min(50, len(user_df))
    user_subset = user_df.head(subset_size)
    for i, user in user_subset.iterrows():
        axes[1, 1].barh(i, user['duration'], left=user['start_time'], 
                       alpha=0.6, height=0.8)
    axes[1, 1].set_title(f'User Task Timeline (First {subset_size} users)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('User ID')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Sample count vs time window scatter plot
    scatter = axes[1, 2].scatter(user_df['duration'], user_df['sample_count'], 
                                alpha=0.7, c=user_df['samples_per_ms'], cmap='viridis', s=50)
    axes[1, 2].set_title('Sample Count vs Time Window')
    axes[1, 2].set_xlabel('Time Window (ms)')
    axes[1, 2].set_ylabel('Sample Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('Sample Density (samples/ms)')
    
    plt.tight_layout()
    plt.savefig('user_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return user_df

def analyze_latency(latency_matrix, num_servers):
    """Analyze communication latency"""
    print("=== Communication Latency Analysis ===")
    
    latency_array = np.array(latency_matrix)
    
    print(f"Latency matrix shape: {latency_array.shape}")
    print(f"Min latency: {latency_array.min()} ms")
    print(f"Max latency: {latency_array.max()} ms")
    print(f"Average latency: {latency_array.mean():.2f} ms")
    print(f"Latency std dev: {latency_array.std():.2f} ms")
    print()
    
    # Plot latency analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Communication Latency Analysis', fontsize=16, fontweight='bold')
    
    # Latency heatmap
    im = axes[0, 0].imshow(latency_array[:100, :], cmap='viridis', aspect='auto')  # Show first 100 users
    axes[0, 0].set_title('User-Server Latency Heatmap (First 100 Users)')
    axes[0, 0].set_xlabel('Server ID')
    axes[0, 0].set_ylabel('User ID')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 0])
    cbar.set_label('Latency (ms)')
    
    # Latency distribution histogram
    axes[0, 1].hist(latency_array.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Latency Distribution')
    axes[0, 1].set_xlabel('Latency (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average latency per server
    server_avg_latency = latency_array.mean(axis=0)
    axes[1, 0].bar(range(1, num_servers + 1), server_avg_latency, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Average Latency per Server')
    axes[1, 0].set_xlabel('Server ID')
    axes[1, 0].set_ylabel('Average Latency (ms)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average latency per user distribution
    user_avg_latency = latency_array.mean(axis=1)
    axes[1, 1].hist(user_avg_latency, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('User Average Latency Distribution')
    axes[1, 1].set_xlabel('Average Latency (ms)')
    axes[1, 1].set_ylabel('Number of Users')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return latency_array

def analyze_memory_requirements(users, memory_params, servers):
    """Analyze memory requirements"""
    print("=== Memory Requirements Analysis ===")
    
    a, b = memory_params['a'], memory_params['b']
    print(f"Memory formula: Memory = {a} × batchsize + {b}")
    
    # Calculate memory requirements for different batch sizes
    batch_sizes = np.arange(1, 1001)
    memory_requirements = a * batch_sizes + b
    
    # Analyze users' maximum possible batch size
    max_possible_batches = []
    for user in users:
        max_batch = user['sample_count']
        max_possible_batches.append(max_batch)
    
    print(f"Max batch size: {max(max_possible_batches)}")
    print(f"Corresponding memory requirement: {a * max(max_possible_batches) + b} MB")
    print()
    
    # Plot memory analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Memory Requirements Analysis', fontsize=16, fontweight='bold')
    
    # Memory requirement curve
    axes[0, 0].plot(batch_sizes, memory_requirements, 'b-', linewidth=2)
    axes[0, 0].set_title('Batch Size vs Memory Requirement')
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Memory Requirement (MB)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # User max batch size distribution
    axes[0, 1].hist(max_possible_batches, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('User Max Batch Size Distribution')
    axes[0, 1].set_xlabel('Max Batch Size')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Server batch size capacity
    server_memories = [server['memory'] for server in servers]
    server_max_batches = [(m - b) // a for m in server_memories]
    server_ids = [f'Server {i+1}' for i in range(len(servers))]
    
    axes[1, 0].bar(server_ids, server_max_batches, alpha=0.7, color='orange')
    axes[1, 0].set_title('Max Batch Size Capacity per Server')
    axes[1, 0].set_ylabel('Max Batch Size')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Inference time estimation
    k_values = [server['speed_coeff'] for server in servers[:3]]  # Use first 3 servers
    sample_batch_sizes = [10, 50, 100, 200, 500]
    
    for i, k in enumerate(k_values):
        inference_times = []
        for batch_size in sample_batch_sizes:
            time = np.ceil(batch_size / (k * np.sqrt(batch_size)))
            inference_times.append(time)
        
        axes[1, 1].plot(sample_batch_sizes, inference_times, 'o-', 
                       label=f'Server Type {i+1} (k={k})', linewidth=2, markersize=6)
    
    axes[1, 1].set_title('Inference Time vs Batch Size')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Inference Time (ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_workload_distribution(users):
    """Analyze workload distribution"""
    print("=== Workload Distribution Analysis ===")
    
    # Create time axis
    max_time = max(user['end_time'] for user in users)
    time_axis = np.arange(0, max_time + 1000, 1000)  # Every 1000ms
    
    # Calculate active users and total sample demand at each time point
    active_users = []
    total_samples = []
    
    for t in time_axis:
        active_count = 0
        sample_demand = 0
        
        for user in users:
            if user['start_time'] <= t < user['end_time']:
                active_count += 1
                # Assume samples are uniformly distributed within time window
                remaining_time = user['end_time'] - t
                window_duration = user['end_time'] - user['start_time']
                remaining_samples = user['sample_count'] * (remaining_time / window_duration)
                sample_demand += remaining_samples
        
        active_users.append(active_count)
        total_samples.append(sample_demand)
    
    # Plot workload analysis
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('System Workload Over Time', fontsize=16, fontweight='bold')
    
    # Active users over time
    axes[0].plot(time_axis / 1000, active_users, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Active Users Over Time')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Number of Active Users')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(time_axis / 1000, active_users, alpha=0.3)
    
    # Sample demand over time
    axes[1].plot(time_axis / 1000, total_samples, 'r-', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Total Sample Demand Over Time')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Remaining Sample Demand')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(time_axis / 1000, total_samples, alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.savefig('workload_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Peak active users: {max(active_users)}")
    print(f"Peak sample demand: {max(total_samples):.0f}")
    print()

def create_summary_report(servers, users, latency_array, memory_params):
    """Create a comprehensive summary report"""
    print("=== Comprehensive Summary Report ===")
    
    # Basic statistics
    total_gpus = sum(server['gpu_count'] for server in servers)
    total_samples = sum(user['sample_count'] for user in users)
    avg_latency = latency_array.mean()
    
    # Time analysis
    user_df = pd.DataFrame(users)
    total_duration = user_df['end_time'].max()
    avg_user_duration = (user_df['end_time'] - user_df['start_time']).mean()
    
    # Memory analysis
    a, b = memory_params['a'], memory_params['b']
    max_batch_size = max(user['sample_count'] for user in users)
    max_memory_needed = a * max_batch_size + b
    max_server_memory = max(server['memory'] for server in servers)
    
    print(f"System Overview:")
    print(f"- Total Servers: {len(servers)}")
    print(f"- Total GPUs: {total_gpus}")
    print(f"- Total Users: {len(users)}")
    print(f"- Total Samples to Process: {total_samples:,}")
    print(f"- Simulation Duration: {total_duration:,} ms ({total_duration/1000:.1f} seconds)")
    print()
    
    print(f"Resource Utilization:")
    print(f"- Average GPU Load: {total_samples / total_gpus:.0f} samples/GPU")
    print(f"- Average Communication Latency: {avg_latency:.2f} ms")
    print(f"- Average User Task Duration: {avg_user_duration:.0f} ms")
    print()
    
    print(f"Memory Analysis:")
    print(f"- Memory Formula: {a} × batch_size + {b}")
    print(f"- Largest Single Batch: {max_batch_size:,} samples")
    print(f"- Memory for Largest Batch: {max_memory_needed:,} MB")
    print(f"- Maximum Server Memory: {max_server_memory:,} MB")
    print(f"- Memory Utilization: {max_memory_needed/max_server_memory*100:.1f}%")
    print()
    
    # Performance insights
    print(f"Performance Insights:")
    server_df = pd.DataFrame(servers)
    best_server = server_df.loc[server_df['speed_coeff'].idxmax()]
    worst_server = server_df.loc[server_df['speed_coeff'].idxmin()]
    
    print(f"- Fastest Server: Server {best_server.name + 1} (speed_coeff={best_server['speed_coeff']}, {best_server['gpu_count']} GPUs)")
    print(f"- Slowest Server: Server {worst_server.name + 1} (speed_coeff={worst_server['speed_coeff']}, {worst_server['gpu_count']} GPUs)")
    
    # Latency insights
    latency_df = pd.DataFrame(latency_array)
    min_latency_server = latency_df.mean().idxmin()
    max_latency_server = latency_df.mean().idxmax()
    
    print(f"- Lowest Latency Server: Server {min_latency_server + 1} (avg: {latency_df.mean().iloc[min_latency_server]:.1f} ms)")
    print(f"- Highest Latency Server: Server {max_latency_server + 1} (avg: {latency_df.mean().iloc[max_latency_server]:.1f} ms)")
    
    print("\nAnalysis complete! Charts have been saved as PNG files.")

def main():
    """Main function"""
    print("Starting Edge Cluster AI Inference Task Scheduling Data Analysis...")
    print("=" * 70)
    
    # Parse data
    servers, users, latency_matrix, memory_params = parse_data_file(data_file_path)
    
    # Basic information
    print(f"Data Overview:")
    print(f"- Number of Server Types: {len(servers)}")
    print(f"- Number of Users: {len(users)}")
    print(f"- Memory Parameters: a={memory_params['a']}, b={memory_params['b']}")
    print("=" * 70)
    
    # Perform various analyses
    server_df = analyze_servers(servers)
    user_df = analyze_users(users)
    latency_array = analyze_latency(latency_matrix, len(servers))
    analyze_memory_requirements(users, memory_params, servers)
    analyze_workload_distribution(users)
    
    # Create comprehensive summary
    create_summary_report(servers, users, latency_array, memory_params)
    
    # Return results for further processing if needed
    results = {
        'servers': server_df,
        'users': user_df,
        'latency_stats': {
            'min': latency_array.min(),
            'max': latency_array.max(),
            'mean': latency_array.mean(),
            'std': latency_array.std()
        },
        'summary': {
            'total_gpus': server_df['gpu_count'].sum(),
            'total_samples': user_df['sample_count'].sum(),
            'avg_latency': latency_array.mean(),
            'load_per_gpu': user_df['sample_count'].sum() / server_df['gpu_count'].sum()
        }
    }
    
    return results

if __name__ == "__main__":
    results = main() 