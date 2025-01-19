import os
from math import sqrt
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 使用聚类算法计算簇数量
def get_clusters_with_dbscan(positions, eps, min_samples=2):
    if len(positions) == 0:
        return 0
    coordinates = calculate_imd_coordinates(positions)  # 计算IMD二维坐标
    clustering = DBSCAN(eps=eps * sqrt(2), min_samples=min_samples).fit(coordinates)
    num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)  # 排除噪声点
    return num_clusters

# 计算突变点的二维坐标
def calculate_imd_coordinates(positions):
    positions.sort()
    imd_values = [0] + [positions[i] - positions[i - 1] for i in range(1, len(positions))]  # 计算IMD值
    coordinates = np.array([[positions[i], imd_values[i]] for i in range(len(positions))])
    return coordinates

# GRCh37 和 GRCh38 基因组染色体大小（单位：bp）
chromosome_sizes_dict = {
    'GRCh37': {
        'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260,
        'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chr10': 135534747,
        'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392,
        'chr16': 90354753,  'chr17': 81195210,  'chr18': 78077248,  'chr19': 59128983,  'chr20': 63025520,
        'chr21': 48129895,  'chr22': 51304566,  'chrX': 155270560, 'chrY': 59373566
    },
    'GRCh38': {
        'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555, 'chr5': 181538259,
        'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717, 'chr10': 133797422,
        'chr11': 135086622, 'chr12': 133275309, 'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
        'chr16': 90338345,  'chr17': 83257441,  'chr18': 80373285,  'chr19': 58617616,  'chr20': 64444167,
        'chr21': 46709983,  'chr22': 50818468,  'chrX': 156040895, 'chrY': 57227415
    }
}

def generate_simulated_mutations(outfile_name, output_directory, genome):
    # 检查基因组版本
    if genome not in chromosome_sizes_dict:
        raise ValueError(f"未知的基因组版本：{genome}。支持的版本为：{list(chromosome_sizes_dict.keys())}")

    # 获取指定基因组的染色体大小
    chromosome_sizes = chromosome_sizes_dict[genome]

    # 读取真实数据文件
    df = pd.read_csv(outfile_name, sep="\t")

    # 统计真实数据的特征
    num_simulations = 1000  # 模拟数据集数量
    chrom_proportions = df['Chromosome'].value_counts(normalize=True)  # 按染色体比例分布

    # 计算IMD分布的核密度估计
    all_imd_values = (
        df.groupby('Chromosome')['Position']
        .apply(lambda x: x.sort_values().diff().dropna())
        .explode()
        .dropna()
        .astype(float)
        .values
    )
    kde = gaussian_kde(all_imd_values)  # 使用核密度估计建模IMD分布

    # 确定热点区域（根据突变密度）
    hotspot_regions = {}
    for chrom in df['Chromosome'].unique():
        chrom_positions = df[df['Chromosome'] == chrom]['Position']
        if len(chrom_positions) > 1:
            hotspot_regions[chrom] = chrom_positions.sample(frac=0.1, random_state=42).values  # 假设10%为热点

    # 生成模拟数据
    for i in range(1, num_simulations + 1):
        simulation_file = os.path.join(output_directory, f"simulation_{i}.txt")
        with open(simulation_file, "w") as f:
            f.write("Chromosome\tPosition\n")

            for chrom, proportion in chrom_proportions.items():
                # 获取染色体的大小
                max_chrom_size = chromosome_sizes.get(str(chrom))
                if not max_chrom_size:
                    print(f"染色体 {chrom} 的大小信息缺失，跳过此染色体！")
                    continue

                # 模拟每条染色体的突变数量
                num_positions = int(len(df) * proportion)
                if num_positions <= 0:
                    continue

                # 从IMD分布的KDE生成突变间距
                sampled_imds = kde.resample(size=num_positions - 1)[0]
                sampled_imds = sampled_imds[sampled_imds > 0]  # 保证IMD为正数

                # 从热点区域中抽样起始点
                if chrom in hotspot_regions:
                    start_position = np.random.choice(hotspot_regions[chrom])
                else:
                    start_position = np.random.randint(1, max_chrom_size)

                # 累加生成突变位置
                positions = np.cumsum(np.insert(sampled_imds, 0, start_position))
                positions = positions[positions < max_chrom_size]  # 限制突变位置在染色体范围内

                for pos in positions:
                    f.write(f"{chrom}\t{int(pos)}\n")

# 读取模拟数据并计算簇数量
def get_simulated_clusters(output_directory, eps):
    simulated_files = [
        os.path.join(output_directory, f"simulation_{i}.txt")
        for i in range(1, 1001)
    ]
    simulated_cluster_counts = []
    all_imd_values = []

    for sim_file in simulated_files:
        try:
            # 读取模拟文件
            df = pd.read_csv(sim_file, sep="\t")

            # 检查必须的列名是否存在
            if 'Position' not in df.columns:
                raise ValueError(f"文件 {sim_file} 缺少 'Position' 列！")

            # 提取并排序突变位置信息
            positions = df['Position'].dropna().astype(int).sort_values().tolist()

            # 计算IMD值
            imd_values = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            all_imd_values.extend(imd_values)

            # 计算簇数量
            clusters = get_clusters_with_dbscan(positions, eps=eps, min_samples=2)
            simulated_cluster_counts.append(clusters)

        except Exception as e:
            print(f"处理文件 {sim_file} 时发生错误: {e}")
            continue

    mean_simulated_clusters = np.mean(simulated_cluster_counts)
    return mean_simulated_clusters, all_imd_values

# 绘制IMD分布对比并计算显著性阈值
def plot_imd_distribution_and_calculate_threshold(real_imd_values, simulated_imd_values, bins=50):
    # 过滤掉小于或等于零的IMD值，转换为对数尺度
    real_log_imd = np.log10([value for value in real_imd_values if value > 0])
    simulated_log_imd = np.log10([value for value in simulated_imd_values if value > 0])

    # 计算直方图及密度分布
    real_hist, real_bins = np.histogram(real_log_imd, bins=bins, density=True)
    simulated_hist, simulated_bins = np.histogram(simulated_log_imd, bins=bins, density=True)

    # 累积分布函数（CDF）计算
    real_cdf = np.cumsum(real_hist) / np.sum(real_hist)
    simulated_cdf = np.cumsum(simulated_hist) / np.sum(simulated_hist)

    # 计算CDF差异并确定阈值
    cdf_diff = real_cdf - simulated_cdf
    max_diff_index = np.argmax(cdf_diff)
    threshold_log10 = real_bins[max_diff_index]  # 找到CDF差异最大的点（对数值）
    threshold_value = 10 ** threshold_log10  # 转换为原始IMD值

    # 绘制IMD分布图
    plt.figure(figsize=(10, 6))
    plt.plot(real_bins[:-1], real_hist, label='Real Data IMD Distribution (log10)', marker='o', linestyle='-', alpha=0.7)
    plt.plot(simulated_bins[:-1], simulated_hist, label='Simulated Data IMD Distribution (log10)', marker='s', linestyle='--', alpha=0.7)
    plt.axvline(threshold_log10, color='red', linestyle='--', label=f'Threshold (log10): {threshold_log10:.2f}')
    plt.xlabel('log10(IMD Values)')
    plt.ylabel('Density')
    plt.title('Comparison of Real and Simulated Data IMD Distributions with Threshold')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

    # 输出阈值结果
    print(f"Threshold IMD (log10): {threshold_log10:.2f}, Threshold IMD: {threshold_value:.2f}")
    return threshold_value

# 主函数中添加显著性阈值的计算和分布绘图
def main():
    genome_size_bp = 3095693983  # GRCh37 基因组总大小
    genome_size_mb = genome_size_bp / 1e6
    output_directory = "./simulations_output"  # 模拟背景模型输出文件
    input_file = "./input_file/example/Cosmic_GenomeScreensMutant_Tsv_v99_GRCh37_PD4103a.tsv"
    outfile_name = "./output_file/Tsv_v99_GRCh37_PD4103a_BRCA.tsv"
    genome = "GRCh37"
    cancer_type = "BRCA"

    # 检查路径
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"输入文件 {input_file} 不存在！")

    # 读取原始输入文件
    df = pd.read_csv(input_file, sep="\t")
    print(f"Total records in input file: {len(df)}")

    # 按 CHROMOSOME 和 GENOME_START 去重
    df = df.drop_duplicates(subset=['CHROMOSOME', 'GENOME_START'])
    all_mutations = len(df)
    print(f"Records after deduplication: {len(df)}")

    # 检查输出文件是否已经存在
    if os.path.exists(outfile_name):
        print(f"Output file already exists: {outfile_name}")
        df = pd.read_csv(outfile_name, sep="\t")
        print(f"Loaded existing file with {len(df)} records")
    else:
        # pretreatment 模块以处理原始数据
        from pretreatment import process_mutation_data
        df = process_mutation_data(
            input_file=input_file,
            df=df,
            outfile_name=outfile_name,
            cancer_type=cancer_type,
            batch_size=40,
            species="human"
        )

    # 提取真实数据的 IMD 值
    mutation_data = {}
    real_imd_values = []
    for _, row in df.iterrows():
        chrom = row['Chromosome']
        pos = row['Position']
        if chrom not in mutation_data:
            mutation_data[chrom] = []
        mutation_data[chrom].append(pos)
    for positions in mutation_data.values():
        positions.sort()
        real_imd_values.extend([positions[i + 1] - positions[i] for i in range(len(positions) - 1)])

    # 生成模拟背景模型
    generate_simulated_mutations(outfile_name, output_directory, genome)

    # 计算模拟数据的簇数量和IMD分布
    mean_simulated_clusters, simulated_imd_values = get_simulated_clusters(output_directory, eps=1000)

    # 绘制IMD分布对比并计算显著性阈值
    threshold_value = plot_imd_distribution_and_calculate_threshold(real_imd_values, simulated_imd_values)

    # 使用阈值计算真实数据簇数量
    total_clusters = sum(
        get_clusters_with_dbscan(positions, eps=threshold_value, min_samples=2)
        for positions in mutation_data.values()
    )

    # 计算假阳性率（FDR）
    fdr = mean_simulated_clusters / total_clusters if total_clusters > 0 else 1.0

    # 输出结果
    mutation_burden = all_mutations / genome_size_mb
    print(f"Mutation Burden (Mut/Mb): {mutation_burden:.6f}")
    print(f"Real Data Cluster Count: {total_clusters}")
    print(f"Mean Simulated Cluster Count: {mean_simulated_clusters:.6f}")
    print(f"False Discovery Rate (FDR): {fdr:.6f}")

if __name__ == "__main__":
    main()



