import os
from math import sqrt
import numpy as np
from sklearn.cluster import DBSCAN
from SigProfilerSimulator import SigProfilerSimulator as sigSim
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import pretreatment

# 使用DBSCAN聚类算法计算簇数量
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

# 使用SigProfilerSimulator生成模拟背景模型
def generate_simulated_mutations(input_file,file_name, input_path, genome):
    """
        将 TSV 文件转换为 TXT 文件
        :param input_file: 原始 TSV 文件路径
        :param output_txt: 输出 TXT 文件路径
        """
    # 读取 TSV 文件
    df = pd.read_csv(input_file, sep="\t")

    """
        调整列顺序并调用 SigProfilerSimulator
        :param df: 输入的 DataFrame
        :param output_txt: 调整后的输出文件路径
        :param genome: 基因组版本
        """
    # 定义目标列顺序
    target_columns = [
        "CANCER_TYPE", "Sample", "Source", "GENOME_VERSION", "MutationType",
        "Chromosome", "Position", "Position", "Reference", "Alternate", "MutationClass"
    ]

    # 添加或填充缺失列
    if "Source" not in df.columns:
        df["Source"] = ""  # 填充为空值
    if "MutationType" not in df.columns:
        df["MutationType"] = "SNV"  # 填充为固定值
    if "MutationClass" not in df.columns:
        df["MutationClass"] = ""  # 填充为空值

    # 检查缺失的目标列
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    # 调整列顺序
    reordered_df = df[target_columns]

    # 保存为调整后的 TXT 文件
    output_txt = os.path.join(input_path, f"{file_name}_txt.txt")  # 添加扩展名 .txt
    reordered_df.to_csv(output_txt, sep="\t", index=False)
    print(f"File with reordered columns saved to: {output_txt}")
    file_name_txt = f"{file_name}_txt.txt"
    sigSim.SigProfilerSimulator(file_name_txt,  input_path, genome, contexts=["96"], simulations=100,exome=True)

    print(f"模拟数据已生成")

# 读取模拟数据并计算簇数量
def get_simulated_clusters(output_directory, eps):
    simulated_files = [
        os.path.join(output_directory, f"simulation_{i}.txt")
        for i in range(1, 1001)
    ]
    simulated_cluster_counts = []
    all_imd_values = []
    for sim_file in simulated_files:
        positions = []
        with open(sim_file, "r") as f:
            for line in f:
                if line.startswith("Chromosome"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 3 or not fields[2].isdigit():
                    raise ValueError(f"文件 {sim_file} 格式不正确，或列数不足！")
                positions.append(int(fields[2]))
        positions.sort()
        imd_values = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        all_imd_values.extend(imd_values)
        clusters = get_clusters_with_dbscan(positions, eps=eps, min_samples=2)
        simulated_cluster_counts.append(clusters)
    mean_simulated_clusters = np.mean(simulated_cluster_counts)
    return mean_simulated_clusters, all_imd_values

# 计算突变负担
def calculate_mutation_burden(num_mutations, genome_size_mb):
    return num_mutations / genome_size_mb

# 计算FDR（假阳性率）
def calculate_fdr(actual_clusters, mean_simulated_clusters):
    return mean_simulated_clusters / actual_clusters if actual_clusters > 0 else 1.0

# 绘制IMD分布图
def plot_imd_distributions(real_imd_values, simulated_imd_values):
    real_imd_values = [v for v in real_imd_values if v > 0]
    simulated_imd_values = [v for v in simulated_imd_values if v > 0]
    real_kde = gaussian_kde(np.log10(real_imd_values))
    simulated_kde = gaussian_kde(np.log10(simulated_imd_values))
    x = np.linspace(0, 6, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, real_kde(x), label="真实样本", color="green")
    plt.plot(x, simulated_kde(x), label="模拟样本", color="red")
    plt.fill_between(x, simulated_kde(x), alpha=0.3, color="red")
    plt.axvline(x=np.log10(100), linestyle="--", color="black", label="推荐IMD值")
    plt.xlabel("log10(IMD)")
    plt.ylabel("密度")
    plt.legend()
    plt.title("IMD分布对比")
    plt.show()

# 动态计算推荐的IMD阈值
def calculate_dynamic_eps(real_imd_values, simulated_imd_values, quantile=0.9):
    real_threshold = np.quantile(real_imd_values, quantile)
    simulated_threshold = np.quantile(simulated_imd_values, quantile)
    recommended_eps = (real_threshold + simulated_threshold) / 2
    return recommended_eps

def main():
    genome_size_bp = 3095693983  # GRCh37 基因组总大小
    genome_size_mb = genome_size_bp / 1e6
    output_directory = "E:/jjm/work/簇突变/Pyth_fdr/simulations_output" #模拟背景模型输出文件
    input_path = "E:/jjm/work/簇突变/Pyth_fdr/input_file/example/"
    file_name = "Cosmic_GenomeScreensMutant_Tsv_v99_GRCh37_PD4103a"
    input_file = os.path.join(input_path, f"{file_name}.tsv")  # 添加扩展名 .tsv
    outfile_name = "E:/jjm/work/簇突变/Pyth_fdr/output_file/Tsv_v99_GRCh37_PD4103a_BRCA.tsv" #原始文件处理后的名字
    cancer_type = "BRCA"  #数据的癌种

    # 检查路径
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"输入文件 {input_file} 不存在！")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

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
        # 如果文件存在，直接加载为 DataFrame
        df = pd.read_csv(outfile_name, sep="\t")
        print(f"Loaded existing file with {len(df)} records")
    else:
        #不存在，处理原始文件
        df = pretreatment.process_mutation_data(
                input_file=input_file,
                df=df,
                outfile_name=outfile_name,
                cancer_type=cancer_type,
            )


    mutation_data = {}
    for _, row in df.iterrows():
        chrom = row['Chromosome']
        pos = row['Position']
        if chrom not in mutation_data:
            mutation_data[chrom] = []
        mutation_data[chrom].append(pos)

    try:
        generate_simulated_mutations(file_name, input_path, genome="GRCh37") #生成模拟背景模型

        real_imd_values = []
        for positions in mutation_data.values():
            positions.sort()
            real_imd_values.extend([positions[i + 1] - positions[i] for i in range(len(positions) - 1)])

        mean_simulated_clusters, simulated_imd_values = get_simulated_clusters(output_directory, eps=100)
        recommended_eps = calculate_dynamic_eps(real_imd_values, simulated_imd_values, quantile=0.9)

        total_clusters = sum(
            get_clusters_with_dbscan(positions, eps=recommended_eps, min_samples=2)
            for positions in mutation_data.values()
        )

        fdr = calculate_fdr(total_clusters, mean_simulated_clusters)
        mutation_burden = calculate_mutation_burden(all_mutations, genome_size_mb)

        print(f"推荐的IMD阈值 (eps): {recommended_eps:.2f}")
        print(f"突变负担 (Mut/Mb): {mutation_burden}")
        print(f"真实数据簇数量: {total_clusters}")
        print(f"模拟数据簇数量均值: {mean_simulated_clusters}")
        print(f"假阳性率 (FDR): {fdr}")

        plot_imd_distributions(real_imd_values, simulated_imd_values)

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
