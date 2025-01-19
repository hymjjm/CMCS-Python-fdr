import os
import re

import pandas as pd
import requests
from datetime import datetime


def fetch_sequence_from_ucsc(chromosome, start, end, build):
    """
    从 UCSC Genome Browser 远程获取基因组序列
    """
    url = f"http://genome.ucsc.edu/cgi-bin/das/{build}/dna?segment={chromosome}:{start},{end}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            sequence = ''.join(response.text.split("<DNA length=")[-1].split("</DNA>")[0].strip().split())
            return sequence.upper()
        except IndexError:
            return "N/A"
    else:
        raise Exception(f"Failed to fetch sequence from UCSC: {response.status_code}, {response.text}")


def map_genome_version(version):
    """
    将基因组版本映射到 UCSC 所需的版本
    """
    mapping = {
        "37": "hg19",
        "38": "hg38"
    }
    return mapping.get(str(version), "unknown")


def process_mutation_data(input_file, df,outfile_name, cancer_type):
    """
    处理突变数据，提取单碱基替换（SBS）并通过远程加载获取上下文信息
    """
    print(f"Loading input file: {input_file}")

    # 筛选单碱基替换（SBS）
    df = df[(df['GENOMIC_WT_ALLELE'].str.len() == 1) & (df['GENOMIC_MUT_ALLELE'].str.len() == 1)]
    print(f"Records after SBS filtering: {len(df)}")

    # 添加手动指定的 CANCER_TYPE 列
    if 'CANCER_TYPE' not in df.columns:
        print(f"CANCER_TYPE column not found. Adding default value: {cancer_type}")
        df['CANCER_TYPE'] = cancer_type

    # 添加或确认 'Sample' 列
    if 'Sample' not in df.columns:
        print("Adding 'Sample' column with default values.")
        df['Sample'] = os.path.splitext(os.path.basename(input_file))[0]

    # 锁定 DataFrame 避免迭代中被修改
    locked_df = df.copy()

    # 提取上下文信息
    contexts = []
    processed_count = 0

    for index, row in locked_df.iterrows():
        processed_count += 1
        chrom = f"chr{row['CHROMOSOME']}"  # 为染色体添加前缀
        pos = row['GENOME_START']
        wt_allele = row['GENOMIC_WT_ALLELE']
        mut_allele = row['GENOMIC_MUT_ALLELE']
        genome_version = row['GENOME_VERSION']

        # 映射到 UCSC 所需的基因组版本
        ucsc_build = map_genome_version(genome_version)
        if ucsc_build == "unknown":
            print(f"Unknown genome version: {genome_version}")
            contexts.append("N/A")
            continue

        print(f"Processing record {processed_count}/{len(locked_df)}: {chrom}:{pos} ({ucsc_build})")

        # 获取上下文序列
        try:
            upstream = fetch_sequence_from_ucsc(chromosome=chrom, start=pos - 1, end=pos - 1, build=ucsc_build)
            downstream = fetch_sequence_from_ucsc(chromosome=chrom, start=pos + 1, end=pos + 1, build=ucsc_build)
            if upstream != "N/A" and downstream != "N/A":
                context = f"{upstream}[{wt_allele}>{mut_allele}]{downstream}"
            else:
                context = "N/A"
        except Exception as e:
            context = "N/A"
            print(f"Error fetching context for {chrom}:{pos} - {e}")

        contexts.append(context)

    # 添加上下文信息到 DataFrame
    locked_df['Context'] = contexts

    # 更新 CHROMOSOME 列，添加 chr 前缀
    locked_df['CHROMOSOME'] = locked_df['CHROMOSOME'].apply(lambda x: f"chr{x}")

    # 更新 GENOME_VERSION 列，添加 GRCh 前缀
    locked_df['GENOME_VERSION'] = locked_df['GENOME_VERSION'].apply(
        lambda x: f"GRCh{x}" if not str(x).startswith("GRCh") else x)

    # 清理 Context 列
    locked_df['Context'] = locked_df['Context'].str.replace(r'[^ACGT\[\]>]', '', regex=True)
    # 删除第一个和倒数第二个字符
    locked_df['Context'] = locked_df['Context'].apply(lambda x: x[1:-1] if len(x) > 2 else x)

    # 按指定顺序选择和重命名列
    columns_to_keep = [
        'GENE_SYMBOL', 'CHROMOSOME', 'GENOME_START', 'GENOMIC_WT_ALLELE',
        'GENOMIC_MUT_ALLELE', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'GENOME_VERSION', 'Context'
    ]
    columns_to_rename = [
        'GENE_SYMBOL', 'Chromosome', 'Position', 'Reference',
        'Alternate', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'GENOME_VERSION', 'Context'
    ]

    # 处理缺少的列
    for col in columns_to_keep:
        if col not in locked_df.columns:
            print(f"Missing column: {col}, filling with 'N/A'")
            locked_df[col] = "N/A"

    output_df = locked_df[columns_to_keep]
    output_df.columns = columns_to_rename

    # 保存结果
    output_df.to_csv(outfile_name, sep="\t", index=False)
    print(f"Processing completed. Results saved to: {outfile_name}")

    return output_df


# # 示例调用
# process_mutation_data(
#     input_file="E:/jjm/work/簇突变/Pyth_fdr/input_file/example/Cosmic_GenomeScreensMutant_Tsv_v99_GRCh37_PD4103a.tsv",
#     df=df,
#     outfile_name ="E:/jjm/work/簇突变/Pyth_fdr/output_file/Tsv_v99_GRCh37_PD4103a_BRCA.tsv", #原始文件处理后的名字
#     cancer_type="BRCA",
# )