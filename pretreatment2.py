import os
import re
import pandas as pd
from Bio import SeqIO
from datetime import datetime


def load_genome(fasta_file):
    """
    从本地基因组文件加载序列
    :param fasta_file: 基因组的FASTA文件路径
    :return: 一个字典 {chromosome: sequence}
    """
    genome = {}
    print(f"Loading genome from: {fasta_file}")
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome[record.id] = str(record.seq).upper()
    print("Genome loaded successfully.")
    return genome


def fetch_sequence_local(genome, chromosome, start, end):
    """
    从本地加载的基因组中提取序列
    :param genome: 加载的基因组字典
    :param chromosome: 染色体名称（如 'chr1'）
    :param start: 起始位置
    :param end: 结束位置
    :return: 提取的序列
    """
    try:
        seq = genome[chromosome][start - 1:end]  # 基因组序列是1-based，而Python是0-based
        return seq
    except KeyError:
        return "N/A"
    except IndexError:
        return "N/A"


def process_mutation_data_local(input_file, fasta_file, outfile_name, cancer_type="Unknown"):
    """
    处理突变数据，提取单碱基替换（SBS）并通过本地基因组加载获取上下文信息
    """
    print(f"Loading input file: {input_file}")

    # 读取输入文件
    df = pd.read_csv(input_file, sep="\t")
    print(f"Total records in input file: {len(df)}")

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

    # 加载基因组文件
    genome = load_genome(fasta_file)

    # 提取上下文信息
    contexts = []
    processed_count = 0

    for index, row in locked_df.iterrows():
        processed_count += 1
        chrom = f"chr{row['CHROMOSOME']}"  # 添加 chr 前缀
        pos = row['GENOME_START']
        wt_allele = row['GENOMIC_WT_ALLELE']
        mut_allele = row['GENOMIC_MUT_ALLELE']

        print(f"Processing record {processed_count}/{len(locked_df)}: {chrom}:{pos}")

        # 获取上下文序列
        try:
            upstream = fetch_sequence_local(genome, chrom, pos - 1, pos - 1)
            downstream = fetch_sequence_local(genome, chrom, pos + 1, pos + 1)
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

    # 清洗 Context 列逻辑
    def clean_context(context):
        match = re.search(r'[ACGT]\[[ACGT]>[ACGT]\][ACGT]', context)
        return match.group() if match else "N/A"

    # 清洗上下文数据
    locked_df['Context'] = locked_df['Context'].apply(clean_context)

    # 按指定顺序选择和重命名列
    columns_to_keep = [
        'GENE_SYMBOL', 'CHROMOSOME', 'GENOME_START', 'GENOMIC_WT_ALLELE',
        'GENOMIC_MUT_ALLELE', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'Context'
    ]
    columns_to_rename = [
        'GENE_SYMBOL', 'Chromosome', 'Position', 'Reference',
        'Alternate', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'Context'
    ]

    output_df = locked_df[columns_to_keep]
    output_df.columns = columns_to_rename

    # 保存结果
    output_df.to_csv(outfile_name, sep="\t", index=False)
    print(f"Processing completed. Results saved to: {outfile_name}")


# 示例调用
# process_mutation_data_local(
#     input_file="path/to/mutation_data.tsv",
#     fasta_file="path/to/genome.fa",
#     outfile_name="output_file.tsv",
#     cancer_type="BRCA"
# )
