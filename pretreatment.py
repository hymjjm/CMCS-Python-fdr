import os
import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

async def fetch_sequence_async(chromosome, start, end, species="human", retries=3, delay=1.5):
    """
    异步从 Ensembl 获取基因组序列，支持重试机制
    """
    url = f"https://rest.ensembl.org/sequence/region/{species}/{chromosome}:{start}..{end}:1"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("seq", "").upper()
                    elif response.status == 429:
                        # print(f"Rate limit hit. Retrying after {delay * (attempt + 1)} seconds...")
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        print(f"Error {response.status}: {await response.text()}")
                        return "N/A"
        except Exception as e:
            print(f"Error fetching sequence: {e}")
            await asyncio.sleep(delay * (attempt + 1))
    print("Max retries reached.")
    return "N/A"


async def process_row(row, species="human"):
    """
    异步处理单条记录，获取上下文序列
    """
    chrom = row['CHROMOSOME']
    pos = row['GENOME_START']
    wt_allele = row['GENOMIC_WT_ALLELE']
    mut_allele = row['GENOMIC_MUT_ALLELE']

    try:
        # 获取上游和下游序列
        upstream = await fetch_sequence_async(chrom, pos - 6, pos - 1, species)
        downstream = await fetch_sequence_async(chrom, pos + 1, pos + 6, species)

        # 检查序列长度并调整为只取 1 个碱基
        if len(upstream) < 6:
            upstream = await fetch_sequence_async(chrom, pos - 1, pos - 1, species)
        else:
            upstream = upstream.rjust(6, "-")

        if len(downstream) < 6:
            downstream = await fetch_sequence_async(chrom, pos + 1, pos + 1, species)
        else:
            downstream = downstream.ljust(6, "-")
        # 拼接
        context = f"{upstream}[{wt_allele}>{mut_allele}]{downstream}"
        # 返回结果
        return context

    except Exception as e:
        print(f"Error processing row: {e}")
        return "N/A"

async def process_chunk_async(df_chunk, species="human"):
    """
    异步处理数据分片
    """
    tasks = [process_row(row, species) for _, row in df_chunk.iterrows()]
    return await tqdm.gather(*tasks, desc="Processing rows", total=len(df_chunk))

def process_mutation_data(input_file, df, outfile_name, cancer_type, batch_size, species):
    """
    主数据处理函数，支持异步批量请求和进度显示
    """
    print(f"Loading input file: {input_file}")

    # 筛选单碱基替换（SBS）
    df = df[(df['GENOMIC_WT_ALLELE'].str.len() == 1) & (df['GENOMIC_MUT_ALLELE'].str.len() == 1)]
    print(f"Records after SBS filtering: {len(df)}")

    # 添加或确认 'CANCER_TYPE' 列
    if 'CANCER_TYPE' not in df.columns:
        df['CANCER_TYPE'] = cancer_type

    # 添加或确认 'Sample' 列
    if 'Sample' not in df.columns:
        df['Sample'] = os.path.splitext(os.path.basename(input_file))[0]

    # 按批次处理数据
    contexts = []
    total_batches = len(df) // batch_size + int(len(df) % batch_size != 0)
    print(f"Processing {total_batches} batches...")

    for i, start in enumerate(range(0, len(df), batch_size)):
        df_chunk = df.iloc[start:start + batch_size]
        print(f"Processing batch {i + 1}/{total_batches}...")

        # 异步处理当前分片
        loop = asyncio.get_event_loop()
        chunk_contexts = loop.run_until_complete(process_chunk_async(df_chunk, species))
        contexts.extend(chunk_contexts)

    # 添加上下文信息到 DataFrame
    df['Context'] = contexts

    # 为 CHROMOSOME 列添加 "chr" 前缀
    df['CHROMOSOME'] = df['CHROMOSOME'].apply(lambda x: f"chr{x}" if not str(x).startswith("chr") else x)

    # 为 GENOME_VERSION 列添加 "GRCh" 前缀
    df['GENOME_VERSION'] = df['GENOME_VERSION'].apply(
        lambda x: f"GRCh{x}" if not str(x).startswith("GRCh") else x
    )

    # 按指定列顺序保存结果
    columns_to_keep = [
        'GENE_SYMBOL', 'CHROMOSOME', 'GENOME_START', 'GENOMIC_WT_ALLELE',
        'GENOMIC_MUT_ALLELE', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'GENOME_VERSION', 'Context'
    ]
    df = df[columns_to_keep]
    df.columns = [
        'GENE_SYMBOL', 'Chromosome', 'Position', 'Reference',
        'Alternate', 'STRAND', 'MUTATION_DESCRIPTION', 'Sample',
        'CANCER_TYPE', 'GENOME_VERSION', 'Context'
    ]

    # 保存结果
    df.to_csv(outfile_name, sep="\t", index=False)
    print(f"Processing completed. Results saved to: {outfile_name}")

    return df


# 示例调用
# process_mutation_data(
#     input_file="path_to_input_file.tsv",
#     df=pd.read_csv("path_to_input_file.tsv", sep="\t"),
#     outfile_name="path_to_output_file.tsv",
#     cancer_type="BRCA",
#     batch_size=100,
#     species="human"
# )
