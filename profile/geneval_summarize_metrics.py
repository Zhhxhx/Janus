import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict

def summarize_metrics_from_jsonl(root_dir: str, output_file: str):
		"""
		遍历根目录，从所有 'metrics.jsonl' 文件中汇总指标，并计算平均值。
		"""
		all_records = []

		prompt_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

		for prompt_key in tqdm(prompt_dirs, desc="Reading metric files"):
				metrics_path = os.path.join(root_dir, prompt_key, "metrics.jsonl")
				if os.path.exists(metrics_path):
						with open(metrics_path, 'r', encoding='utf-8') as f:
								for line in f:
										try:
												record = json.loads(line)
												all_records.append(record)
										except json.JSONDecodeError:
												print(f"Warning: Could not decode JSON line in {metrics_path}")

		if not all_records:
				print("No metric records found. Exiting.")
				return

		# --- 将扁平的JSONL记录转换为适合DataFrame的格式 ---
		flat_records = []
		for record in all_records:
				key = record.get("key")
				
				# 处理 sparsity
				if "sparsity" in record:
						for cfg_type, values in record["sparsity"].items():
								flat_rec = {"key": key, "cfg_type": cfg_type, "metric_type": "sparsity"}
								flat_rec.update(values)
								flat_records.append(flat_rec)

				# 处理 mlp_similarity
				if "mlp_similarity" in record:
						for cfg_type, value in record["mlp_similarity"].items():
								flat_rec = {"key": key, "cfg_type": cfg_type, "metric_type": "mlp_similarity", "value": value}
								flat_records.append(flat_rec)

				# 处理 avg_self_attn_score
				if "avg_self_attn_score" in record:
						for cfg_type, value in record["avg_self_attn_score"].items():
								flat_rec = {"key": key, "cfg_type": cfg_type, "metric_type": "avg_self_attn_score", "value": value}
								flat_records.append(flat_rec)

		df = pd.DataFrame(flat_records)
		
		# --- 计算并格式化汇总结果 ---
		summary_lines = []

		# Sparsity
		df_sparsity = df[df['metric_type'] == 'sparsity']
		if not df_sparsity.empty:
				avg_sparsity = df_sparsity.groupby('cfg_type')[['vae_vit_sparsity', 'self_attn_sparsity']].mean().reset_index()
				summary_lines.append("--- Average Sparsity ---\n")
				summary_lines.append(avg_sparsity.to_string(index=False) + "\n")

		# MLP Similarity
		df_similarity = df[df['metric_type'] == 'mlp_similarity']
		if not df_similarity.empty:
				avg_similarity = df_similarity.groupby('cfg_type')['value'].mean().reset_index().rename(columns={'value': 'avg_mlp_similarity'})
				summary_lines.append("\n--- Average MLP Similarity ---\n")
				summary_lines.append(avg_similarity.to_string(index=False) + "\n")

		# Self-Attention Score
		df_attn = df[df['metric_type'] == 'avg_self_attn_score']
		if not df_attn.empty:
				avg_attn = df_attn.groupby('cfg_type')['value'].mean().reset_index().rename(columns={'value': 'avg_self_attn_score'})
				summary_lines.append("\n--- Average Self-Attention Score ---\n")
				summary_lines.append(avg_attn.to_string(index=False) + "\n")

		# 将汇总结果写入文件
		with open(output_file, 'w') as f:
				f.writelines(summary_lines)
		
		print(f"\nSummary report saved to {output_file}")
		for line in summary_lines:
				print(line.strip())


if __name__ == "__main__":
		parser = argparse.ArgumentParser(description="Summarize metrics from GenEval runs using JSONL files.")
		parser.add_argument("--root_dir", type=str, required=True,
												help="The root output directory containing all prompt subdirectories.")
		parser.add_argument("--output_file", type=str, default="metrics_summary.txt",
												help="The file to save the final summary report.")
		
		args = parser.parse_args()
		
		summarize_metrics_from_jsonl(args.root_dir, args.output_file)