# 本机压缩vexact并上传到oss
cd /Users/gonggu/Desktop/workplace/rl_mismatch/projects/ && rm -rf vexact.zip && zip -r vexact.zip vexact && ossutil cp -f vexact.zip oss://tre-normandy/t06/packs/gg/zip_files/verl_repos/ && cd vexact

# 远端服务器更新vexact
cd /workspace/gg/projects/vexact_repo/ && rm -rf __MACOSX vexact vexact.zip && cp /root/data/gg/zip_files/verl_repos/vexact.zip ./ && unzip vexact.zip && source /workspace/gg/venvs/vexact_0706/bin/activate && cd vexact && pip install -e . --no-deps


# 从远端服务器拷贝log到oss
export PROJECT_NAME=GRPO-QAT-W4A16-QWEN3-1-7b-Base-GSM8K-FOLD-TEST
export EXP_NAME=0710_gsm8k_qat_live_wq_path
cp /xpfs/fp4/gg/exp_ckpts/${PROJECT_NAME}/${EXP_NAME}/train.log /root/data/gg/zip_files/exp_logs/2026_07/terminal_logs/${EXP_NAME}.log

# 从oss拷贝log到本地
export EXP_NAME=0710_gsm8k_qat_live_wq_path
ossutil cp oss://tre-normandy/t06/packs/gg/zip_files/exp_logs/2026_07/terminal_logs/${EXP_NAME}.log /Users/gonggu/Desktop/workplace/rl_mismatch/exp_logs/terminal_logs/${EXP_NAME}.log

# 从oss拷贝tensorboard到本地
