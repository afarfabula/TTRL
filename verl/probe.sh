#!/usr/bin/env bash
# probe.sh — 一键查看 SPS-TTRL(GPU0-3) 与 Majority-Voting TTRL(GPU4-7) 两条训练的状态
# 用法: bash /opt/tiger/TTRL/verl/probe.sh
set -u

SPS_SESS=/tmp/ray/session_latest
MV_SESS=/tmp/ray_majvote/ray/session_latest
SPS_LOG=/opt/tiger/TTRL/verl/sps_ttrl_4gpu.log
MV_LOG=/opt/tiger/TTRL/verl/majvote_ttrl_4gpu.log
TOTAL=310

hr(){ printf '%s\n' "------------------------------------------------------------"; }

# 取最新 step: metrics 行均以 "step:N - ..." 开头(N 为整数), 提取最大 N
last_step(){
  local sess="$1"
  grep -ahoE "step:[0-9]+ -" "$sess"/logs/worker-*.out 2>/dev/null \
    | awk -F'[: ]' '{print $2}' | sort -n | tail -1
}

# 打印一条 run 的 val 表 (只挑带 val 的 step)
val_table(){
  local sess="$1"
  printf "%-6s %-9s %-8s %-8s %-8s\n" "step" "epoch" "mean@4" "maj@4" "best@4"
  for s in 0 $(seq 5 5 $TOTAL); do
    local line
    line=$(grep -aE "step:$s -" "$sess"/logs/worker-*.out 2>/dev/null | head -1)
    [ -z "$line" ] && continue
    echo "$line" | grep -aq "val-core/MATH-TTT/acc/mean@4" || continue
    local ep mean maj best
    ep=$(echo "$line"   | grep -aoE "training/epoch:[0-9.]+" | head -1 | cut -d: -f2)
    mean=$(echo "$line" | grep -aoE "val-core/MATH-TTT/acc/mean@4:[0-9.]+" | head -1 | cut -d: -f2)
    maj=$(echo "$line"  | grep -aoE "val-core/MATH-TTT/acc/maj@4/mean:[0-9.]+" | head -1 | cut -d: -f2)
    best=$(echo "$line" | grep -aoE "val-core/MATH-TTT/acc/best@4/mean:[0-9.]+" | head -1 | cut -d: -f2)
    printf "%-6s %-9s %-8s %-8s %-8s\n" "$s" "${ep:-0}" "${mean:-—}" "${maj:-—}" "${best:-—}"
  done
}

# 打印最近 N 个训练 step 的健康度
health(){
  local sess="$1"; local lst="$2"
  [ -z "$lst" ] && { echo "  (尚无训练 step)"; return; }
  printf "%-6s %-10s %-9s %-10s %-9s\n" "step" "effK" "grad" "pg_loss" "resp_len"
  local start=$(( lst>6 ? lst-5 : 1 ))
  for s in $(seq $start $lst); do
    local line; line=$(grep -aE "step:$s -" "$sess"/logs/worker-*.out 2>/dev/null | head -1)
    [ -z "$line" ] && continue
    local effk gn pg rl
    effk=$(echo "$line" | grep -aoE "train/sps/effective_K:[0-9.]+" | head -1 | cut -d: -f2)
    gn=$(echo "$line"   | grep -aoE "actor/grad_norm:[0-9.]+" | head -1 | cut -d: -f2)
    pg=$(echo "$line"   | grep -aoE "actor/pg_loss:[0-9.eE-]+" | head -1 | cut -d: -f2)
    rl=$(echo "$line"   | grep -aoE "response_length/mean:[0-9.]+" | head -1 | cut -d: -f2)
    printf "%-6s %-10s %-9s %-10s %-9s\n" "$s" "${effk:-NA}" "${gn:-NA}" "${pg:-NA}" "${rl:-NA}"
  done
}

# 报错检测
errs(){ grep -aE "Traceback|RuntimeError|Error executing|CUDA out of memory|Killed|NCCL error" "$1" 2>/dev/null | tail -3; }

# ETA
eta(){
  local lst="$1"; [ -z "$lst" ] && { echo "  (未开始计步)"; return; }
  local remain=$(( TOTAL - lst ))
  [ "$remain" -lt 0 ] && remain=0
  # 平均每步约 94s (4 普通步*50s + 1 val步*270s)/5
  local sec=$(( remain * 94 ))
  printf "  已完成 %s/%s, 剩 %s 步, 预计还需约 %d 分钟 (~%d 小时%d分)\n" \
    "$lst" "$TOTAL" "$remain" "$((sec/60))" "$((sec/3600))" "$(((sec%3600)/60))"
}

echo "================ TTRL 训练探测  $(date '+%H:%M:%S') ================"
echo "### GPU 状态 (0-3=SPS, 4-7=MajVote)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

SPS_LAST=$(last_step "$SPS_SESS")
MV_LAST=$(last_step "$MV_SESS")

hr
echo "■■■ [1] SPS-TTRL  (GPU 0-3)   进度: step ${SPS_LAST:-初始化中} / $TOTAL"
eta "$SPS_LAST"
echo "-- val 曲线 --"; val_table "$SPS_SESS"
echo "-- 最近训练健康度 --"; health "$SPS_SESS" "$SPS_LAST"
e=$(errs "$SPS_LOG"); [ -n "$e" ] && { echo "!! 报错:"; echo "$e"; } || echo ">> 无报错"

hr
echo "■■■ [2] Majority-Voting  (GPU 4-7)   进度: step ${MV_LAST:-初始化/首次val中} / $TOTAL"
eta "$MV_LAST"
echo "-- val 曲线 --"; val_table "$MV_SESS"
e=$(errs "$MV_LOG"); [ -n "$e" ] && { echo "!! 报错:"; echo "$e"; } || echo ">> 无报错"

hr
echo "■■■ [3] 对比 (mean@4)"
sps_cur=$(val_table "$SPS_SESS" 2>/dev/null | tail -n +2 | awk 'NF{v=$3} END{print v}')
mv_cur=$(val_table "$MV_SESS" 2>/dev/null | tail -n +2 | awk 'NF{v=$3} END{print v}')
echo "  SPS 最新 mean@4 = ${sps_cur:-—}   |   MajVote 最新 mean@4 = ${mv_cur:-(暂无)}   (baseline≈0.537)"
echo "================================================================"
