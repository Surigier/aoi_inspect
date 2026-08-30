#!/bin/bash
# Web演示台启动器:pidfile管理,不做任何模式匹配杀进程(pkill/pgrep -f自匹配已累计坑8次)。
# 本机(4060L WSL)环境损伤会在导入阶段随机炸,起不来就多试几轮。
cd "$(dirname "$0")/.."
mkdir -p _logs   # _logs/被gitignore(实验日志目录),全新解压的交付包里不存在,需现建
PIDF=_logs/app.pid
[ -f "$PIDF" ] && kill -9 "$(cat "$PIDF")" 2>/dev/null
sleep 1
set -a; [ -f .env ] && . ./.env; set +a
for try in 1 2 3 4 5 6 7 8; do
  nohup env PYTHONPATH=. PYTHONFAULTHANDLER=1 python3 -u scripts/app.py --port "${1:-7860}" > _logs/app.log 2>&1 &
  echo $! > "$PIDF"
  sleep 27
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "http://127.0.0.1:${1:-7860}/" || echo 000)
  echo "try$try: HTTP=$code"
  [ "$code" = "200" ] && exit 0
  kill -9 "$(cat "$PIDF")" 2>/dev/null
  sleep 2
done
echo "8次均失败,最后日志:"; tail -5 _logs/app.log
exit 1
