#!/bin/bash
# Web演示台启动器:pidfile管理,不做任何模式匹配杀进程(pkill/pgrep -f自匹配已累计坑8次)。
#
# 【根治,不是绕过】2026-08-30 实测抓到崩溃真凶:gradio导入时默认会开一个后台
# 线程,请求 https://api.gradio.app/gradio-messaging/en 查"有没有新公告"
# (gradio/strings.py,与我们的代码逻辑无关),这次网络请求的SSL握手在本机
# WSL环境下会直接把整个Python解释器崩掉("Fatal Python error: Unreachable
# C code path reached"),不是异常、无法被try/except捕获,表现为"随机"启动
# 失败——实为每次都会崩,只是偶尔那个线程跑得比主线程慢、赶在探测前就挂了。
# gradio源码本身认这个环境变量(严格等于"True"才起线程),设成非True值即可
# 彻底关闭这条网络请求,从根上消除崩溃诱因,而非"多试几次赌一次不崩"。
export GRADIO_ANALYTICS_ENABLED=False
cd "$(dirname "$0")/.."
mkdir -p _logs   # _logs/被gitignore(实验日志目录),全新解压的交付包里不存在,需现建
# 用install.sh装依赖时记下的那个python3(见install.sh末尾),而不是直接写死
# "python3"——同一台机器上可能有多套Python(系统自带/某个虚拟环境),终端不同、
# 有没有激活虚拟环境都会导致"python3"实际指向不同解释器,装的地方和跑的地方
# 对不上就报ModuleNotFoundError。找不到记录时才退回裸"python3"。
PYBIN=python3
[ -f _logs/python_bin.txt ] && [ -x "$(cat _logs/python_bin.txt)" ] && PYBIN="$(cat _logs/python_bin.txt)"
PIDF=_logs/app.pid
[ -f "$PIDF" ] && kill -9 "$(cat "$PIDF")" 2>/dev/null
sleep 1
set -a; [ -f .env ] && . ./.env; set +a
for try in 1 2 3 4 5 6 7 8; do
  nohup env PYTHONPATH=. PYTHONFAULTHANDLER=1 "$PYBIN" -u scripts/app.py --port "${1:-7860}" > _logs/app.log 2>&1 &
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
