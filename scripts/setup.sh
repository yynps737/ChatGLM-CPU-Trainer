#!/bin/bash
# 设置和配置环境的辅助脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 重置颜色

# 打印带颜色的信息
function print_info {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warning {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
print_info "检查Docker环境..."
if ! command -v docker &> /dev/null; then
    print_error "Docker未安装! 请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_warning "未找到docker-compose命令，尝试使用docker compose..."
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose未安装! 请安装Docker Compose"
        exit 1
    fi
    use_new_compose=true
else
    use_new_compose=false
fi

# 获取系统内存信息
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    total_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    total_memory_gb=$((total_memory_kb / 1024 / 1024))
elif [[ "$OSTYPE" == "darwin"* ]]; then
    total_memory_bytes=$(sysctl hw.memsize | awk '{print $2}')
    total_memory_gb=$((total_memory_bytes / 1024 / 1024 / 1024))
else
    print_warning "无法检测系统内存，将使用默认配置"
    total_memory_gb=8
fi

# 根据内存大小设置配置
print_info "检测到系统内存: ${total_memory_gb}GB"
if [[ $total_memory_gb -lt 6 ]]; then
    memory_config="4gb"
    print_info "将使用4GB内存优化配置"
elif [[ $total_memory_gb -lt 12 ]]; then
    memory_config="8gb"
    print_info "将使用8GB内存优化配置"
elif [[ $total_memory_gb -lt 24 ]]; then
    memory_config="16gb"
    print_info "将使用16GB内存优化配置"
else
    memory_config="32gb"
    print_info "将使用32GB内存优化配置"
fi

# 创建或更新.env文件
cp -f .env.example .env
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS的sed语法不同
    sed -i '' "s/MEMORY_CONFIG=default/MEMORY_CONFIG=$memory_config/" .env
    sed -i '' -e "/# ${memory_config}/,/# [0-9]/s/^# //" .env
else
    # Linux的sed语法
    sed -i "s/MEMORY_CONFIG=default/MEMORY_CONFIG=$memory_config/" .env
    sed -i -e "/# ${memory_config}/,/# [0-9]/s/^# //" .env
fi

print_info "环境配置已更新为${memory_config}配置"

# 建立必要的目录
mkdir -p data/input data/output models
print_info "目录结构已创建"

# 提示用户下一步操作
print_info "设置完成! 接下来的步骤:"
print_info "1. 将训练数据放入data/input/dataset.txt文件"
print_info "2. 构建Docker镜像: docker build -t chatglm-cpu-trainer ."
if [[ "$use_new_compose" == true ]]; then
    print_info "3. 开始训练: docker compose run train"
    print_info "4. 测试模型: docker compose run predict"
else
    print_info "3. 开始训练: docker-compose run train"
    print_info "4. 测试模型: docker-compose run predict"
fi