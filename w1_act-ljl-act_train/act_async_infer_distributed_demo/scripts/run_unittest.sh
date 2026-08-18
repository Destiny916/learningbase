#!/bin/bash
# ================================================
# w1_act 项目单元测试一键运行脚本
# 项目结构: /home/workspace/code/w1_act
# 测试包路径: act_async_infer_distributed_demo.scripts.unittest
# ================================================

set -e  # 遇到错误时退出脚本

# 项目根目录（可以根据需要修改）
#PROJECT_ROOT="/home/workspace/code/w1_act"
# 或者使用脚本所在目录作为项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 测试包路径
TEST_PACKAGE="act_async_infer_distributed_demo.scripts.unittest"
# 测试目录相对路径
TEST_DIR_RELATIVE="act_async_infer_distributed_demo/scripts/unittest"
# 完整的测试目录路径
TEST_DIR="$PROJECT_ROOT/$TEST_DIR_RELATIVE"

# 默认配置
VERBOSE=true
USE_PYTEST=false
GENERATE_REPORT=false
COVERAGE=false
USE_ORIGINAL_METHOD=false  # 是否使用原来的执行方式

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
w1_act 项目单元测试脚本

项目路径: $PROJECT_ROOT
测试包: $TEST_PACKAGE

使用方法: $0 [选项]

选项:
  -h, --help          显示此帮助信息
  -q, --quiet         安静模式，减少输出
  --pytest            使用pytest运行测试
  --original          使用原来的执行方式 (python -m 包路径.test_*)
  --report            生成XML测试报告
  --coverage          计算测试覆盖率
  --list              列出所有测试模块
  --test MODULE       运行特定测试模块 (例如: test_module1)
  --all               运行所有测试模块（逐个运行）

示例:
  $0                    # 运行所有测试（使用unittest discover）
  $0 --original         # 使用原来的方式运行所有测试模块
  $0 --pytest          # 使用pytest运行测试
  $0 --coverage        # 运行测试并计算覆盖率
  $0 --test test_module1  # 运行特定测试模块
  $0 --list            # 列出所有测试模块
EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quiet)
            VERBOSE=false
            shift
            ;;
        --pytest)
            USE_PYTEST=true
            shift
            ;;
        --original)
            USE_ORIGINAL_METHOD=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --test)
            SPECIFIC_MODULE="$2"
            shift 2
            ;;
        --all)
            RUN_ALL_MODULES=true
            shift
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查项目目录是否存在
check_project_structure() {
    print_info "检查项目结构..."
    
    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "项目根目录不存在: $PROJECT_ROOT"
        exit 1
    fi
    
    if [ ! -d "$TEST_DIR" ]; then
        print_error "测试目录不存在: $TEST_DIR"
        echo "尝试查找测试目录..."
        find "$PROJECT_ROOT" -type d -name "unittest" | head -5
        exit 1
    fi
    
    print_info "项目根目录: $PROJECT_ROOT"
    print_info "测试目录: $TEST_DIR"
    
    # 切换到项目目录
    cd "$PROJECT_ROOT"
    print_info "已切换到项目目录: $(pwd)"
}

# 检查Python环境
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "未找到Python解释器"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "使用Python版本: $PYTHON_VERSION ($PYTHON_CMD)"
}

# 激活虚拟环境（如果存在）
activate_venv() {
    local venv_paths=("venv" ".venv" "../venv")
    
    for venv_path in "${venv_paths[@]}"; do
        if [ -f "$venv_path/bin/activate" ]; then
            print_info "激活虚拟环境: $venv_path"
            source "$venv_path/bin/activate"
            return 0
        fi
    done
    
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "未找到虚拟环境，使用系统Python"
    else
        print_info "已在虚拟环境中: $VIRTUAL_ENV"
    fi
}

# 列出所有测试模块
list_test_modules() {
    print_info "测试目录: $TEST_DIR"
    echo ""
    
    # 查找所有测试文件
    local test_files=()
    while IFS= read -r -d '' file; do
        test_files+=("$file")
    done < <(find "$TEST_DIR" -name "test_*.py" -type f -print0)
    
    if [ ${#test_files[@]} -eq 0 ]; then
        print_warning "未找到 test_*.py 测试文件"
        echo "当前目录内容:"
        ls -la "$TEST_DIR"
    else
        echo "找到以下测试模块 (${#test_files[@]}个):"
        
        # 提取模块名
        for file in "${test_files[@]}"; do
            # 获取相对路径
            rel_path="${file#$PROJECT_ROOT/}"
            # 转换为Python模块路径
            module_path="${rel_path%.py}"
            module_path="${module_path//\//.}"
            
            # 获取文件名
            filename=$(basename "$file")
            module_name="${filename%.py}"
            
            echo "  - 文件: $filename"
            echo "    模块: $module_path"
            echo "    原始方式: $PYTHON_CMD -m $module_path"
            echo ""
        done
    fi
}

# 获取所有测试模块名
get_all_test_modules() {
    local modules=()
    while IFS= read -r -d '' file; do
        rel_path="${file#$PROJECT_ROOT/}"
        module_path="${rel_path%.py}"
        module_path="${module_path//\//.}"
        modules+=("$module_path")
    done < <(find "$TEST_DIR" -name "test_*.py" -type f -print0)
    echo "${modules[@]}"
}

# 使用原来的执行方式运行单个模块
run_original_module() {
    local module_name="$1"
    print_info "运行模块: $module_name"
    
    if [ "$VERBOSE" = true ]; then
        $PYTHON_CMD -m "$module_name"
    else
        $PYTHON_CMD -m "$module_name" 2>/dev/null || true
    fi
    
    echo "----------------------------------------"
}

# 使用原来的方式运行所有模块（逐个运行）
run_original_all() {
    print_info "使用原始方式运行所有测试模块..."
    
    local modules=($(get_all_test_modules))
    
    if [ ${#modules[@]} -eq 0 ]; then
        print_error "未找到测试模块"
        return 1
    fi
    
    local total=${#modules[@]}
    local success=0
    local failed=0
    
    for module in "${modules[@]}"; do
        echo ""
        print_info "运行模块 ($((success + failed + 1))/$total): $module"
        
        if $PYTHON_CMD -m "$module" 2>&1; then
            print_success "模块 $module 运行成功"
            ((success++))
        else
            print_error "模块 $module 运行失败"
            ((failed++))
        fi
        
        echo "----------------------------------------"
    done
    
    echo ""
    print_info "测试完成汇总:"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "  总计: $total"
    
    if [ $failed -eq 0 ]; then
        print_success "所有测试模块运行成功！"
        return 0
    else
        print_error "有 $failed 个测试模块失败"
        return 1
    fi
}

# 使用unittest discover运行测试
run_unittest_discover() {
    print_info "使用 unittest discover 运行测试..."
    
    local unittest_cmd=("$PYTHON_CMD" -m unittest discover \
        -s "$TEST_DIR" \
        -p "test_*.py" \
        -v)
    
    if [ "$GENERATE_REPORT" = true ]; then
        REPORT_FILE="test_results_$(date +%Y%m%d_%H%M%S).xml"
        print_info "测试报告将保存到: $REPORT_FILE"
        "${unittest_cmd[@]}" 2>&1 | tee "$REPORT_FILE"
    else
        "${unittest_cmd[@]}"
    fi
}

# 使用pytest运行测试
run_pytest() {
    print_info "使用 pytest 运行测试..."
    
    if ! $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        print_error "pytest未安装，请先安装: pip install pytest"
        exit 1
    fi
    
    local pytest_cmd=("$PYTHON_CMD" -m pytest "$TEST_DIR")
    
    if [ "$VERBOSE" = true ]; then
        pytest_cmd+=("-v")
    fi
    
    if [ -n "$SPECIFIC_MODULE" ]; then
        # 如果指定了模块，运行该模块
        pytest_cmd+=("$TEST_DIR/test_${SPECIFIC_MODULE}.py")
    fi
    
    if [ "$GENERATE_REPORT" = true ]; then
        REPORT_FILE="pytest_results_$(date +%Y%m%d_%H%M%S).xml"
        pytest_cmd+=("--junitxml=$REPORT_FILE")
        print_info "测试报告将保存到: $REPORT_FILE"
    fi
    
    "${pytest_cmd[@]}"
}

# 运行特定测试模块（原始方式）
run_specific_module() {
    local module_name="$TEST_PACKAGE.test_$SPECIFIC_MODULE"
    
    if [ ! -f "$TEST_DIR/test_$SPECIFIC_MODULE.py" ]; then
        print_error "测试模块不存在: test_$SPECIFIC_MODULE.py"
        echo "可用的测试模块:"
        find "$TEST_DIR" -name "test_*.py" -exec basename {} \; | sed 's/^/  - /'
        exit 1
    fi
    
    print_info "运行特定测试模块: $module_name"
    $PYTHON_CMD -m "$module_name"
}

# 运行覆盖率测试
run_coverage() {
    print_info "运行测试覆盖率分析..."
    
    if ! $PYTHON_CMD -c "import coverage" 2>/dev/null; then
        print_error "coverage未安装，请先安装: pip install coverage"
        exit 1
    fi
    
    # 设置coverage要监控的源代码路径
    local source_dir="act_async_infer_distributed_demo"
    
    # 运行测试并收集覆盖率数据
    if [ "$USE_PYTEST" = true ]; then
        $PYTHON_CMD -m coverage run --source="$source_dir" -m pytest "$TEST_DIR"
    elif [ "$USE_ORIGINAL_METHOD" = true ]; then
        $PYTHON_CMD -m coverage run --source="$source_dir" -m pytest "$TEST_DIR"
    else
        $PYTHON_CMD -m coverage run --source="$source_dir" -m unittest discover -s "$TEST_DIR" -p "test_*.py"
    fi
    
    # 生成覆盖率报告
    echo ""
    print_info "生成覆盖率报告..."
    $PYTHON_CMD -m coverage report -m
    $PYTHON_CMD -m coverage html -d "htmlcov"
    print_info "HTML覆盖率报告已生成到: htmlcov/index.html"
}

# 主函数
main() {
    print_info "启动 w1_act 项目单元测试..."
    echo "========================================"
    
    # 检查项目结构
    check_project_structure
    
    # 检查Python环境
    check_python
    
    # 激活虚拟环境
    activate_venv
    
    # 如果只需要列出测试模块
    if [ "$LIST_ONLY" = true ]; then
        list_test_modules
        exit 0
    fi
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 根据选项运行测试
    if [ "$COVERAGE" = true ]; then
        run_coverage
    elif [ -n "$SPECIFIC_MODULE" ]; then
        run_specific_module
    elif [ "$RUN_ALL_MODULES" = true ]; then
        run_original_all
    elif [ "$USE_ORIGINAL_METHOD" = true ]; then
        run_original_all
    elif [ "$USE_PYTEST" = true ]; then
        run_pytest
    else
        run_unittest_discover
    fi
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "========================================"
    print_success "测试完成！耗时: ${DURATION}秒"
    
    # 提示如何查看报告
    if [ "$COVERAGE" = true ]; then
        print_info "打开覆盖率报告:"
        echo "  open htmlcov/index.html  # macOS"
        echo "  xdg-open htmlcov/index.html  # Linux"
    fi
}

# 执行主函数
main

