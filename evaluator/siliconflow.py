import time
import re
import json
import os
import sys
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 加载.env文件
load_dotenv()

# 与 data_analysis/utils.py 同目录
_da = Path(__file__).resolve().parent
if str(_da) not in sys.path:
    sys.path.insert(0, str(_da))

try:
    from utils import LLM 
except ImportError:
    print("错误：无法导入 'utils.py'。请确保该文件存在于脚本的同一目录或Python路径中。")
    exit()

# --- LLM实例初始化 ---
LLM_PLATFORM = 'siliconflow'  # 或者根据您的utils.py配置
LLM_MODEL_NAME = 'deepseek-ai/DeepSeek-V3.2'

# 为多线程创建LLM实例池
LLM_INSTANCES = []
LLM_INSTANCE_LOCK = threading.Lock()

def get_llm_instance():
    """获取或创建LLM实例（线程安全）"""
    with LLM_INSTANCE_LOCK:
        if not LLM_INSTANCES:
            try:
                # 为每个线程创建独立的LLM实例
                for _ in range(10):
                    instance = LLM(model_name=LLM_MODEL_NAME, platform=LLM_PLATFORM)
                    LLM_INSTANCES.append(instance)
                print(f"已创建 {len(LLM_INSTANCES)} 个LLM实例用于多线程处理")
            except Exception as e:
                print(f"错误：无法初始化LLM实例。请检查您的 'utils.py' 文件和API密钥配置。错误信息: {e}")
                exit()
        return LLM_INSTANCES.pop() if LLM_INSTANCES else None

def return_llm_instance(instance):
    """归还LLM实例到池中"""
    with LLM_INSTANCE_LOCK:
        LLM_INSTANCES.append(instance)

# 不在 import 时创建 LLM：否则即使用 OpenRouter 跑 evaluator，也会要求 SILICONFLOW_API_KEY。
LLM_INSTANCE = None


def _ensure_legacy_llm_instance():
    """旧测试/脚本用的 SiliconFlow 单例，仅在调用时创建。"""
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        LLM_INSTANCE = LLM(model_name=LLM_MODEL_NAME, platform=LLM_PLATFORM)
    return LLM_INSTANCE

# 模型价格配置 (每1000 tokens的价格，单位：美元)
MODEL_PRICING = {
    "deepseek-ai/DeepSeek-V3": {
        "input": 0.00028,    # SiliconFlow pricing: ¥2/M tokens ≈ $0.28/M tokens ≈ $0.00028/1K tokens
        "output": 0.00112    # SiliconFlow pricing: ¥8/M tokens ≈ $1.12/M tokens ≈ $0.00112/1K tokens
    },
    "deepseek-ai/DeepSeek-R1": {
        "input": 0.00056,    # SiliconFlow pricing: ¥4/M tokens
        "output": 0.00224    # SiliconFlow pricing: ¥16/M tokens
    },
    "Qwen/Qwen3-235B-A22B": {
        "input": 0.00035,    # SiliconFlow pricing: ¥2.5/M tokens
        "output": 0.0014     # SiliconFlow pricing: ¥10/M tokens
    },
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "input": 0.00035,    # SiliconFlow pricing: ¥2.5/M tokens (假设与Qwen3-235B-A22B相同)
        "output": 0.0014     # SiliconFlow pricing: ¥10/M tokens
    }
}

# 实验统计类
class ExperimentTracker:
    def __init__(self, experiment_id: str, output_file: str = "experiment_stats.json"):
        self.experiment_id = experiment_id
        self.output_file = output_file
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = datetime.now().isoformat()
        self.lock = threading.Lock()  # 添加线程锁
        
    def add_call(self, usage, model_name):
        """添加一次API调用记录（线程安全）"""
        if usage:
            print(f"[DEBUG] add_call: 开始添加调用记录...", flush=True)
            with self.lock:
                print(f"[DEBUG] add_call: 获取锁成功", flush=True)
                self.total_calls += 1
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens
                self.total_tokens += usage.total_tokens
                
                # 计算成本
                print(f"[DEBUG] add_call: 计算成本...", flush=True)
                cost = calculate_cost(usage, model_name)
                self.total_cost += cost
                print(f"[DEBUG] add_call: 统计更新完成", flush=True)
            
            # 在锁外保存统计，避免死锁
            print(f"[DEBUG] add_call: 准备保存统计（在锁外）...", flush=True)
            self.save_stats()
            print(f"[DEBUG] add_call: 保存统计完成", flush=True)
    
    def save_stats(self):
        """保存统计数据到JSON文件（线程安全）"""
        print(f"[DEBUG] save_stats: 开始保存统计到 {self.output_file}...", flush=True)
        # 注意：这个函数应该在锁外调用，或者使用内部方法
        with self.lock:
            print(f"[DEBUG] save_stats: 获取锁成功", flush=True)
            stats = {
                "experiment_id": self.experiment_id,
                "start_time": self.start_time,
                "last_updated": datetime.now().isoformat(),
                "total_stats": {
                    "total_calls": self.total_calls,
                    "total_input_token": self.total_input_tokens,
                    "total_output_token": self.total_output_tokens,
                    "total_token": self.total_tokens,
                    "total_cost": round(self.total_cost, 6),
                    "total_cost_cny": round(self.total_cost * 7.2, 6)  # 添加人民币成本
                }
            }
            print(f"[DEBUG] save_stats: 统计数据准备完成", flush=True)
        
        # 在锁外进行文件写入，避免长时间持有锁
        print(f"[DEBUG] save_stats: 准备写入文件（在锁外）...", flush=True)
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] save_stats: 文件写入完成", flush=True)
        except Exception as e:
            print(f"[DEBUG] save_stats: 文件写入失败: {e}", flush=True)
            raise
    
    def get_stats(self):
        """获取当前统计数据"""
        return {
            "experiment_id": self.experiment_id,
            "total_stats": {
                "total_calls": self.total_calls,
                "total_input_token": self.total_input_tokens,
                "total_output_token": self.total_output_tokens,
                "total_token": self.total_tokens,
                "total_cost": round(self.total_cost, 6)
            }
        }

def calculate_cost(usage, model_name):
    """计算API调用成本"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
    output_cost = (usage.completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost
# --- 单个测试用例 ---

def test_single_finding():
    """测试单个研究发现的验证"""
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker("LLM-test-001", "experiment_stats.json")
    
    # 测试用例
    finding = "Metal nanoparticles are synthesized using a green synthesis method from the fresh flowers of Clitoria ternatea."
    
    system_message = """You are a highly analytical AI research assistant. Your task is to find reliable academic sources that directly support a given claim. You must base your decision on verifiable sources such as peer-reviewed journal articles, conference papers, or official publications."""
    
    json_structure_example = """
    {
      "analysis_summary": "A brief, one-sentence summary of your findings based on the sources you found.",
      "selected_sources": [
        {
          "title": "The exact title of the selected paper.",
          "doi": "The paper's DOI. Use null if not available.",
          "source_url": "The URL where the paper can be accessed."
        }
      ]
    }
    """

    prompt = f"""
    **Claim to Verify:** "{finding}"

    **Your Task:**
    1. Search for 5 reliable, peer-reviewed source that directly supports the claim.
    2. Evaluate their abstracts or content to ensure relevance.
    3. Only include sources that clearly support the claim.
    4. If no reliable sources are found, return an empty `selected_sources` array.
    5. Format your answer strictly as a single JSON object following the required structure.

    **Required JSON Structure:**
    {json_structure_example}
    """
    
    print("="*60)
    print(f"模型: {LLM_MODEL_NAME}")
    print(f"测试Finding: {finding}")
    print("="*60)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        llm = _ensure_legacy_llm_instance()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        raw_response_str = llm.generate(
            prompt=messages,
            temperature=0.0,
            web_search=False
        )
        
        # 计算运行时间
        end_time = time.time()
        runtime = end_time - start_time
        
        # 获取响应内容
        response_content = raw_response_str
        
        # 调试信息：检查last_usage的状态
        print(f"\n调试信息:")
        print(f"LLM_INSTANCE.last_usage 是否存在: {hasattr(llm, 'last_usage')}")
        print(f"LLM_INSTANCE.last_usage 值: {llm.last_usage}")
        if hasattr(llm, 'last_usage') and llm.last_usage:
            print(f"Usage 对象类型: {type(llm.last_usage)}")
            print(f"Usage 对象属性: {dir(llm.last_usage)}")
        
        # 尝试获取token使用量（如果LLM_INSTANCE支持）
        if hasattr(llm, 'last_usage') and llm.last_usage:
            usage = llm.last_usage
            cost = calculate_cost(usage, LLM_MODEL_NAME)
            
            # 添加到实验跟踪器
            tracker.add_call(usage, LLM_MODEL_NAME)
            
            print("\n" + "="*50)
            print("API调用统计")
            print("="*50)
            print(f"输入Token: {usage.prompt_tokens:,}")
            print(f"输出Token: {usage.completion_tokens:,}")
            print(f"总Token: {usage.total_tokens:,}")
            print(f"成本: ${cost:.6f} (SiliconFlow平台)")
            print(f"成本: ¥{cost * 7.2:.6f} (人民币)")
            print(f"运行时间: {runtime:.2f}秒")
            print("="*50)
            
            # 显示累计统计
            print("\n" + "="*50)
            print("累计实验统计")
            print("="*50)
            stats = tracker.get_stats()
            print(f"实验ID: {stats['experiment_id']}")
            print(f"总调用次数: {stats['total_stats']['total_calls']}")
            print(f"总输入Token: {stats['total_stats']['total_input_token']:,}")
            print(f"总输出Token: {stats['total_stats']['total_output_token']:,}")
            print(f"总Token: {stats['total_stats']['total_token']:,}")
            print(f"总成本: ${stats['total_stats']['total_cost']:.6f}")
            print(f"总成本: ¥{stats['total_stats']['total_cost'] * 7.2:.6f}")
            print("="*50)
        else:
            print(f"\n运行时间: {runtime:.2f}秒")
            print("警告: 无法获取token使用量信息")
        
        print("\n" + "="*50)
        print("模型响应")
        print("="*50)
        print(response_content)
        
        # 尝试解析JSON响应
        try:
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                print("\n" + "="*50)
                print("解析后的JSON结果")
                print("="*50)
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            else:
                print("\n警告: 响应中未找到有效的JSON格式")
        except json.JSONDecodeError as e:
            print(f"\n警告: JSON解析失败: {e}")
            
    except Exception as e:
        print(f"\n错误: API调用失败: {e}")

def create_new_experiment(experiment_id: str, output_file: str = None):
    """创建新的实验跟踪器"""
    if output_file is None:
        output_file = f"experiment_{experiment_id}.json"
    return ExperimentTracker(experiment_id, output_file)

def run_experiment_with_tracking(experiment_id: str, findings: list, output_file: str = None):
    """运行带跟踪的实验"""
    tracker = create_new_experiment(experiment_id, output_file)
    
    print(f"开始实验: {experiment_id}")
    print(f"输出文件: {tracker.output_file}")
    print("="*60)
    
    for i, finding in enumerate(findings, 1):
        print(f"\n处理第 {i}/{len(findings)} 个发现...")
        test_single_finding_with_tracker(finding, tracker)
    
    # 最终统计
    print("\n" + "="*60)
    print("实验完成 - 最终统计")
    print("="*60)
    final_stats = tracker.get_stats()
    print(json.dumps(final_stats, indent=2, ensure_ascii=False))
    print(f"统计数据已保存到: {tracker.output_file}")

def test_single_finding_with_tracker(finding: str, tracker: ExperimentTracker):
    """使用跟踪器测试单个发现"""
    system_message = """You are a highly analytical AI research assistant. Your task is to find reliable academic sources that directly support a given claim. You must base your decision on verifiable sources such as peer-reviewed journal articles, conference papers, or official publications."""
    
    json_structure_example = """
    {
      "analysis_summary": "A brief, one-sentence summary of your findings based on the sources you found.",
      "selected_sources": [
        {
          "title": "The exact title of the selected paper.",
          "doi": "The paper's DOI. Use null if not available.",
          "source_url": "The URL where the paper can be accessed."
        }
      ]
    }
    """

    prompt = f"""
    **Claim to Verify:** "{finding}"

    **Your Task:**
    1. Search for 5 reliable, peer-reviewed source that directly supports the claim.
    2. Evaluate their abstracts or content to ensure relevance.
    3. Only include sources that clearly support the claim.
    4. If no reliable sources are found, return an empty `selected_sources` array.
    5. Format your answer strictly as a single JSON object following the required structure.

    **Required JSON Structure:**
    {json_structure_example}
    """
    
    print(f"验证发现: {finding}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        llm = _ensure_legacy_llm_instance()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        raw_response_str = llm.generate(
            prompt=messages,
            temperature=0.0,
            web_search=False
        )
        
        # 计算运行时间
        end_time = time.time()
        runtime = end_time - start_time
        
        # 尝试获取token使用量并添加到跟踪器
        if hasattr(llm, 'last_usage') and llm.last_usage:
            usage = llm.last_usage
            tracker.add_call(usage, LLM_MODEL_NAME)
            
            print(f"  - 输入Token: {usage.prompt_tokens:,}")
            print(f"  - 输出Token: {usage.completion_tokens:,}")
            print(f"  - 成本: ${calculate_cost(usage, LLM_MODEL_NAME):.6f}")
            print(f"  - 运行时间: {runtime:.2f}秒")
        else:
            print(f"  - 运行时间: {runtime:.2f}秒")
            print("  - 警告: 无法获取token使用量信息")
            
    except Exception as e:
        print(f"  - 错误: API调用失败: {e}")

def extract_topic_from_metadata(row, tracker: ExperimentTracker, llm_instance=None):
    """从文章metadata中提取topic，生成Query字段
    
    参数:
        row: pandas Series，包含文章metadata
        tracker: ExperimentTracker实例，用于记录统计
        llm_instance: LLM实例，如果为None则从池中获取
    """
    
    # 获取LLM实例
    instance = llm_instance if llm_instance else get_llm_instance()
    if not instance:
        return None, None, {'error': '无法获取LLM实例'}
    
    try:
        # 构建metadata信息（pandas Series支持.get()方法）
        title = row.get('display_name', '')
        primary_topic = row.get('primary_topic.display_name', '')
        authors = row.get('authorships.author.display_name', '')
        journal = row.get('primary_location.source.display_name', '')
        year = row.get('publication_year', '')
        doi = row.get('doi', '')
        
        # 构建metadata字符串
        metadata = f"""Title: {title}
Primary Topic: {primary_topic}
Authors: {authors}
Journal: {journal}
Year: {year}
DOI: {doi}"""
        
        system_message = """You are an expert research assistant. Your task is to extract the main research topic from a review article's metadata. 
The topic should be concise, specific, and suitable for a literature review query. 
Return ONLY the topic phrase, without any additional explanation or formatting."""
        
        prompt = f"""Based on the following review article metadata, extract the main research topic. 
The topic should be a concise phrase that captures the core subject of this review article.

**Article Metadata:**
{metadata}

**Instructions:**
1. Analyze the title, primary topic, and other metadata to identify the main research topic.
2. Extract a concise, specific topic phrase (typically 2-8 words).
3. The topic should be suitable for a literature review query.
4. Return ONLY the topic phrase, nothing else.

**Example:**
If the title is "Recent advances in 3D printing of tough hydrogels: A review" and primary topic is "3D Printing in Biomedical Research", 
the extracted topic should be: "3D printing of tough hydrogels"

**Extracted Topic:**"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        response = instance.generate(
            prompt=messages,
            temperature=0.3,
            web_search=False
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # 清理响应，提取topic
        topic = response.strip()
        # 移除可能的引号
        topic = topic.strip('"').strip("'")
        # 移除可能的"Topic:"前缀
        topic = re.sub(r'^topic:\s*', '', topic, flags=re.IGNORECASE)
        topic = topic.strip()
        
        # 生成Query字段
        query = f"Conduct a literature review on {topic}."
        
        # 记录token使用量（线程安全）
        if hasattr(instance, 'last_usage') and instance.last_usage:
            usage = instance.last_usage
            with LLM_INSTANCE_LOCK:
                tracker.add_call(usage, LLM_MODEL_NAME)
            cost = calculate_cost(usage, LLM_MODEL_NAME)
            return topic, query, {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'cost': cost,
                'runtime': runtime
            }
        else:
            return topic, query, {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'cost': 0.0,
                'runtime': runtime
            }
            
    except Exception as e:
        return None, None, {'error': str(e)}
    finally:
        # 归还LLM实例到池中
        if not llm_instance:
            return_llm_instance(instance)


def process_sampled_csv(input_file='sampled_500.csv', output_file='sampled_500_with_query.csv', 
                        start_idx=0, end_idx=None, batch_size=10, resume=True, num_threads=10):
    """处理sampled_500.csv，为每篇文章提取topic并生成Query字段
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        start_idx: 开始处理的索引（从0开始）
        end_idx: 结束处理的索引（None表示处理到最后）
        batch_size: 每处理多少条保存一次
        resume: 是否启用断点续传（如果输出文件已存在，会跳过已处理的记录）
    """
    
    print("="*60)
    print("开始处理CSV文件，提取topic并生成Query字段")
    print("="*60)
    
    # 读取CSV文件
    print(f"\n正在读取: {input_file}")
    df = pd.read_csv(input_file)
    print(f"总记录数: {len(df)}")
    
    # 断点续传：如果输出文件已存在，读取已处理的数据
    if resume and os.path.exists(output_file):
        print(f"\n检测到输出文件已存在: {output_file}")
        print("正在加载已处理的数据...")
        try:
            df_existing = pd.read_csv(output_file)
            # 检查列是否匹配
            if len(df_existing) == len(df):
                # 合并已处理的Query列
                if 'Query' in df_existing.columns:
                    df['Query'] = df_existing['Query']
                
                # 统计已处理的记录数（只检查Query列）
                processed_count = (
                    df['Query'].notna() & 
                    (df['Query'].astype(str).str.strip() != '')
                )
                processed_num = processed_count.sum()
                print(f"已处理记录数: {processed_num}/{len(df)}")
                
                if processed_num > 0:
                    print("将跳过已处理的记录，从未处理的记录继续...")
            else:
                print("警告: 输出文件记录数与输入文件不匹配，将重新开始处理")
                if 'Query' not in df.columns:
                    df['Query'] = ''
        except Exception as e:
            print(f"警告: 读取输出文件失败 ({e})，将重新开始处理")
            if 'Query' not in df.columns:
                df['Query'] = ''
    else:
        # 添加新列（只添加Query列）
        if 'Query' not in df.columns:
            df['Query'] = ''
    
    # 确定处理范围
    if end_idx is None:
        end_idx = len(df)
    else:
        end_idx = min(end_idx, len(df))
    
    # 如果启用断点续传，找出需要处理的记录
    if resume:
        # 找出在指定范围内且未处理的记录
        need_process = []
        for idx in range(start_idx, end_idx):
            query = df.at[idx, 'Query'] if 'Query' in df.columns else ''
            # 如果Query为空，则需要处理
            query_str = '' if pd.isna(query) else str(query).strip()
            if query_str == '':
                need_process.append(idx)
        
        if len(need_process) == 0:
            print(f"\n所有记录（第 {start_idx+1} 到 {end_idx} 条）都已处理完成！")
            return df
        
        print(f"\n需要处理的记录: {len(need_process)} 条（在范围 {start_idx+1}-{end_idx} 中）")
        print(f"处理范围: 第 {need_process[0]+1} 到 {need_process[-1]+1} 条记录")
        indices_to_process = need_process
    else:
        print(f"处理范围: 第 {start_idx+1} 到 {end_idx} 条记录")
        indices_to_process = list(range(start_idx, end_idx))
    
    # 初始化实验跟踪器
    experiment_id = f"topic_extraction_{start_idx}_{end_idx}"
    tracker = ExperimentTracker(experiment_id, f"experiment_{experiment_id}.json")
    
    # 线程安全的锁和计数器
    df_lock = threading.Lock()
    stats_lock = threading.Lock()
    processed = 0
    failed = 0
    skipped = 0
    completed_count = 0
    
    def process_single_record(args):
        """处理单条记录的函数（用于多线程）"""
        nonlocal processed, failed, skipped, completed_count
        
        i, idx = args
        row = df.iloc[idx]
        title = row.get('display_name', 'N/A')
        
        # 再次检查是否已处理（双重保险，只检查Query列）
        with df_lock:
            query_existing = df.at[idx, 'Query'] if 'Query' in df.columns else ''
            if resume and pd.notna(query_existing) and str(query_existing).strip() != '':
                with stats_lock:
                    skipped += 1
                    completed_count += 1
                return {
                    'idx': idx,
                    'status': 'skipped',
                    'title': str(title)[:80],
                    'message': f"[{completed_count}/{len(indices_to_process)}] 跳过已处理"
                }
        
        # 提取topic和生成Query
        topic, query, stats = extract_topic_from_metadata(row, tracker)
        
        if topic and query:
            # 线程安全地更新DataFrame
            with df_lock:
                df.at[idx, 'Query'] = query
            
            with stats_lock:
                processed += 1
                completed_count += 1
            
            return {
                'idx': idx,
                'status': 'success',
                'title': str(title)[:80],
                'topic': topic,
                'query': query,
                'stats': stats,
                'message': f"[{completed_count}/{len(indices_to_process)}] 处理完成"
            }
        else:
            with stats_lock:
                failed += 1
                completed_count += 1
            return {
                'idx': idx,
                'status': 'failed',
                'title': str(title)[:80],
                'error': stats.get('error', 'Unknown error'),
                'message': f"[{completed_count}/{len(indices_to_process)}] 处理失败"
            }
    
    # 使用多线程处理
    print(f"\n使用 {num_threads} 个线程并行处理...")
    print("="*60)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_record, (i+1, idx)): (i+1, idx) 
            for i, idx in enumerate(indices_to_process)
        }
        
        # 处理完成的任务
        save_counter = 0
        for future in as_completed(future_to_idx):
            result = future.result()
            save_counter += 1
            
            # 打印结果
            if result['status'] == 'skipped':
                print(f"\n{result['message']}: {result['title']}...")
            elif result['status'] == 'success':
                print(f"\n{result['message']}: {result['title']}...")
                print(f"  - Topic: {result['topic']} (仅用于生成Query，不保存)")
                print(f"  - Query: {result['query']}")
                if 'stats' in result and 'cost' in result['stats']:
                    stats = result['stats']
                    print(f"  - Tokens: {stats.get('total_tokens', 0):,} | Cost: ${stats.get('cost', 0):.6f} | Time: {stats.get('runtime', 0):.2f}s")
            else:
                print(f"\n{result['message']}: {result['title']}...")
                print(f"  - 失败: {result.get('error', 'Unknown error')}")
            
            # 每batch_size条保存一次
            if save_counter % batch_size == 0:
                with df_lock:
                    df.to_csv(output_file, index=False)
                print(f"\n  [已保存进度到 {output_file}]")
    
    # 最终保存
    df.to_csv(output_file, index=False)
    print(f"\n处理完成！结果已保存到: {output_file}")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("处理统计")
    print("="*60)
    print(f"成功处理: {processed} 条")
    print(f"失败: {failed} 条")
    if skipped > 0:
        print(f"跳过已处理: {skipped} 条")
    
    final_stats = tracker.get_stats()
    print(f"\n实验统计:")
    print(f"  总调用次数: {final_stats['total_stats']['total_calls']}")
    print(f"  总输入Token: {final_stats['total_stats']['total_input_token']:,}")
    print(f"  总输出Token: {final_stats['total_stats']['total_output_token']:,}")
    print(f"  总Token: {final_stats['total_stats']['total_token']:,}")
    print(f"  总成本: ${final_stats['total_stats']['total_cost']:.6f}")
    print(f"  总成本: ¥{final_stats['total_stats']['total_cost'] * 7.2:.6f}")
    print(f"  统计数据已保存到: {tracker.output_file}")
    
    return df


if __name__ == "__main__":
    process_sampled_csv(
        input_file='OpenAlex_Review_2022_2025_citation_50.csv',
        output_file='OpenAlex_Review_2022_2025_citation_50_with_query.csv',
        start_idx=0,      # 从第0条开始（第一条）
        end_idx=None,     # None表示处理所有记录，可以设置为数字如10来只处理前10条
        batch_size=100,    # 每处理10条保存一次
        resume=True,      # 启用断点续传功能（如果输出文件已存在，会跳过已处理的记录）
        num_threads=10  
    )

