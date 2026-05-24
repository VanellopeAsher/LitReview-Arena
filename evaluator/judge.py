"""
Evaluator module for the Evaluator Agent

Implements LLM calling, prompt building, and result parsing.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from loguru import logger
from tqdm import tqdm

try:
    from .config import DIMENSIONS, OUTCOMES, MAX_RETRIES, API_CALL_DELAY, RATE_LIMIT_RETRIES, RATE_LIMIT_BASE_DELAY, CHECKPOINT_INTERVAL
    from .context_builder import ContextBuilder
    from .context_cache import (
        append_cache_record,
        expert_outcomes_fingerprint,
        load_cache_index,
        make_cache_record,
        retrieval_settings_fingerprint,
        topic_fingerprint,
        try_get_cached,
    )
    from .naive_judge import NaiveContextBuilder
except ImportError:
    from evaluator.config import DIMENSIONS, OUTCOMES, MAX_RETRIES, API_CALL_DELAY, RATE_LIMIT_RETRIES, RATE_LIMIT_BASE_DELAY, CHECKPOINT_INTERVAL
    from evaluator.context_builder import ContextBuilder
    from evaluator.context_cache import (
        append_cache_record,
        expert_outcomes_fingerprint,
        load_cache_index,
        make_cache_record,
        retrieval_settings_fingerprint,
        topic_fingerprint,
        try_get_cached,
    )
    from evaluator.naive_judge import NaiveContextBuilder


def _suggestion_text_for_diversity(raw_response: str) -> str:
    """固定、可复现：下游 diversity 分析使用；默认使用完整 judge 原文。"""
    return (raw_response or "").strip()


class Evaluator:
    """
    评估器，负责调用LLM进行评估并解析结果
    """
    
    def __init__(
        self,
        context_builder: ContextBuilder,
        llm=None,
        model_name: str = "deepseek-ai/DeepSeek-V3.2",
        platform: str = "siliconflow",
        expert_outcomes_map: Optional[Dict[str, Dict[str, str]]] = None,
        context_cache_path: Optional[Path] = None,
        context_cache_write: bool = True,
    ):
        """
        初始化Evaluator
        
        Args:
            context_builder: ContextBuilder实例
            llm: LLM实例（如果为None则创建新的）
            model_name: LLM模型名称
            platform: LLM平台
            expert_outcomes_map: 可选，{battle_id: {D1: ...}}，来自 expert_outcomes.jsonl
            context_cache_path: 可选 JSONL，命中则跳过 build_context（省检索侧 token）
            context_cache_write: 是否将新构建的上下文追加写入缓存文件
        """
        self.context_builder = context_builder
        self.expert_outcomes_map = expert_outcomes_map or {}
        self.context_cache_path = Path(context_cache_path) if context_cache_path else None
        self.context_cache_write = context_cache_write
        self._context_cache_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._context_cache_lock = Lock()

        # 导入LLM类（evaluator/utils.py）
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import LLM

        self.llm = llm or LLM(model_name=model_name, platform=platform)
        self.model_name = model_name

    def _litjudge_uses_context_cache(self) -> bool:
        if not self.context_cache_path:
            return False
        return not isinstance(self.context_builder, NaiveContextBuilder)

    def _get_context_cache_index(self) -> Dict[str, Dict[str, Any]]:
        if self._context_cache_index is not None:
            return self._context_cache_index
        with self._context_cache_lock:
            if self._context_cache_index is None:
                self._context_cache_index = (
                    load_cache_index(self.context_cache_path)
                    if self.context_cache_path
                    else {}
                )
            return self._context_cache_index
    
    def parse_result(self, response_text: str) -> Optional[Dict[str, str]]:
        """
        解析LLM返回的结果
        
        Args:
            response_text: LLM返回的文本
        
        Returns:
            解析后的结果字典 {dimension: outcome}，如果解析失败返回None
        """
        # 尝试提取JSON
        json_match = re.search(r'\{[^{}]*"D5"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                # 验证结果格式
                if self._validate_result(result):
                    return result
            except json.JSONDecodeError:
                pass
        
        # 尝试提取所有维度
        result = {}
        for dim in DIMENSIONS.keys():
            # 查找 "D1"–"D5": "A" 这样的模式
            pattern = rf'"{dim}"\s*:\s*"([^"]+)"'
            match = re.search(pattern, response_text)
            if match:
                outcome = match.group(1).strip()
                if outcome in OUTCOMES:
                    result[dim] = outcome
        
        if self._validate_result(result):
            return result
        
        return None
    
    def _validate_result(self, result: Dict[str, str]) -> bool:
        """
        验证结果格式是否正确
        
        Args:
            result: 结果字典
        
        Returns:
            是否有效
        """
        # 检查是否包含所有维度
        if set(result.keys()) != set(DIMENSIONS.keys()):
            return False
        
        # 检查每个outcome是否有效
        for dim, outcome in result.items():
            if outcome not in OUTCOMES:
                return False
        
        return True
    
    def evaluate_battle(
        self,
        battle: Dict,
        max_retries: int = MAX_RETRIES
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict]]:
        """
        评估单个battle，包含完善的网络错误处理和重试机制
        
        Returns:
            (record, stats_dict)
            record: 含 D1–D5、raw_judge_response、suggestion_text_for_diversity、retrieval_metrics 等
        """
        battle_id = battle.get("battle_id")
        logger.debug(f"评估battle: {battle_id}")
        
        # 获取expert outcomes（如果可用，用于demonstrations）
        expert_outcomes = None
        if self.expert_outcomes_map:
            expert_outcomes = dict(self.expert_outcomes_map)
        elif hasattr(self.context_builder, 'all_battles'):
            try:
                from .aggregator import load_expert_outcomes_from_battles
            except ImportError:
                from evaluator.aggregator import load_expert_outcomes_from_battles
            expert_outcomes = load_expert_outcomes_from_battles(self.context_builder.all_battles)

        cache_hit = False
        if self._litjudge_uses_context_cache():
            idx = self._get_context_cache_index()
            cached = try_get_cached(idx, battle, self.context_builder, expert_outcomes)
            if cached:
                context, retrieval_metrics = cached
                cache_hit = True
                logger.info(f"Context cache hit: {battle_id}")

        if not cache_hit:
            context, retrieval_metrics = self.context_builder.build_context(
                battle, expert_outcomes=expert_outcomes
            )
            if self._litjudge_uses_context_cache() and self.context_cache_write:
                rec = make_cache_record(
                    str(battle_id),
                    topic_fingerprint(battle),
                    retrieval_settings_fingerprint(self.context_builder),
                    expert_outcomes_fingerprint(expert_outcomes),
                    context,
                    retrieval_metrics,
                )
                append_cache_record(self.context_cache_path, rec, self._context_cache_lock)
                with self._context_cache_lock:
                    if self._context_cache_index is None:
                        self._context_cache_index = {}
                    self._context_cache_index[str(battle_id)] = rec
        
        # Rate limit重试计数器（独立于普通重试）
        rate_limit_retries_left = RATE_LIMIT_RETRIES
        retries_left = max_retries
        
        # 调用LLM
        while retries_left > 0 or rate_limit_retries_left > 0:
            try:
                # 构建messages
                messages = [
                    {"role": "system", "content": "You are an expert evaluator for literature reviews. You must provide judgments as JSON."},
                    {"role": "user", "content": context}
                ]
                
                # 调用LLM
                response_text = self.llm.generate(
                    prompt=messages,
                    temperature=0.0,
                    web_search=False
                )
                
                # 解析结果
                result = self.parse_result(response_text)
                
                if result:
                    stats = {}
                    if hasattr(self.llm, 'last_usage') and self.llm.last_usage:
                        usage = self.llm.last_usage
                        stats = {
                            'input_tokens': usage.prompt_tokens,
                            'output_tokens': usage.completion_tokens,
                            'total_tokens': usage.total_tokens
                        }

                    record: Dict[str, Any] = {
                        **result,
                        "raw_judge_response": response_text,
                        "suggestion_text_for_diversity": _suggestion_text_for_diversity(response_text),
                        "retrieval_metrics": retrieval_metrics,
                    }
                    logger.debug(f"Battle {battle_id} 评估成功")
                    return record, stats
                else:
                    logger.warning(f"Battle {battle_id} 结果解析失败，尝试 {max_retries - retries_left + 1}/{max_retries}")
                    if retries_left > 0:
                        retries_left -= 1
                        if retries_left > 0:
                            time.sleep(API_CALL_DELAY * (max_retries - retries_left))
                        continue
            
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                # 检测网络相关错误
                is_network_error = (
                    "timeout" in error_str.lower() or
                    "timed out" in error_str.lower() or
                    "connection" in error_str.lower() or
                    "network" in error_str.lower() or
                    "socket" in error_str.lower() or
                    "ConnectionError" in error_type or
                    "TimeoutError" in error_type or
                    "ConnectTimeout" in error_type or
                    "ReadTimeout" in error_type
                )
                
                # 检测rate limit错误
                is_rate_limit = (
                    "429" in error_str or
                    "rate limit" in error_str.lower() or
                    "rate limiting" in error_str.lower() or
                    "TPM limit" in error_str or
                    "quota" in error_str.lower() or
                    "too many requests" in error_str.lower()
                )
                
                # 检测服务器错误（5xx）
                is_server_error = (
                    "500" in error_str or
                    "502" in error_str or
                    "503" in error_str or
                    "504" in error_str or
                    "internal server error" in error_str.lower() or
                    "bad gateway" in error_str.lower() or
                    "service unavailable" in error_str.lower()
                )
                
                # Rate limit处理（使用指数退避）
                if is_rate_limit and rate_limit_retries_left > 0:
                    delay = RATE_LIMIT_BASE_DELAY * (2 ** (RATE_LIMIT_RETRIES - rate_limit_retries_left))
                    logger.warning(
                        f"Battle {battle_id}: 遇到速率限制，等待 {delay:.1f}秒后重试 "
                        f"(剩余重试次数: {rate_limit_retries_left}/{RATE_LIMIT_RETRIES})"
                    )
                    time.sleep(delay)
                    rate_limit_retries_left -= 1
                    continue  # 继续重试，不减少retries_left
                
                # 网络错误处理（使用指数退避）
                elif is_network_error and retries_left > 0:
                    delay = API_CALL_DELAY * (2 ** (max_retries - retries_left))
                    logger.warning(
                        f"Battle {battle_id}: 网络错误 ({error_type}): {error_str[:100]}，"
                        f"等待 {delay:.1f}秒后重试 (剩余重试次数: {retries_left}/{max_retries})"
                    )
                    time.sleep(delay)
                    retries_left -= 1
                    continue
                
                # 服务器错误处理（使用指数退避）
                elif is_server_error and retries_left > 0:
                    delay = API_CALL_DELAY * (2 ** (max_retries - retries_left)) * 2  # 服务器错误等待更久
                    logger.warning(
                        f"Battle {battle_id}: 服务器错误 ({error_type}): {error_str[:100]}，"
                        f"等待 {delay:.1f}秒后重试 (剩余重试次数: {retries_left}/{max_retries})"
                    )
                    time.sleep(delay)
                    retries_left -= 1
                    continue
                
                # 其他错误
                else:
                    logger.error(f"Battle {battle_id}: API调用失败 ({error_type}): {e}", exc_info=True)
                    if is_rate_limit:
                        logger.error(f"Battle {battle_id}: 速率限制重试次数用尽，跳过该battle")
                        return None, None
                    elif retries_left > 0:
                        retries_left -= 1
                        logger.warning(f"Battle {battle_id}: 重试API调用 (剩余重试次数: {retries_left}/{max_retries})")
                        time.sleep(API_CALL_DELAY * (max_retries - retries_left))
                        continue
                    else:
                        logger.error(f"Battle {battle_id}: API调用失败且重试次数用尽，跳过该battle")
                        return None, None
        
        logger.error(f"Battle {battle_id} 评估失败，已达到最大重试次数")
        return None, None
    
    def _save_checkpoint(self, checkpoint_file: str, results: Dict, stats: Dict):
        """保存checkpoint文件"""
        try:
            checkpoint_data = {
                "judge_outcomes": results,
                "stats": stats,
                "checkpoint_timestamp": time.time()
            }
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Checkpoint saved: {len(results)} results")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def evaluate_battles(
        self,
        battles: List[Dict],
        progress_callback: Optional[callable] = None,
        existing_results: Optional[Dict[str, Dict[str, Any]]] = None,
        show_progress: bool = True,
        max_workers: int = 1,
        checkpoint_file: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict]]:
        """
        批量评估battles，支持断点重续、进度条和并行处理
        
        Returns:
            results_dict: {battle_id: 含 D1–D5、raw_judge_response、suggestion_text_for_diversity、retrieval_metrics 等}
        """
        results = existing_results.copy() if existing_results else {}
        stats = {}
        
        total = len(battles)
        skipped = len(results)
        
        # 过滤出需要评估的battles
        battles_to_evaluate = []
        for battle in battles:
            battle_id = battle.get("battle_id")
            if battle_id not in results:
                battles_to_evaluate.append(battle)
        
        if max_workers == 1:
            # 串行处理（原有逻辑）
            logger.info(f"开始批量评估 {total} 个battles (串行模式)")
            if skipped > 0:
                logger.info(f"发现 {skipped} 个已有结果，将跳过")
            
            pbar = None
            if show_progress:
                pbar = tqdm(
                    total=total,
                    initial=skipped,
                    desc="Evaluating battles",
                    unit="battle",
                    ncols=100
                )
            
            try:
                for i, battle in enumerate(battles_to_evaluate, 1):
                    battle_id = battle.get("battle_id")
                    result, battle_stats = self.evaluate_battle(battle)
                    
                    if result:
                        results[battle_id] = result
                        if battle_stats:
                            stats[battle_id] = battle_stats
                    
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': len(results),
                            'failed': i - len(results)
                        })
                    
                    if progress_callback:
                        progress_callback(len(results), total)
                    
                    # 定期保存checkpoint（串行模式）
                    if checkpoint_file and CHECKPOINT_INTERVAL > 0 and i % CHECKPOINT_INTERVAL == 0:
                        self._save_checkpoint(checkpoint_file, results, stats)
                        logger.info(f"Checkpoint saved: {len(results)}/{total} battles completed")
                    
                    if i < len(battles_to_evaluate):
                        time.sleep(API_CALL_DELAY)
            finally:
                if pbar:
                    pbar.close()
        else:
            # 并行处理
            logger.info(f"开始批量评估 {total} 个battles (并行模式，{max_workers}个并发)")
            if skipped > 0:
                logger.info(f"发现 {skipped} 个已有结果，将跳过")
            
            # 线程安全的锁和结果字典
            results_lock = Lock()
            completed_count = [0]  # 使用列表以便在闭包中修改
            last_checkpoint_count = [0]  # 上次checkpoint时的完成数量
            
            pbar = None
            if show_progress:
                pbar = tqdm(
                    total=total,
                    initial=skipped,
                    desc=f"Evaluating battles ({max_workers} workers)",
                    unit="battle",
                    ncols=100
                )
            
            def evaluate_single_battle(battle):
                """评估单个battle的包装函数"""
                battle_id = battle.get("battle_id")
                try:
                    result, battle_stats = self.evaluate_battle(battle)
                    
                    with results_lock:
                        if result:
                            results[battle_id] = result
                            if battle_stats:
                                stats[battle_id] = battle_stats
                        completed_count[0] += 1
                        
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({
                                'success': len(results),
                                'failed': completed_count[0] - len(results)
                            })
                        
                        if progress_callback:
                            progress_callback(len(results), total)
                        
                        # 定期保存checkpoint（并行模式）
                        if checkpoint_file and CHECKPOINT_INTERVAL > 0:
                            current_count = completed_count[0]
                            if current_count - last_checkpoint_count[0] >= CHECKPOINT_INTERVAL:
                                self._save_checkpoint(checkpoint_file, results, stats)
                                last_checkpoint_count[0] = current_count
                                logger.info(f"Checkpoint saved: {len(results)}/{total} battles completed")
                    
                    return battle_id, result, battle_stats
                except Exception as e:
                    logger.error(f"Error evaluating battle {battle_id}: {e}")
                    with results_lock:
                        completed_count[0] += 1
                        if pbar:
                            pbar.update(1)
                    return battle_id, None, None
            
            try:
                # 使用ThreadPoolExecutor进行并行处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_battle = {
                        executor.submit(evaluate_single_battle, battle): battle 
                        for battle in battles_to_evaluate
                    }
                    
                    # 等待所有任务完成
                    for future in as_completed(future_to_battle):
                        battle = future_to_battle[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Exception in battle {battle.get('battle_id')}: {e}")
            finally:
                if pbar:
                    pbar.close()
        
        logger.info(f"批量评估完成，成功 {len(results)}/{total} (跳过 {skipped} 个)")
        
        return results, stats
