import os
import requests
from openai import AzureOpenAI, OpenAI
from typing import List, Dict, Union, Optional, Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量读取平台配置
API_KEY_MAP = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'siliconflow': os.getenv('SILICONFLOW_API_KEY'),
    'openrouter': os.getenv('OPENROUTER_API_KEY'),
    'bing': os.getenv('BING_API_KEY')  # Bing API 密钥
}

BASE_URL_MAP = {
    'openai': os.getenv('OPENAI_BASE_URL'),
    'siliconflow': os.getenv('SILICONFLOW_BASE_URL'),
    'openrouter': os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1',
}


class LLM:
    """
    简化的语言模型类，用于生成文本。
    """
    def __init__(self, model_name: str, platform: str = 'openai', api_key: Optional[str] = None):
        """
        初始化 LLM 实例。
        """
        self.model_name = model_name
        self.platform = platform
        self.api_key = api_key
        self.last_usage = None  # 存储最后一次 API 调用的 token 使用量
        
        self.client = self._init_client(platform)

    def _init_client(self, platform: str):
        """
        初始化 API 客户端。
        """
        assert platform in API_KEY_MAP, f"Platform {platform} is not supported."
        if self.api_key:
            api_key = self.api_key
        else:
            api_key = API_KEY_MAP.get(platform)

        assert api_key, f"API key for platform {platform} is not found in config or environment variables."
        base_url = BASE_URL_MAP.get(platform)
        
        if platform == 'openrouter':
            client = OpenAI(
                api_key=api_key,
                base_url=base_url or 'https://openrouter.ai/api/v1',
                timeout=120.0,
            )
            return client

        if platform == 'openai':
            # Azure OpenAI 配置
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version="2024-02-01",  # 使用一个较新的稳定版本
                timeout=120.0  # 设置120秒超时
            )
        else:
            # 其他平台配置 (如 SiliconFlow)
            client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                timeout=120.0  # 设置120秒超时
            )
        return client

    def _search_web(self, query: str) -> List[str]:
        """
        使用 Bing Web Search API 执行网页搜索。
        """
        api_key = API_KEY_MAP.get('bing')
        if not api_key:
            raise ValueError("Bing API key is missing.")
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
        params = {
            'q': query,
            'count': 5,  # 返回前 5 条搜索结果
            'mkt': 'zh-CN',  # 设置市场为中文（简体）
            'safesearch': 'Moderate'  # 设置安全搜索级别
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json().get('webPages', {}).get('value', [])
            return [item['url'] for item in results]
        else:
            raise Exception(f"Error fetching search results: {response.status_code}")

    def generate(
        self, 
        prompt: Union[List[Dict[str, Any]], str], 
        model: Optional[str] = None, 
        temperature: float = 1.0,
        web_search: bool = False
    ) -> str:
        """
        基于提示生成文本。
        """
        if model is None:
            model = self.model_name

        if isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        else:
            messages = prompt
        
        # 构建 API 请求参数
        api_params = {
            'model': model,
            'messages': messages,
            'temperature': temperature
        }

        # 如果启用网页搜索
        if web_search:
            query = prompt if isinstance(prompt, str) else prompt[0]['content']
            search_results = self._search_web(query)
            api_params['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': 'web_search',
                        'description': '执行网页搜索',
                        'parameters': {
                            'results': search_results
                        }
                    }
                }
            ]
        
        # 使用解包方式传入所有参数
        completion = self.client.chat.completions.create(**api_params)
        
        # 保存 token 使用量信息
        if hasattr(completion, 'usage') and completion.usage:
            self.last_usage = completion.usage

        return completion.choices[0].message.content or ""
