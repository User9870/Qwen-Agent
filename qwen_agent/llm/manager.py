import copy
from typing import Optional, Dict, Any, Union
from .base import LLM_REGISTRY
from .base import BaseChatModel

def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided:

    if 'azure_endpoint' in cfg:
        model_type = 'azure'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if '-audio' in model.lower():
        model_type = 'qwenaudio_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'qwen' in model.lower():
        model_type = 'qwen_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')

class LLMManager:
    """LLM实例的生命周期管理"""
    
    def __init__(self, initial_llm: Optional[Union[Dict, BaseChatModel]] = None):
        """初始化LLM管理器
        
        Args:
            initial_llm: 初始的LLM配置或实例
        """
        self._current_llm: Optional[BaseChatModel] = None
        self._llm_config: Optional[Dict] = None
        
        if initial_llm is not None:
            self.set_llm(initial_llm)
    
    def set_llm(self, llm: Union[Dict, BaseChatModel]) -> None:
        """设置当前LLM实例
        
        Args:
            llm: LLM配置字典或LLM实例
        """
        if isinstance(llm, dict):
            self._llm_config = llm.copy()
            self._current_llm = get_chat_model(llm)
        else:
            self._llm_config = None
            self._current_llm = llm
    
    def get_llm(self) -> Optional[BaseChatModel]:
        """获取当前LLM实例
        
        Returns:
            当前的LLM实例，如果未设置则返回None
        """
        return self._current_llm
    
    def get_llm_config(self) -> Optional[Dict]:
        """获取当前LLM配置
        
        Returns:
            当前的LLM配置，如果未设置则返回None
        """
        return self._llm_config
    
    def has_llm(self) -> bool:
        """检查是否已设置LLM
        
        Returns:
            如果已设置LLM返回True，否则返回False
        """
        return self._current_llm is not None
    
    def clear_llm(self) -> None:
        """清除当前LLM实例"""
        self._current_llm = None
        self._llm_config = None
    
    def switch_llm(self, llm_config: Dict) -> BaseChatModel:
        """切换到新的LLM实例
        
        Args:
            llm_config: 新的LLM配置
            
        Returns:
            新的LLM实例
        """
        self.set_llm(llm_config)
        return self._current_llm
    
    def call(self, *args, **kwargs) -> Any:
        """调用当前LLM的chat方法
        
        Args:
            *args: 传递给chat方法的位置参数
            **kwargs: 传递给chat方法的关键字参数
            
        Returns:
            LLM的响应结果
            
        Raises:
            ValueError: 如果未设置LLM实例
        """
        if not self.has_llm():
            raise ValueError("No LLM instance is set. Please set an LLM before calling it.")
        
        return self._current_llm.chat(*args, **kwargs)
    
    async def async_call(self, *args, **kwargs) -> Any:
        """异步调用当前LLM的chat方法
        
        Args:
            *args: 传递给chat方法的位置参数
            **kwargs: 传递给chat方法的关键字参数
            
        Returns:
            LLM的响应结果
            
        Raises:
            ValueError: 如果未设置LLM实例
        """
        if not self.has_llm():
            raise ValueError("No LLM instance is set. Please set an LLM before calling it.")
        
        # Check if LLM has async_chat method
        if hasattr(self._current_llm, 'async_chat'):
            return await self._current_llm.async_chat(*args, **kwargs)
        else:
            # Fallback to synchronous method
            return self._current_llm.chat(*args, **kwargs)