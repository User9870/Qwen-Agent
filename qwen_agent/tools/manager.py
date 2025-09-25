from typing import Dict, List, Union, Optional, Any

from .base import BaseTool, TOOL_REGISTRY
from .mcp_manager import MCPManager


class ToolManager:
    """工具管理类，负责工具的注册、管理和调用"""
    
    def __init__(self, initial_tools: Optional[List[Union[str, Dict, BaseTool]]] = None):
        """初始化工具管理器
        
        Args:
            initial_tools: 初始工具列表，可以是工具名称、配置字典或工具实例
        """
        self._tools: Dict[str, BaseTool] = {}
        
        if initial_tools:
            for tool in initial_tools:
                self.add_tool(tool)
    
    def add_tool(self, tool: Union[str, Dict, BaseTool]) -> None:
        """添加工具到管理器
        
        Args:
            tool: 工具名称、配置字典或工具实例
        """
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self._tools[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            # 处理MCP服务器配置
            mcp_tools = MCPManager().initConfig(tool)
            for mcp_tool in mcp_tools:
                tool_name = mcp_tool.name
                self._tools[tool_name] = mcp_tool
        else:
            # 处理工具名称或配置字典
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')
            
            self._tools[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)
    
    def remove_tool(self, tool_name: str) -> bool:
        """从管理器中移除工具
        
        Args:
            tool_name: 要移除的工具名称
            
        Returns:
            如果成功移除返回True，如果工具不存在返回False
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        获取指定名称的工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例，如果不存在返回None
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出所有已加载的工具名称
        
        Returns:
            工具名称列表
        """
        return list(self._tools.keys())
    
    def get_tool_functions(self) -> List[Dict]:
        """获取所有工具的函数描述信息
        
        Returns:
            工具函数描述信息列表
        """
        return [tool.function for tool in self._tools.values()]
    
    def has_tool(self, tool_name: str) -> bool:
        """检查是否存在指定名称的工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            如果存在返回True，否则返回False
        """
        return tool_name in self._tools
    
    def clear_tools(self) -> None:
        """清除所有工具"""
        self._tools.clear()
    
    def call(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Any:
        """
        调用指定工具
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            **kwargs: 其他参数
            
        Returns:
            工具调用结果
            
        Raises:
            ValueError: 如果工具不存在
        """
        if tool_name not in self._tools:
            raise ValueError(f'Tool {tool_name} does not exist.')
        
        tool = self._tools[tool_name]
        return tool.call(tool_args, **kwargs)
    
    async def async_call(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Any:
        """
        异步调用指定工具
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            **kwargs: 其他参数
            
        Returns:
            工具调用结果
            
        Raises:
            ValueError: 如果工具不存在
        """
        if tool_name not in self._tools:
            raise ValueError(f'Tool {tool_name} does not exist.')
        
        tool = self._tools[tool_name]
        # Check if tool has async_call method
        if hasattr(tool, 'async_call'):
            return await tool.async_call(tool_args, **kwargs)
        else:
            # Fallback to synchronous method
            return tool.call(tool_args, **kwargs)
    
    def replace_tool(self, tool_name: str, new_tool: Union[str, Dict, BaseTool]) -> bool:
        """替换指定名称的工具
        
        Args:
            tool_name: 要替换的工具名称
            new_tool: 新的工具（名称、配置或实例）
            
        Returns:
            如果成功替换返回True，如果原工具不存在返回False
        """
        if tool_name not in self._tools:
            return False
        
        self.remove_tool(tool_name)
        self.add_tool(new_tool)
        return True