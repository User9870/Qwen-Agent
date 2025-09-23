# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.tools import TOOL_REGISTRY, BaseTool, MCPManager
from qwen_agent.tools.base import ToolServiceError
from qwen_agent.tools.simple_doc_parser import DocParserError
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs

from qwen_agent.llm.manager import LLMManager
from qwen_agent.tools.manager import ToolManager


class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
        """
        self._llm_manager = LLMManager(llm)
        self._tool_manager = ToolManager(function_list)
        
        self.extra_generate_cfg: dict = {}
        self.system_message = system_message
        self.name = name
        self.description = description

    @property
    def llm(self) -> Optional[BaseChatModel]:
        return self._llm_manager.get_llm()
    
    @property
    def function_map(self) -> Dict[str, BaseTool]:
        return {name: self._tool_manager.get_tool(name) for name in self._tool_manager.list_tools()}

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                # Add the system instruction to the agent
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                # Already got system message in new_messages
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]
                else:
                    assert isinstance(new_messages[0][CONTENT], list)
                    assert new_messages[0][CONTENT][0].text
                    new_messages[0][CONTENT] = [ContentItem(text=self.system_message + '\n\n')
                                               ] + new_messages[0][CONTENT]  # noqa

        for rsp in self._run(messages=new_messages, **kwargs):
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            if _return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    @abstractmethod
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        return self._llm_manager.call(
            messages=messages,
            functions=functions,
            stream=stream,
            extra_generate_cfg=merge_generate_cfgs(
                base_generate_cfg=self.extra_generate_cfg,
                new_generate_cfg=extra_generate_cfg,
            )
        )

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        try:
            return self._tool_manager.call(tool_name, tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        self._tool_manager.add_tool(tool)

    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text

    def set_llm(self, llm: Union[dict, BaseChatModel]) -> None:
        """
        动态设置LLM实例
        
        Args:
            llm: LLM配置字典或LLM实例
        """
        self._llm_manager.set_llm(llm)
    
    def get_llm(self) -> Optional[BaseChatModel]:
        """获取当前LLM实例
        
        Returns:
            当前的LLM实例，如果未设置则返回None
        """
        return self._llm_manager.get_llm()
    
    def switch_llm(self, llm_config: Dict) -> BaseChatModel:
        """切换到新的LLM实例
        
        Args:
            llm_config: 新的LLM配置
            
        Returns:
            新的LLM实例
        """
        return self._llm_manager.switch_llm(llm_config)
    
    def add_tool(self, tool: Union[str, Dict, BaseTool]) -> None:
        """添加工具到Agent
        
        Args:
            tool: 工具名称、配置字典或工具实例
        """
        self._tool_manager.add_tool(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """从Agent中移除工具
        
        Args:
            tool_name: 要移除的工具名称
            
        Returns:
            如果成功移除返回True，如果工具不存在返回False
        """
        return self._tool_manager.remove_tool(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取指定名称的工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例，如果不存在返回None
        """
        return self._tool_manager.get_tool(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出所有已加载的工具名称
        
        Returns:
            工具名称列表
        """
        return self._tool_manager.list_tools()
    
    def get_tool_functions(self) -> List[Dict]:
        """获取所有工具的函数描述信息
        
        Returns:
            工具函数描述信息列表
        """
        return self._tool_manager.get_tool_functions()
    
    def replace_tool(self, tool_name: str, new_tool: Union[str, Dict, BaseTool]) -> bool:
        """替换指定名称的工具
        
        Args:
            tool_name: 要替换的工具名称
            new_tool: 新的工具（名称、配置或实例）
            
        Returns:
            如果成功替换返回True，如果原工具不存在返回False
        """
        return self._tool_manager.replace_tool(tool_name, new_tool)


# The most basic form of an agent is just a LLM, not augmented with any tool or workflow.
class BasicAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = {'lang': lang}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
