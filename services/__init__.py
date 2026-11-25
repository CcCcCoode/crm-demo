"""业务服务层。

本层负责把「数据库 / 向量库 / LLM / 工具」组合成完整业务流程：

- ``services.user_info``：用户画像读取、创建、抽取与更新；
- ``services.rag``：按意图调用 Function Calling 工具检索网络与向量库资料；
- ``services.chat``：主流程编排（欢迎语 / 意图识别 / 检索 / 回答 / 落库 / 画像更新 / 流式输出）。

所有服务都以「异常不外抛」为原则：内部记录日志并返回友好结构，由 ``app.py`` 转为 HTTP 响应。
"""

from __future__ import annotations

__all__ = ["chat", "rag", "user_info"]
