# AI CRM

![image-20261114162929910](C:\Users\IKUN\AppData\Roaming\Typora\typora-user-images\image-20261114162929910.png)



下面是上述产品工作流的伪代码，请自行实现内部函数功能

伪代码：

```python
#导入必要的库并配置环境
import necessary_libraries
configure_logging（）  #配置日志记录

#定义全局变量和配置信息
set_up_openai_api()  #设置OpenAI API
load_prompt_templates()  #加载提示模板
initialize_vectorstores()  #初始化向量存储

# 定义处理访谈的主函数
def SeaChatInterview(phone_number, query):
    # 连接到 SQLite 数据库并获取聊天历史
    connect_to_database()
    chat_history = fetch_chat_history(phone_number)

    # 判断是新用户还是已有用户
    if not chat_history:
        initialize_chat_history()  # 初始化聊天历史
        return initial_response()  # 返回初始问候语

    # 分析用户的查询，确定编号（意图）
    intent_code = detect_intent_code(query)  # 使用意图识别模型检测用户意图编号

    # 根据意图编号处理查询
    if intent_code == 1:  # 编号 1 代表 AI 职业咨询
        response = get_response_from_market_vectorstore(query, intent_code)  # 从市场向量存储获取响应
    elif intent_code == 2:  # 编号 2 代表 课程咨询
        response = get_response_from_course_vectorstore(query, intent_code)  # 从课程向量存储获取响应
    else:
        response = handle_other_intents(query, intent_code)  # 处理其他查询

    # 更新聊天历史并保存到数据库
    update_chat_history(chat_history, phone_number, response)  # 更新并存储聊天记录
    return response  # 返回响应

# 定义 Flask API 接口处理聊天请求
@app.route('/chat', methods=['POST'])
def chat_api():
    # 解析传入的 JSON 请求
    data = parse_request()  # 从请求中提取数据
    if not valid_request(data):  # 验证请求内容是否有效
        return error_response()  # 返回错误响应

    # 以流式方式返回响应给客户端
    return Response(stream_response(data), content_type='application/json')

# 启动 Flask 应用
if __name__ == '__main__':
    start_flask_app()  # 运行 Flask 应用
```



项目要求：

1.基于上述伪代码完成整体流程使程序可以正式运行

2.至少涉及一个function calling工具的定义

3.本地部署向量数据库，完成本地向量数据库创建与更新功能

4.创建本地sql数据库（sqlite比较稳定）并完成sqlagent的搭建

5.对外开放向量数据库更新接口，用户请求接口以完成整体系统的封装