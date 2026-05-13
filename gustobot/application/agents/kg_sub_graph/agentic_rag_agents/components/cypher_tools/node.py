from typing import Any, Callable, Coroutine, Dict, List
from pydantic import BaseModel, Field
# 导入必要的模块
from gustobot.application.agents.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from gustobot.infrastructure.core.logger import get_logger
from langchain_openai import ChatOpenAI
from gustobot.config import settings
from gustobot.application.safety.langgraph_bridge import evidence_from_payload
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.recipe_retriever import RecipeCypherRetriever
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.cypher_tools.utils import create_text2cypher_generation_node, create_text2cypher_validation_node, create_text2cypher_execution_node



# 获取日志记录器
logger = get_logger(service="cypher_tools")

# 定义GraphRAG查询的输入状态类型
class CypherQueryInputState(BaseModel):
    task: str
    query: str
    steps: List[str]

# 定义GraphRAG查询的输出状态类型
class CypherQueryOutputState(BaseModel):
    task: str
    query: str
    statement: str = ""
    parameters: Any = ""
    errors: List[str]
    records: Dict[str, Any]
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    steps: List[str]

# 定义GraphRAG API包装器

def create_cypher_query_node(
) -> Callable[
    [CypherQueryInputState],
    Coroutine[Any, Any, Dict[str, List[CypherQueryOutputState] | List[str]]],
]:
    """
    创建 Text2Cypher 查询节点，用于LangGraph工作流。

    返回
    -------
    Callable[[CypherQueryInputState], Dict[str, List[CypherQueryOutputState] | List[str]]]
        名为`cypher_query`的LangGraph节点。
    """

    async def cypher_query(
        state: Dict[str, Any],
    ) -> Dict[str, List[CypherQueryOutputState] | List[str]]:
        """
        执行Text2Cypher查询并返回结果。
        """
        errors = list()
        # 获取查询文本
        query = state.get("task", "")
        if not query:
            errors.append("未提供查询文本")
        # 使用 OpenAI 大模型执行查询/多跳/并行查询计划
        openai_kwargs: Dict[str, Any] = {
            "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0.7,
            "tags": ["research_plan"],
        }
        openai_api_key = getattr(settings, "OPENAI_API_KEY", None)
        if openai_api_key:
            openai_kwargs["openai_api_key"] = openai_api_key
        openai_api_base = getattr(settings, "OPENAI_API_BASE", None)
        if openai_api_base:
            openai_kwargs["openai_api_base"] = openai_api_base
        model = ChatOpenAI(**openai_kwargs)

        # 获取 Neo4j 图数据库连接
        try:
            neo4j_graph = get_neo4j_graph()
            logger.info("success to get Neo4j graph database connection")
        except Exception as e:
            logger.error(f"failed to get Neo4j graph database connection: {e}")

        # 创建自定义检索器实例，根据 Graph Schema 创建 Cypher 示例，用来引导大模型生成正确的 Cypher 查询语句
        cypher_retriever = RecipeCypherRetriever()

        # 根据自定义的 Cypher，引导大模型生成当前输入问题的 Cypher 查询语句
        try:
            neo4j_graph = get_neo4j_graph()
            logger.info("success to get Neo4j graph database connection")
        except Exception as e:
            logger.error(f"failed to get Neo4j graph database connection: {e}")
            errors.append(f"failed to get Neo4j graph database connection: {e}")
            # 返回错误状态而不是继续执行
            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": state.get("task", ""),
                            "query": query,
                            "statement": "",
                            "parameters": "",
                            "errors": errors,
                            "records": {"result": []},
                            "evidence": [],
                            "validation_warnings": errors,
                            "steps": state.get("steps", []),
                        }
                    )
                ],
                "steps": state.get("steps", []),
            }

        # 创建自定义检索器，根据 Graph Schema 创建 Cypher语句，用来引导大模型生成正确的 Cypher 查询语句
        cypher_retriever = RecipeCypherRetriever()

        # 根据自定义的 Cypher语句，引导大模型生成当前输入问题的 Cypher 查询语句
        cypher_generation = create_text2cypher_generation_node(
            llm=model, graph=neo4j_graph, cypher_example_retriever=cypher_retriever
        )

        cypher_result = await cypher_generation(state)
        cypher_statement = ""
        if isinstance(cypher_result, dict):
            cypher_statement = cypher_result.get("statement", "")
            generation_steps = cypher_result.get("steps", [])
            if generation_steps:
                steps = state.get("steps", list())
                steps.extend(step for step in generation_steps if step not in steps)
                state["steps"] = steps
        elif isinstance(cypher_result, str):
            cypher_statement = cypher_result
        else:
            cypher_statement = str(cypher_result or "")

        state["statement"] = cypher_statement
        #  TODO: Example 1. 直接使用大模型生成 Cypher 查询语句
        """
        # 安装依赖
        pip install neo4j-graphrag
        
        from neo4j_graphrag.retrievers import Text2CypherRetriever
        from neo4j_graphrag.llm import OpenAILLM
        import time
        import pandas as pd
        from neo4j import GraphDatabase

        NEO4J_URI="bolt://localhost"
        NEO4J_USERNAME="neo4j"
        NEO4J_PASSWORD="Snowball2019"
        NEO4J_DATABASE="neo4j"

        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )

        # 定义用户输入：
        examples = [
        "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
        ]

        # 初始化检索器
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=client,
            neo4j_schema=neo4j_schema,  # 可以通过 retrieve_and_parse_schema_from_graph_for_prompts 获取动态的Schema
            examples=examples,
        )

        
        # 执行检索：
        query_text = "muyu 都有哪些朋友？"
        print(retriever.search(query_text=query_text))
        """

        #  验证生成的 Cypher 查询语句是否正确
        validate_cypher = create_text2cypher_validation_node(
            llm=model,
            graph=neo4j_graph,
            llm_validation=True,
        )

        #  获取执行Cypher查询的全部信息
        execute_info = await validate_cypher(state)
        validation_warnings: List[str] = []
        if isinstance(execute_info, dict):
            validation_warnings.extend(str(item) for item in execute_info.get("errors", []) if item)
            next_action = execute_info.get("next_action_cypher")
            if next_action and next_action != "execute_cypher":
                validation_warnings.append(str(next_action))
        validation_warnings.extend(str(item) for item in state.get("errors", []) if item)

        #  执行 Cypher 查询语句
        execute_cypher = create_text2cypher_execution_node(
            graph=neo4j_graph, cypher=execute_info
        )

        final_result = await execute_cypher(state)
        records_payload = (
            {"result": final_result["cyphers"][0]["records"]}
            if final_result.get("cyphers") and len(final_result["cyphers"]) > 0
            else {"result": []}
        )
        raw_records = records_payload.get("result")
        has_records = bool(raw_records)
        if isinstance(raw_records, list) and all(
            isinstance(row, dict) and row.get("error") for row in raw_records
        ):
            has_records = False
        evidence = (
            evidence_from_payload(
                {
                    "cypher_records": records_payload,
                    "tool_outputs": {
                        "task": state.get("task", ""),
                        "statement": cypher_statement,
                        "records": records_payload,
                    },
                },
                route="graphrag-query",
            )
            if has_records
            else []
        )

        # 封装 单次子任务执行的 输出结果并通过Pydantic模型限定格式
        return {
            "cyphers": [
                CypherQueryOutputState(
                        **{
                            "task": state.get("task", ""),
                            "query": query,
                            "statement": cypher_statement,
                            "parameters": state.get("parameters") or {},
                            "errors": errors,
                            "records": records_payload,
                            "evidence": evidence,
                            "validation_warnings": validation_warnings,
                            "steps": ["execute_cypher_query"],
                        }
                    )
                ],
                "steps": ["execute_cypher_query"],
            }
  
    return cypher_query
